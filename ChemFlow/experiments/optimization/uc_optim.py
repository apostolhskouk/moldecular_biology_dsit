import torch
import numpy as np
import pandas as pd
from accelerate.utils import set_seed

from absl import logging
import os
from collections import defaultdict
from pandarallel import pandarallel
from tap import Tap
from tqdm import trange
from typing import Literal
import bisect

from pathlib import Path



from ChemFlow.experiments.utils.traversal_step import Traversal
from ChemFlow.src.utils.scores import *
from ChemFlow.experiments.utils.utils import partitionIndexes

PROTEIN_FILES = {
    "1err": "ChemFlow/data/raw/1err/1err.maps.fld",
    "2iik": "ChemFlow/data/raw/2iik/2iik.maps.fld",
}


class Args(Tap):
    prop: str = "plogp"  # property to optimize
    n: int = 100_000  # number of molecules to generate
    steps: int = 10  # number of optimization steps
    batch_size: int = 10_000  # batch size
    method: Literal[
        "random",
        "random_1d",
        "fp",
        "limo",
        "chemspace",
        "wave_sup",
        "wave_unsup",
        "hj_sup",
        "hj_unsup",
        "neural_ode",
        "neural_ode_unsup",
        "latent_stepper",
        "hybrid_sup_unsup",
        "rl_policy"
    ] = "random"  # optimization method
    rl_action_scale: float = 0.05
    step_size: float = 0.1  # step size
    relative: bool = True  # relative step size
    data_name: str = "zmc"  # data name
    seed: int = 42
    binding_affinity: bool = False
    limo: bool = False
    alpha_hybrid: float = 0.5 # Weight for supervised component
    hybrid_unsup_pde_type: Literal["wave", "hj"] = "wave" # PDE type for unsupervised part
    hybrid_unsup_k_idx: int = None
    
    def process_args(self):
        self.model_name = self.prop + "_" + self.method
        self.model_name += f"_{self.step_size}"
        self.model_name += "_relative" if self.relative else "_absolute"
        if self.method == "rl_policy":
             self.model_name += f"_ascale{self.rl_action_scale}"
        
if __name__ == "__main__":
    args = Args().parse_args()

    logging.set_verbosity(logging.INFO)
    n_workers = torch.cuda.device_count() if args.binding_affinity else os.cpu_count()
    # pandarallel with split the dataframe into n_cpus chunks
    chunk_idx = list(partitionIndexes(args.n, n_workers))[1:]
    pandarallel.initialize(nb_workers=n_workers, progress_bar=False, verbose=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(args.seed)
    traversal = Traversal(
        method=args.method,
        prop=args.prop,
        data_name=args.data_name,
        step_size=args.step_size,
        relative=args.relative,
        minimize=args.prop in MINIMIZE_PROPS or args.binding_affinity,
        device=device,
        # k_idx is for the main unsupervised methods, not directly for hybrid's unsup part here
        # k_idx=None, # Or pass if method is wave_unsup/hj_unsup
        alpha_hybrid=args.alpha_hybrid,
        hybrid_unsup_pde_type=args.hybrid_unsup_pde_type,
        hybrid_unsup_k_idx=args.hybrid_unsup_k_idx,
    )
    set_seed(args.seed)

    z0 = torch.randn(args.n, traversal.vae.latent_dim)

    print(f"prop: {args.prop}")

    all_final_smiles = []
    for i in trange(0, args.n, args.batch_size, desc="Optimizing Batches"):
        z_batch = z0[i : i + args.batch_size].to(device)
        
        # Optimizer for LIMO is not used in this hybrid setup directly
        # if args.method == "limo" and args.limo:
        #     z_batch.requires_grad = True
        #     optimizer = torch.optim.Adam([z_batch], lr=args.step_size) # This was for a specific LIMO variant

        for t in range(args.steps):
            # The LIMO optimizer logic was specific and not part of the general step
            # if args.method == "limo" and args.limo:
            #     traversal.step(z_batch, t + 1, optimizer=optimizer) # This was for a specific LIMO variant
            #     continue 
            u_z_batch = traversal.step(z_batch, t + 1) # t+1 because step 0 is initial state
            with torch.no_grad(): # Ensure operations on z_batch don't track gradients if not needed
                z_batch += u_z_batch
        
        with torch.no_grad():
            # Decode final z states for the batch
            # vae.decode returns log_softmax, .exp() is not needed if dm.decode handles log_probs
            final_x_batch_log_probs = traversal.vae.decode(z_batch)
        all_final_smiles.extend(traversal.dm.decode(final_x_batch_log_probs))

    df = pd.DataFrame(all_final_smiles, columns=["smiles"])
    df_unique = df.drop_duplicates().reset_index(drop=True) # Reset index for parallel_apply name access

    # Update chunk_idx for df_unique if its size changed significantly
    if df_unique.shape[0] > 0 and n_workers > 0:
         chunk_idx = list(partitionIndexes(df_unique.shape[0], n_workers))[1:]
    else:
         chunk_idx = []


    def func(_x: pd.Series): # _x is a row from df_unique
        smiles_str = _x["smiles"]
        if args.binding_affinity:
            # device_idx logic might need adjustment if df_unique has different indexing
            # Assuming _x.name is the new index (0 to len(df_unique)-1)
            device_idx = 0 
            if chunk_idx: # If there are chunks (i.e. n_workers > 0 and data)
                 device_idx = bisect.bisect_right(chunk_idx, _x.name) % torch.cuda.device_count() if torch.cuda.is_available() else 0

            _x[args.prop] = smiles2affinity(smiles_str, PROTEIN_FILES.get(args.prop), device_idx=device_idx)
        else:
            _x[args.prop] = PROP_FN[args.prop](smiles_str)
        return _x

    print(f"Calculating properties for {len(df_unique)} unique SMILES, this may take a while...")
    if not df_unique.empty:
        df_unique = df_unique.parallel_apply(func, axis=1)
        # Merge unique properties back to the original df which has all (potentially duplicate) final SMILES
        df = df.merge(df_unique, on="smiles", how="left")
    else:
        df[args.prop] = np.nan # Add empty column if no unique smiles

    output_path_dir = Path("ChemFlow/data/interim/uc_optim") # Renamed for clarity
    output_path_dir.mkdir(parents=True, exist_ok=True)
    output_file_path = output_path_dir / f"{args.model_name}.csv"
    df.to_csv(output_file_path, index=False) # Save without index

    print(f"Results saved to {output_file_path}")