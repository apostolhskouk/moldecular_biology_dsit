import torch
import pandas as pd
from accelerate.utils import set_seed

from absl import logging
import os
from pandarallel import pandarallel
from tap import Tap
from tqdm import trange
from typing import Literal
from pathlib import Path


from ChemFlow.experiments.utils.traversal_step import Traversal
from ChemFlow.src.utils.scores import *


class Args(Tap):
    prop: str = "plogp"  # property to optimize
    n: int = 800  # number of molecules to generate
    steps: int = 1000  # number of optimization steps
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
        "neural_ode"
    ] = "random"  # optimization method
    step_size: float = 0.1  # step size
    relative: bool = False  # relative step size
    data_name: str = "zmc"  # data name

    def process_args(self):
        self.model_name = self.prop + "_" + self.method
        self.model_name += f"_{self.step_size}"
        self.model_name += "_relative" if self.relative else "_absolute"


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    pandarallel.initialize(nb_workers=os.cpu_count(), progress_bar=False, verbose=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(42)

    args = Args().parse_args()
    traversal = Traversal(
        method=args.method,
        prop=args.prop,
        data_name=args.data_name,
        step_size=args.step_size,
        relative=args.relative,
        minimize=args.prop in MINIMIZE_PROPS,
        device=device,
    )

    # Always read zinc250k for optimization
    df = pd.read_csv("ChemFlow/data/interim/props/zinc250k.csv")[["smiles", args.prop]]
    df = df.sort_values(args.prop, ascending=args.prop not in MINIMIZE_PROPS)

    smiles_raw = df["smiles"].astype(str).tolist()
    smiles = [s.strip() for s in smiles_raw[: args.n]]
    x = traversal.dm.encode(smiles).to(device)
    z, *_ = traversal.vae.encode(x)

    del smiles, x, df

    optimizer = None
    if args.method == "limo":
        z = z.clone().detach()
        z.requires_grad = True
        optimizer = torch.optim.Adam([z], lr=args.step_size)

    print(f"prop: {args.prop}")
    results = []

    for t in trange(args.steps):
        u_z = traversal.step(z, t)
        if args.method == "limo":
            traversal.step(z, t + 1, optimizer=optimizer)
        else:
            z += u_z
        smiles = traversal.dm.decode(traversal.vae.decode(z))
        results.extend(
            {
                "idx": i,
                "t": t,
                "smiles": s,
            }
            for i, s in enumerate(smiles)
        )

    df = pd.DataFrame(results)
    df_unique = df[["smiles"]].drop_duplicates("smiles")

    def func(_x: pd.Series):
        _x[args.prop] = PROP_FN[args.prop](_x["smiles"])
        return _x

    print("Calculating properties, this may take a while...")
    df_unique = df_unique.parallel_apply(func, axis=1)

    # merge with original df to get the original order
    df = df.merge(df_unique, on="smiles", how="left")

    # save results to csv
    output_path = Path("ChemFlow/data/interim/optimization")
    output_path.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path / f"{args.model_name}.csv")

    print(f"Results saved to {output_path / args.model_name}.csv")