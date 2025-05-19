# train prop uses 110K data points, generating them and compute the properties on the
# fly is too slow, so we generate them and save them in a csv file
import bisect
import os
#set only one gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import pandas as pd
import numpy as np
from pandarallel import pandarallel
from ChemFlow.src.vae.utils import load_vae
from ChemFlow.src.utils.scores import *
from ChemFlow.src.vae.datamodule import NOP
import torch
from tqdm import tqdm, trange
from tap import Tap

from pathlib import Path
from rdkit import Chem

from accelerate.utils import set_seed

from cd2root import cd2root

cd2root()

from src.utils.scores import *
from experiments.utils.utils import partitionIndexes


class Args(Tap):
    smiles_file: str = "data/processed/zmc.smi"
    vae_path: str = "checkpoints/vae/zmc/checkpoint.pt"
    seed: int = 42
    n: int = 110_000
    batch_size: int = 10_000
    binding_affinity: bool = False


    def process_args(self):
        self.name = f"prop_predictor_latent_z_{self.n}_seed{self.seed}"
        if self.binding_affinity:
            self.name += "_binding_affinity"



PROPS = ["plogp", "uplogp", "qed", "sa", "gsk3b", "jnk3", "drd2"]
PROTEIN_FILES = {
    "1err": "data/raw/1err/1err.maps.fld",
    "2iik": "data/raw/2iik/2iik.maps.fld",
}


def make_latent_dataset(): # Renamed function for clarity
    dm, vae = load_vae(
        file_path=args.smiles_file,
        model_path=args.vae_path,
        device=device # Pass device to load_vae
    )
    # Ensure VAE is in eval mode and no gradients are computed for it here
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False

    set_seed(args.seed)

    # Generate initial SMILES to encode.
    # We need a source of diverse SMILES to get diverse z vectors.
    # Option 1: Use a subset of the training data for the VAE.
    # This requires dm.dataset to be populated, so call dm.setup()
    dm.prepare_data() # Ensure dataset is loaded/created if not already
    dm.setup()
    
    if len(dm.dataset) < args.n:
        print(f"Warning: Requested {args.n} samples, but VAE dataset has only {len(dm.dataset)}. Using all available.")
        num_samples_to_take = len(dm.dataset)
        all_indices = torch.randperm(len(dm.dataset))[:num_samples_to_take]
    else:
        num_samples_to_take = args.n
        all_indices = torch.randperm(len(dm.dataset))[:num_samples_to_take]


    all_z_vectors = torch.zeros(num_samples_to_take, vae.latent_dim, device='cpu') # Store on CPU to save GPU memory
    all_smiles_for_props = []


    print(f"Encoding {num_samples_to_take} molecules to get latent vectors z...")
    with torch.no_grad():
        for i in trange(0, num_samples_to_take, args.batch_size):
            batch_indices = all_indices[i : i + args.batch_size]
            # Correctly get items from MolDataset which are already tensors
            batch_smiles_indices = [dm.dataset.dataset[idx] for idx in batch_indices]
            
            # Pad and stack if necessary, assuming dm.dataset items are lists of token IDs
            max_len_batch = max(len(s) for s in batch_smiles_indices)
            padded_batch_tokens = []
            for s_tokens in batch_smiles_indices:
                padding_needed = max_len_batch - len(s_tokens)
                padded_tokens = s_tokens + [dm.dataset.token_to_id[NOP]] * padding_needed
                padded_batch_tokens.append(padded_tokens)

            input_tokens_tensor = torch.tensor(padded_batch_tokens, dtype=torch.long).to(device)
            
            z_batch, _, _ = vae.encode(input_tokens_tensor)
            all_z_vectors[i : i + z_batch.size(0)] = z_batch.cpu()
            
            # Decode back to SMILES for property calculation
            # We need to decode the z we just got, not the original input SMILES,
            # to ensure properties are calculated for what the z represents.
            reconstructed_x_log_probs = vae.decode(z_batch) # This is log_softmax output
            reconstructed_x_probs = reconstructed_x_log_probs.exp() # Convert to probabilities if needed by dm.decode
                                                                # or pass log_probs if dm.decode handles it.
                                                                # Assuming dm.decode expects probabilities or argmax internally
            
            # The dm.decode method expects a tensor of shape (batch, max_len * vocab_size)
            # or (batch, max_len, vocab_size) and then it argmaxes.
            # Let's ensure the input to dm.decode is what it expects.
            # If vae.decode returns (batch, max_len * vocab_size) directly, it's fine.
            # If it returns (batch, max_len, vocab_size), it's also fine as dm.decode handles it.
            smiles_batch = dm.decode(reconstructed_x_log_probs) # Pass log_probs, dm.decode should handle argmax
            all_smiles_for_props.extend(smiles_batch)


    torch.save(all_z_vectors, output_dir / f"{args.name}.pt")
    print(f"Saved {num_samples_to_take} latent vectors to {output_dir / f'{args.name}.pt'}")
    return all_smiles_for_props[:num_samples_to_take] # Ensure correct length if num_samples_to_take was adjusted


def main():
    # This function now receives SMILES that correspond to the saved z vectors
    smiles_corresponding_to_z = make_latent_dataset()
    chunk_idx = list(partitionIndexes(len(smiles_corresponding_to_z), n_cpus))[1:]

    def func(_x: pd.Series): # _x is now a row with just 'smiles' initially
        smiles_str = _x["smiles"]
        if args.binding_affinity:
            device_idx = bisect.bisect_right(chunk_idx, _x.name) # _x.name is the original df index
            mol = Chem.MolFromSmiles(smiles_str)
            if mol is None:
                for name in PROTEIN_FILES.keys(): _x[name] = np.nan # Or some default bad value
                return _x
            for name, file in PROTEIN_FILES.items():
                _x[name] = smiles2affinity(smiles_str, file, device_idx=device_idx)
        else:
            for prop in PROPS:
                _x[prop] = PROP_FN[prop](smiles_str)
        return _x

    df_props = pd.DataFrame(smiles_corresponding_to_z, columns=["smiles"])
    
    print(f"Finished generating {len(df_props)} SMILES corresponding to latent vectors.")
    print("Starting to compute properties for these SMILES.")

    if args.binding_affinity:
        df_props = df_props.parallel_apply(func, axis=1)
    else:
        df_props = df_props.parallel_apply(func, axis=1)
    
    # df_props = df_props.set_index("smiles") # Don't set index if we need to align with .pt file by row order
    df_props.to_csv(output_dir / f"{args.name}.csv", index=False) # Save with default numerical index
    print(f"Saved properties to {output_dir / f'{args.name}.csv'}")


if __name__ == "__main__":
    args = Args().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Define device globally for make_latent_dataset
    n_cpus = torch.cuda.device_count() if args.binding_affinity and torch.cuda.is_available() else os.cpu_count()

    if args.binding_affinity:
        print(f"Using {n_cpus} GPUs for binding affinity computation if available, else CPUs.")

    pandarallel.initialize(nb_workers=n_cpus, progress_bar=True, verbose=2)
    output_dir = Path("data/interim/props")
    output_dir.mkdir(parents=True, exist_ok=True)
    main()