from dotenv import load_dotenv
import os
load_dotenv()
project_path = os.getenv('PROJECT_PATH')
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from MolTransformer_repo.MolTransformer import GenerateMethods 

def generate_moltransformer_latent_representations(csv_file_path: str, output_dir_path: str, smiles_column_name: str = "smiles", batch_size: int = 256):
    output_dir = Path(output_dir_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    gm = GenerateMethods(gpu_mode=True)

    df = pd.read_csv(csv_file_path)

    if smiles_column_name not in df.columns:
        return

    all_smiles = df[smiles_column_name].astype(str).tolist()
    all_latent_vectors = []

    num_batches = (len(all_smiles) + batch_size - 1) // batch_size

    for i in tqdm(range(num_batches), desc=f"Processing"):
        batch_smiles = all_smiles[i * batch_size : (i + 1) * batch_size]
        if not batch_smiles:
            continue
        latent_batch_np = gm.smiles_2_latent_space(batch_smiles)
        all_latent_vectors.append(latent_batch_np)

    if not all_latent_vectors:
        return

    final_latent_tensor_np = np.concatenate(all_latent_vectors, axis=0)
    final_latent_tensor = torch.from_numpy(final_latent_tensor_np).float()

    output_pt_filename = "/data/hdd1/users/akouk/moldecular_biology_dsit/custom/assets/molgen.pt"
    output_pt_path = output_dir / output_pt_filename
    torch.save(final_latent_tensor, output_pt_path)


if __name__ == "__main__":
    csv_path = "ChemFlow/data/interim/props/prop_predictor_110000_seed42_vae_pde.csv"
    output_pt_directory = "custom/assets/"
    smiles_col = "smiles"
    processing_batch_size = 128


    generate_moltransformer_latent_representations(
        csv_file_path=csv_path,
        output_dir_path=output_pt_directory,
        smiles_column_name=smiles_col,
        batch_size=processing_batch_size
    )