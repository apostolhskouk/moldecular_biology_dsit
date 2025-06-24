import os
#set only one gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
import torch
import numpy as np
import pandas as pd
from accelerate.utils import set_seed
import argparse
from pathlib import Path
from tqdm import tqdm
from pandarallel import pandarallel
import sys

# Ensure the root of the repository is in the Python path
# You may need to adjust this path based on your project structure.
# If your script is inside 'ChemFlow/extend/', this should work.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from MolTransformer_repo.MolTransformer.generative import GenerateMethods 
# Mockup for demonstration if ChemFlow is not available.
# Replace with your actual import if needed.
PROP_FN = {'qed': lambda x: np.random.rand()}
MINIMIZE_PROPS = []


def main():
    parser = argparse.ArgumentParser(description="Run guided molecular optimization using MolGen-Transformer.")
    parser.add_argument("prop_to_calculate", type=str, default="qed", help="Property to optimize.")
    args = parser.parse_args()
    prop_to_calculate = args.prop_to_calculate

    # --- Configuration ---
    n_initial_molecules = 100
    max_optimization_steps = 100
    neighbor_search_num_vectors = 10
    neighbor_search_range = 30.0
    neighbor_search_resolution = 0.05
    pareto_alpha = 0.5
    top_k_candidates = 100
    sa_threshold = 6.0

    data_file_path = "ChemFlow/data/interim/props/zinc250k.csv"
    output_dir_base = Path("ChemFlow/extend/optimization_results_molgen_guided_single_gpu")

    # --- Initialization ---
    pandarallel.initialize(nb_workers=os.cpu_count(), progress_bar=True, verbose=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    output_dir_base.mkdir(parents=True, exist_ok=True)

    print("Initializing GenerateMethods...")
    # Set gpu_mode=False. The library will still use the 'cuda' device if available.
    GM = GenerateMethods(gpu_mode=False, save=False)

    print("Setting property prediction model...")
    GM.set_property_model(dataset='qm9')

    if hasattr(GM, 'model'): GM.model.eval()
    if hasattr(GM, 'property_model'): GM.property_model.eval()

    # --- Load and Prepare Data ---
    print(f"Loading initial molecules from {data_file_path}...")
    df_initial_full = pd.read_csv(data_file_path)
    df_initial_full = df_initial_full.sort_values(
        prop_to_calculate, ascending=(prop_to_calculate in MINIMIZE_PROPS)
    )
    initial_smiles_list = df_initial_full["smiles"].unique()[:n_initial_molecules].tolist()

    # --- Run Guided Optimization ---
    all_trajectories = []
    print(f"Running guided optimization for {len(initial_smiles_list)} molecules...")
    
    with torch.no_grad():
        for i, start_smile in enumerate(tqdm(initial_smiles_list, desc="Optimizing Molecules")):
            results = GM.optimistic_property_driven_molecules_generation(
                initial_smile=start_smile, dataset='qm9', max_step=max_optimization_steps,
                num_vector=neighbor_search_num_vectors, search_range=neighbor_search_range,
                resolution=neighbor_search_resolution, alpha=pareto_alpha,
                k=top_k_candidates, sa_threshold=sa_threshold
            )
            for step, (smile, prop) in enumerate(zip(results['SMILES'], results['Property'])):
                all_trajectories.append({
                    "trajectory_id": i, "step": step, "smiles": smile, "internal_prop": prop
                })

    # --- Post-processing and Saving ---
    if not all_trajectories:
        print("No trajectories were generated.")
        return

    df_results = pd.DataFrame(all_trajectories)

    print("Calculating final properties for all generated molecules...")
    df_unique_smiles = df_results[["smiles"]].drop_duplicates("smiles").copy()
    df_unique_smiles = df_unique_smiles[df_unique_smiles['smiles'].notna() & (df_unique_smiles['smiles'] != '')]

    def calculate_property_for_row(row_series):
        row_series[prop_to_calculate] = PROP_FN[prop_to_calculate](row_series["smiles"])
        return row_series

    if not df_unique_smiles.empty:
        df_props = df_unique_smiles.parallel_apply(calculate_property_for_row, axis=1)
        df_results = df_results.merge(df_props, on="smiles", how="left")

    output_file_name = f"guided_single_gpu_{prop_to_calculate}_t{max_optimization_steps}_n{n_initial_molecules}.csv"
    output_file_path = output_dir_base / output_file_name
    df_results.to_csv(output_file_path, index=False)
    print(f"Optimization complete. Results saved to {output_file_path}")

if __name__ == "__main__":
    main()