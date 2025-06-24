import torch
import numpy as np
import pandas as pd
from accelerate.utils import set_seed
import argparse
import os
from pathlib import Path
from tqdm import tqdm

# DISTRIBUTED: Import necessary libraries for distributed computing
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import sys
# Assuming pandarallel is not compatible with this distributed setup, we remove it.
# Property calculation will be done on the main process at the end.
# from pandarallel import pandarallel

# Ensure the root of the repository is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from MolTransformer_repo.MolTransformer.generative import GenerateMethods 
# Mockup for demonstration if ChemFlow is not available
PROP_FN = {'qed': lambda x: np.random.rand()}
MINIMIZE_PROPS = []

# DISTRIBUTED: A simple dataset class to wrap our list of SMILES
class SmilesDataset(Dataset):
    def __init__(self, smiles_list):
        self.smiles_list = smiles_list
    def __len__(self):
        return len(self.smiles_list)
    def __getitem__(self, idx):
        return self.smiles_list[idx]

def setup_distributed():
    """Initializes the distributed environment."""
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank

def main():
    # DISTRIBUTED: Setup the distributed process group
    rank, world_size, local_rank = setup_distributed()
    is_main_process = (rank == 0)

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
    output_dir_base = Path("ChemFlow/extend/optimization_results_molgen_guided_dist")

    # --- Initialization ---
    # pandarallel.initialize is removed
    device = torch.device("cuda", local_rank) # DISTRIBUTED: Each process gets its own GPU
    set_seed(42)
    if is_main_process:
        output_dir_base.mkdir(parents=True, exist_ok=True)

    if is_main_process: print("Initializing GenerateMethods...")
    # DISTRIBUTED: NOW we can set gpu_mode=True because we are in a proper distributed environment
    GM = GenerateMethods(gpu_mode=True, save=False)

    if is_main_process: print("Setting property prediction model...")
    GM.set_property_model(dataset='qm9')

    if hasattr(GM, 'model'): GM.model.eval()
    if hasattr(GM, 'property_model'): GM.property_model.eval()

    # --- Load and Prepare Data ---
    if is_main_process: print(f"Loading initial molecules from {data_file_path}...")
    df_initial_full = pd.read_csv(data_file_path)
    df_initial_full = df_initial_full.sort_values(
        prop_to_calculate, ascending=(prop_to_calculate in MINIMIZE_PROPS)
    )
    initial_smiles_list = df_initial_full["smiles"].unique()[:n_initial_molecules].tolist()

    # DISTRIBUTED: Create a dataset and a sampler to distribute the data
    dataset = SmilesDataset(initial_smiles_list)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    # Batch size of 1 means each GPU process gets one molecule at a time to optimize
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler)

    # --- Run Guided Optimization ---
    local_trajectories = []
    # DISTRIBUTED: Only the main process shows the progress bar
    progress_bar = tqdm(dataloader, desc="Optimizing Molecules", disable=not is_main_process)

    with torch.no_grad():
        for batch in progress_bar:
            start_smile = batch[0] # Dataloader returns a list, get the string
            trajectory_id = initial_smiles_list.index(start_smile) # Get original index for tracking

            try:
                results = GM.optimistic_property_driven_molecules_generation(
                    initial_smile=start_smile, dataset='qm9', max_step=max_optimization_steps,
                    num_vector=neighbor_search_num_vectors, search_range=neighbor_search_range,
                    resolution=neighbor_search_resolution, alpha=pareto_alpha,
                    k=top_k_candidates, sa_threshold=sa_threshold
                )
                for step, (smile, prop) in enumerate(zip(results['SMILES'], results['Property'])):
                    local_trajectories.append({
                        "trajectory_id": trajectory_id, "step": step, "smiles": smile, "internal_prop": prop
                    })
            except Exception as e:
                if is_main_process: print(f"Error on rank {rank} for molecule {trajectory_id} ({start_smile}): {e}")
                local_trajectories.append({
                    "trajectory_id": trajectory_id, "step": 0, "smiles": start_smile, "internal_prop": None
                })

    # DISTRIBUTED: Gather results from all processes to the main process
    dist.barrier() # Wait for all processes to finish
    all_gathered_trajectories = [None] * world_size
    dist.all_gather_object(all_gathered_trajectories, local_trajectories)

    # --- Post-processing and Saving (only on the main process) ---
    if is_main_process:
        print("\nGathering and processing results...")
        # Flatten the list of lists into a single list
        final_trajectories = [item for sublist in all_gathered_trajectories for item in sublist]

        if not final_trajectories:
            print("No trajectories were generated.")
            return

        df_results = pd.DataFrame(final_trajectories)
        df_results.sort_values(by=["trajectory_id", "step"], inplace=True)

        print("Calculating final properties for all generated molecules...")
        df_unique_smiles = df_results[["smiles"]].drop_duplicates("smiles").copy()
        df_unique_smiles = df_unique_smiles[df_unique_smiles['smiles'].notna() & (df_unique_smiles['smiles'] != '')]

        # Since pandarallel is out, we do this serially. It's usually fast enough.
        def calculate_property_for_row(row_series):
            try:
                row_series[prop_to_calculate] = PROP_FN[prop_to_calculate](row_series["smiles"])
            except:
                row_series[prop_to_calculate] = np.nan
            return row_series

        if not df_unique_smiles.empty:
            df_props = df_unique_smiles.apply(calculate_property_for_row, axis=1)
            df_results = df_results.merge(df_props, on="smiles", how="left")

        output_file_name = f"guided_dist_{prop_to_calculate}_t{max_optimization_steps}_n{n_initial_molecules}.csv"
        output_file_path = output_dir_base / output_file_name
        df_results.to_csv(output_file_path, index=False)
        print(f"Optimization complete. Results saved to {output_file_path}")

if __name__ == "__main__":
    main()