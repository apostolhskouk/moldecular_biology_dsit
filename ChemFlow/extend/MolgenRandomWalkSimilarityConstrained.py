import torch
import numpy as np
import pandas as pd
from accelerate.utils import set_seed

import os
import random
from pathlib import Path
from tqdm import tqdm, trange 
from pandarallel import pandarallel

from cd2root import cd2root
cd2root()

# Assuming your GenerateMethods is here:
from MolTransformer_repo.MolTransformer.generative import GenerateMethods 
from ChemFlow.src.utils.scores import PROP_FN, MINIMIZE_PROPS


def main():
    prop_to_calculate = "qed"
    n_molecules_to_process = 800
    n_exploration_steps = 100
    step_size_magnitude = 2.0

    neighbor_search_num_vectors = 10 
    neighbor_search_range = 20.0      
    neighbor_search_resolution = 0.05 

    data_file_path = "data/interim/props/zinc250k.csv"
    output_dir_base = Path("/data/hdd1/users/akouk/moldecular_biology_dsit/ChemFlow/extend/optimization_results_molgen_truncated_100steps") 

    pandarallel.initialize(nb_workers=os.cpu_count(), progress_bar=True, verbose=0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    output_dir_base.mkdir(parents=True, exist_ok=True)

    GM = GenerateMethods(gpu_mode=torch.cuda.is_available(), save=False)
    
    if hasattr(GM, 'model') and isinstance(GM.model, torch.nn.Module):
        GM.model.eval()


    df_initial_full = pd.read_csv(data_file_path)
    if prop_to_calculate not in df_initial_full.columns:
        df_initial_full = df_initial_full[["smiles"]]
        df_initial_full[prop_to_calculate] = 0.0
    else:
        df_initial_full = df_initial_full[["smiles", prop_to_calculate]]

    df_initial_full = df_initial_full.sort_values(
        prop_to_calculate,
        ascending=(prop_to_calculate in MINIMIZE_PROPS)
    )
    initial_smiles_list_all = df_initial_full["smiles"].unique()[:n_molecules_to_process].tolist()

    if not initial_smiles_list_all:
        print("No initial SMILES to process. Exiting.")
        return

    # Encoding is done once: use no_grad here too.
    with torch.no_grad():
        z_molgen_np_initial = GM.smiles_2_latent_space(initial_smiles_list_all)
    
    latent_seq_len = z_molgen_np_initial.shape[1] 
    latent_embed_dim = z_molgen_np_initial.shape[2]

    exploration_methods = [
        #"random_walk",
        #"fixed_random_direction",
        #"fixed_1d_direction",
        "neighboring_search" 
    ]

    for method_name in exploration_methods:
        print(f"\n--- Running Method: {method_name} ---")
        
        current_method_results = []
        
        if method_name == "neighboring_search":
            print(f"Performing neighboring search for {len(initial_smiles_list_all)} initial molecules...")
            for i, initial_smile_for_search in enumerate(tqdm(initial_smiles_list_all, desc="Neighbor Search Progress")):
                current_method_results.append({
                    "idx": i, 
                    "t": -1, 
                    "smiles": initial_smile_for_search, 
                    "method": method_name,
                    "neighbor_idx": -1 
                })
                with torch.no_grad(): # GM.neighboring_search involves model inference
                    generated_neighbors_data, _ = GM.neighboring_search(
                        initial_smile=initial_smile_for_search,
                        num_vector=neighbor_search_num_vectors,
                        search_range=neighbor_search_range,
                        resolution=neighbor_search_resolution
                    )
                for neighbor_j, neighbor_smiles in enumerate(generated_neighbors_data['SMILES']):
                    current_method_results.append({
                        "idx": i, 
                        "t": 0,   
                        "smiles": neighbor_smiles,
                        "method": method_name,
                        "neighbor_idx": neighbor_j 
                    })
        else: 
            z_molgen = torch.tensor(z_molgen_np_initial, dtype=torch.float32, device=device)

            for i, s_init in enumerate(initial_smiles_list_all):
                current_method_results.append({"idx": i, "t": -1, "smiles": s_init, "method": method_name})

            fixed_direction_vector_for_method = None
            if method_name == "fixed_random_direction":
                fixed_direction_vector_for_method = torch.randn(latent_seq_len, latent_embed_dim, device=device)
                norm = torch.norm(fixed_direction_vector_for_method) 
                if norm > 1e-9: fixed_direction_vector_for_method /= norm
            
            elif method_name == "fixed_1d_direction":
                fixed_direction_vector_for_method = torch.zeros(latent_seq_len, latent_embed_dim, device=device)
                rand_seq_idx = random.randint(0, latent_seq_len - 1)
                rand_embed_idx = random.randint(0, latent_embed_dim - 1)
                rand_val = 1.0 if random.random() < 0.5 else -1.0
                fixed_direction_vector_for_method[rand_seq_idx, rand_embed_idx] = rand_val
            
            z_molgen_current_trajectory = z_molgen.clone()

            # Wrap the exploration loop with torch.no_grad() for walk methods
            with torch.no_grad():
                for t_step in tqdm(range(n_exploration_steps), desc=f"Steps for {method_name}"):

                    if method_name == "random_walk":
                        raw_perturbation_matrices = torch.randn_like(z_molgen_current_trajectory) 
                        norms = torch.norm(raw_perturbation_matrices.view(n_molecules_to_process, -1), dim=1, keepdim=True)
                        norms = norms.view(n_molecules_to_process, 1, 1) 
                        norms[norms < 1e-9] = 1.0
                        normalized_directions = raw_perturbation_matrices / norms
                        step_perturbation = normalized_directions * step_size_magnitude
                    
                    elif method_name in ["fixed_random_direction", "fixed_1d_direction"]:
                        scaled_direction_matrix = fixed_direction_vector_for_method * step_size_magnitude
                        step_perturbation = scaled_direction_matrix.unsqueeze(0).repeat(n_molecules_to_process, 1, 1)
                    
                    z_molgen_current_trajectory += step_perturbation
                    
                    input_to_gm = z_molgen_current_trajectory.cpu().numpy()
                    
                    # This call involves model inference
                    decoded_mols_data = GM.latent_space_2_strings(input_to_gm)
                    generated_smiles_this_step = decoded_mols_data['SMILES']

                    for i, s_gen in enumerate(generated_smiles_this_step):
                        current_method_results.append({"idx": i, "t": t_step, "smiles": s_gen, "method": method_name})
        
        df_method_all_steps = pd.DataFrame(current_method_results)
        
        df_method_all_steps['smiles'] = df_method_all_steps['smiles'].astype(str)
        df_method_unique_smiles = df_method_all_steps[["smiles"]].drop_duplicates("smiles")
        df_method_unique_smiles = df_method_unique_smiles[df_method_unique_smiles['smiles'].notna() & (df_method_unique_smiles['smiles'] != '')]

        def calculate_property_for_row(row_series):
            row_series[prop_to_calculate] = PROP_FN[prop_to_calculate](row_series["smiles"])
            return row_series

        print(f"Calculating properties for {method_name}...")
        if not df_method_unique_smiles.empty:
            df_method_unique_smiles = df_method_unique_smiles.parallel_apply(calculate_property_for_row, axis=1)
            df_method_all_steps = df_method_all_steps.merge(df_method_unique_smiles, on="smiles", how="left")
        else:
            df_method_all_steps[prop_to_calculate] = np.nan

        output_file_name_parts = [prop_to_calculate, method_name]
        if method_name != "neighboring_search": 
            output_file_name_parts.extend([f"s{step_size_magnitude}", f"t{n_exploration_steps}"])
        else: 
            output_file_name_parts.extend([f"nv{neighbor_search_num_vectors}", f"r{neighbor_search_range}", f"res{neighbor_search_resolution}"])
        output_file_name_parts.append(f"n{n_molecules_to_process}")
        
        output_file_name = "_".join(output_file_name_parts) + ".csv"
        output_file_path = output_dir_base / output_file_name
        df_method_all_steps.to_csv(output_file_path, index=False)
        print(f"Results for {method_name} saved to {output_file_path}")

    print("\nAll methods completed.")

if __name__ == "__main__":
    main()