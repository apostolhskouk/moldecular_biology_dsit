import torch
import numpy as np
from pathlib import Path
import sys

# Assuming running from ChemFlow-main directory
# Add src to path if necessary, though not strictly needed just for loading tensor
# sys.path.insert(0, str(Path(__file__).parent.parent))

# --- Configuration ---
# Match the settings used in prepare_random_data.py
N_SAMPLES = 110000
SEED = 42
BINDING_AFFINITY = False # Set to True if you generated binding affinity data
DATA_BASE_PATH = Path("data/interim/props")
# --- End Configuration ---


logits_file = "/data/hdd1/users/akouk/moldecular_biology_dsit/data/interim/props/prop_predictor_110000_seed42_molgen.pt"
# Load the tensor
try:
    logits_tensor = torch.load(logits_file, map_location='cpu') # Load to CPU
    print(f"Successfully loaded tensor with shape: {logits_tensor.shape}")
    print(f"Tensor dtype: {logits_tensor.dtype}")

    # Check tensor dimensions
    if len(logits_tensor.shape) != 2 or logits_tensor.shape[0] != N_SAMPLES:
        print(f"Warning: Tensor shape {logits_tensor.shape} does not match expected ({N_SAMPLES}, features).")

    # Check for NaNs
    nan_mask = torch.isnan(logits_tensor)
    num_nans = torch.sum(nan_mask).item()
    if num_nans > 0:
        print(f"\n--- !!! Found {num_nans} NaN values in the tensor !!! ---")
        # Optionally print locations or affected samples
        nan_indices = torch.where(nan_mask)
        print(f"  First few NaN locations (sample_idx, feature_idx): {list(zip(nan_indices[0][:10].tolist(), nan_indices[1][:10].tolist()))}")
    else:
        print("\n--- No NaN values found in the tensor. ---")

    # Check for Infs
    inf_mask = torch.isinf(logits_tensor)
    num_infs = torch.sum(inf_mask).item()
    if num_infs > 0:
        print(f"\n--- !!! Found {num_infs} Inf values in the tensor !!! ---")
        inf_indices = torch.where(inf_mask)
        print(f"  First few Inf locations (sample_idx, feature_idx): {list(zip(inf_indices[0][:10].tolist(), inf_indices[1][:10].tolist()))}")
    else:
        print("\n--- No Inf values found in the tensor. ---")

    # Check min/max values (can indicate extreme values)
    if num_nans == 0 and num_infs == 0:
        min_val = torch.min(logits_tensor).item()
        max_val = torch.max(logits_tensor).item()
        mean_val = torch.mean(logits_tensor).item()
        std_val = torch.std(logits_tensor).item()
        print("\n--- Tensor Value Statistics ---")
        print(f"  Min value: {min_val}")
        print(f"  Max value: {max_val}")
        print(f"  Mean value: {mean_val}")
        print(f"  Std value: {std_val}")

except Exception as e:
    print(f"An error occurred while loading or checking the tensor: {e}")
    sys.exit(1)