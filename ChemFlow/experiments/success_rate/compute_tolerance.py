import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import os
from tqdm.auto import tqdm
from rdkit import Chem
from pandarallel import pandarallel
from ChemFlow.src.utils.scores import *
import sys
TRAINING_DATA_PATH = "ChemFlow/data/interim/props/zmc.csv"
SMILES_COLUMN_NAME = "smiles" 
OUTPUT_DIR = Path("ChemFlow/experiments/tolerance")

PROPERTIES = ["plogp", "sa", "qed", "drd2", "jnk3", "gsk3b"]

# Factor to use for tolerance calculation (5% as paper)
TOLERANCE_FACTOR = 0.05

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
pandarallel.initialize(nb_workers=os.cpu_count(), progress_bar=False, verbose=0)
use_parallel = True

def calculate_property(smiles_series, prop_name, prop_fn_dict):
    """Applies a property calculation function to a series of SMILES."""
    if prop_name not in prop_fn_dict:
        print(f"Warning: Property function for '{prop_name}' not found. Skipping.")
        return pd.Series([np.nan] * len(smiles_series), index=smiles_series.index)

    prop_fn = prop_fn_dict[prop_name]

    def safe_calc(s):
        # RDKit functions often need a Mol object
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return np.nan
        # Adapt based on whether your function takes SMILES or Mol
        # Assuming they take SMILES as in the original script context
        return prop_fn(s)


    tqdm.pandas(desc=f"Calculating {prop_name}")
    return smiles_series.progress_apply(safe_calc)

def calculate_property_parallel(smiles_series, prop_name, prop_fn_dict):
    """Applies a property calculation function using pandarallel."""
    if prop_name not in prop_fn_dict:
        print(f"Warning: Property function for '{prop_name}' not found. Skipping.")
        return pd.Series([np.nan] * len(smiles_series), index=smiles_series.index)

    prop_fn = prop_fn_dict[prop_name]

    def safe_calc_parallel(s):
        try:
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                return np.nan
            return prop_fn(s)
        except Exception:
            return np.nan

    print(f"Calculating {prop_name} (parallel)...")
    return smiles_series.parallel_apply(safe_calc_parallel)



print(f"Loading training data from: {TRAINING_DATA_PATH}")
if TRAINING_DATA_PATH.lower().endswith(".csv"):
    df = pd.read_csv(TRAINING_DATA_PATH)
elif TRAINING_DATA_PATH.lower().endswith(".smi"):
        # SMI files often just have SMILES, maybe with an ID
        # Adjust based on your SMI file format
    df = pd.read_csv(TRAINING_DATA_PATH, sep=' ', header=None, names=['smiles', 'id'])
    SMILES_COLUMN_NAME = 'smiles' # Override if needed
else:
    # Add loaders for other formats if necessary
    raise ValueError(f"Unsupported file format: {TRAINING_DATA_PATH}")

if SMILES_COLUMN_NAME not in df.columns:
    raise ValueError(f"SMILES column '{SMILES_COLUMN_NAME}' not found in {TRAINING_DATA_PATH}")
print(f"Loaded {len(df)} molecules.")


print("\nCalculating properties for the training set...")
for prop in PROPERTIES:
    print("-" * 20)
    if use_parallel:
        df[prop] = calculate_property_parallel(df[SMILES_COLUMN_NAME], prop, PROP_FN)
    else:
        df[prop] = calculate_property(df[SMILES_COLUMN_NAME], prop, PROP_FN)
    
    # Report basic stats
    num_valid = df[prop].notna().sum()
    if num_valid > 0:
        print(f"Property '{prop}': Calculated for {num_valid}/{len(df)} molecules.")
        print(f"  Mean: {df[prop].mean():.4f}, Std: {df[prop].std():.4f}")
        print(f"  Min: {df[prop].min():.4f}, Max: {df[prop].max():.4f}")
    else:
         print(f"Property '{prop}': No valid values calculated.")
         
    print("-" * 20)


# Calculate tolerances
print("\nCalculating tolerances...")
tolerance_range_dict = {}
tolerance_iqr_dict = {}
tolerance_std_dict = {}

for prop in PROPERTIES:
    prop_values = df[prop].dropna()
    if len(prop_values) < 2: # Need at least 2 points for stats
        print(f"Warning: Not enough valid data points for '{prop}' to calculate tolerance. Setting to 0.")
        tolerance_range_dict[prop] = 0.0
        tolerance_iqr_dict[prop] = 0.0
        tolerance_std_dict[prop] = 0.0
        continue

    # Range tolerance
    p_range = prop_values.max() - prop_values.min()
    tolerance_range_dict[prop] = TOLERANCE_FACTOR * p_range

    # IQR tolerance
    q75, q25 = np.percentile(prop_values, [75 ,25])
    p_iqr = q75 - q25
    tolerance_iqr_dict[prop] = TOLERANCE_FACTOR * p_iqr

    # Std tolerance
    p_std = prop_values.std()
    tolerance_std_dict[prop] = TOLERANCE_FACTOR * p_std

    print(f"Property '{prop}':")
    print(f"  Range = {p_range:.4f} -> Tolerance = {tolerance_range_dict[prop]:.4f}")
    print(f"  IQR = {p_iqr:.4f} -> Tolerance = {tolerance_iqr_dict[prop]:.4f}")
    print(f"  Std = {p_std:.4f} -> Tolerance = {tolerance_std_dict[prop]:.4f}")

# Save tolerance dictionaries to pickle files
print(f"\nSaving tolerance files to {OUTPUT_DIR}...")

range_file = OUTPUT_DIR / "relaxed_tolerance_range.pkl"
iqr_file = OUTPUT_DIR / "relaxed_tolerance_IQR.pkl"
std_file = OUTPUT_DIR / "relaxed_tolerance_std.pkl"

try:
    with open(range_file, "wb") as f:
        pickle.dump(tolerance_range_dict, f)
    print(f"Saved: {range_file}")

    with open(iqr_file, "wb") as f:
        pickle.dump(tolerance_iqr_dict, f)
    print(f"Saved: {iqr_file}")

    with open(std_file, "wb") as f:
        pickle.dump(tolerance_std_dict, f)
    print(f"Saved: {std_file}")

except Exception as e:
    print(f"Error saving pickle files: {e}")

print("\nScript finished.")