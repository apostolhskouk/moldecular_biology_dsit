# File: custom/experiments/prepare_random_data.py
import os
#sent only one gpu available
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import bisect # Note: bisect is used in the original calculate_props, but not in the new one. Kept for now if other parts rely on it.
import pandas as pd
import numpy as np
from pandarallel import pandarallel
import torch
from tqdm import tqdm, trange, tqdm_notebook # Use tqdm_notebook if in Jupyter
from tap import Tap
from pathlib import Path
from rdkit import Chem, RDLogger
from accelerate.utils import set_seed
# import traceback # Re-imported below
import gc # Garbage collector
from typing import Optional, List, Tuple
import queue
# --- Add imports for multiprocessing ---
import torch.multiprocessing as mp
from functools import partial
import traceback # For detailed error logging
import json # For loading MolGen config

# --- cd2root (keep if necessary for your project structure) ---
from cd2root import cd2root
cd2root()

# --- ChemFlow Imports (Updated) ---
from ChemFlow.src.utils.scores import (
    PROP_FN,
    PROTEIN_FILES,
    smiles2affinity,
    MINIMIZE_PROPS # Keep all from *
)
#from experiments.utils.utils import partitionIndexes

# --- MolGen Interface Import (Updated) ---
from custom.molgen_interface import MolGenInterface

# --- Disable RDKit Logs ---
RDLogger.DisableLog('rdApp.*')

# === Argument Parser ===

class Args(Tap):
    seed: int = 42
    n: int = 11000 # Number of molecules to generate for predictor training
    batch_size: int = 128 # Batch size for decoding WITHIN each worker process

    binding_affinity: bool = False # Option for binding affinity calculation

    # --- MolGen Arguments (adjust as needed) ---
    molgen_config_json: Optional[str] = None # Path to MolGen model config (if needed by Interface)
    # Add other relevant paths or parameters required by MolGenInterface
    # e.g., molgen_checkpoint_path: Optional[str] = None

    def process_args(self):
        # Update output name to reflect MolGen usage
        self.output_base_name = f"prop_predictor_{self.n}_seed{self.seed}_molgen"
        if self.binding_affinity:
            self.output_base_name += "_binding_affinity"

# === Global Lists/Dictionaries (ensure accessible in `calculate_props`) ===
PROPS = ["plogp", "uplogp", "qed", "sa", "gsk3b", "jnk3", "drd2"]
BINDING_AFFINITY_PROPS = ["1err", "2iik"] # Used if binding_affinity is True
# Note: PROTEIN_FILES and PROP_FN are imported from src.utils.scores

# --- MODIFIED decode_worker ---
def decode_worker(
    z_chunk: torch.Tensor,
    worker_id: int,
    q_results: mp.Queue,
    args: Args, # Pass full args
    molgen_latent_seq_len: int, # Pass dimensions explicitly
    molgen_latent_embed_dim: int,
    molgen_output_seq_len: int,
    molgen_vocab_size: int,
    device_name: str
):
    """
    Worker function to decode a chunk of latent vectors.
    Now takes necessary MolGen dimensions explicitly.
    """
    worker_device = torch.device(device_name)
    # print(f"[Worker {worker_id}] Using device: {worker_device}") # User requested no new prints
    log_prefix = f"[Worker {worker_id}]" # Kept for existing prints

    try:
        # --- Initialize MolGenInterface within the worker ---
        gm_config_worker = {}
        if args.molgen_config_json and os.path.exists(args.molgen_config_json):
            with open(args.molgen_config_json, 'r') as f:
                gm_config_worker = json.load(f)
            # print(f"{log_prefix} MolGen config loaded from {args.molgen_config_json}") # User requested no new prints
        
        molgen_interface = MolGenInterface(gm_config=gm_config_worker, device=worker_device)
        # print(f"{log_prefix} MolGenInterface instantiated.") # User requested no new prints

        if molgen_interface.output_seq_len != molgen_output_seq_len or \
           molgen_interface.model_vocab_size != molgen_vocab_size:
            # print(f"{log_prefix} WARNING: Interface dimensions mismatch with main thread!") # User requested no new prints
            # print(f"  Interface output_seq_len: {molgen_interface.output_seq_len}, Expected: {molgen_output_seq_len}")
            # print(f"  Interface vocab_size: {molgen_interface.model_vocab_size}, Expected: {molgen_vocab_size}")
            pass # Silenced warnings as requested

        logits_dim_worker = molgen_output_seq_len * molgen_vocab_size
        if logits_dim_worker <= 0:
             # print(f"{log_prefix} Error: Invalid logits dimension calculated ({logits_dim_worker}). Check passed MolGen dimensions.") # User requested no new prints
             q_results.put((worker_id, None, None))
             return

        worker_all_logits = torch.zeros((z_chunk.shape[0], logits_dim_worker), dtype=torch.float32, device='cpu')
        worker_all_smiles = [None] * z_chunk.shape[0]

        # print(f"{log_prefix} Processing {z_chunk.shape[0]} samples in batches of {args.batch_size}...") # User requested no new prints

        for i in trange(0, z_chunk.shape[0], args.batch_size, desc=f"Worker {worker_id} Batches", leave=False, position=worker_id):
            start_idx_local = i
            end_idx_local = min(i + args.batch_size, z_chunk.shape[0])
            z_batch_local = z_chunk[start_idx_local:end_idx_local].to(worker_device)

            with torch.no_grad():
                logits_batch = molgen_interface.decode_to_logits(z_batch_local)
                smiles_batch = molgen_interface.decode_to_smiles(z_batch_local)

            worker_all_logits[start_idx_local:end_idx_local] = logits_batch.cpu()
            
            if len(smiles_batch) == (end_idx_local - start_idx_local):
                 worker_all_smiles[start_idx_local:end_idx_local] = smiles_batch
            else:
                 # print(f"{log_prefix} Warning: SMILES batch length mismatch at local index {start_idx_local}. Expected {end_idx_local - start_idx_local}, got {len(smiles_batch)}. Filling with None.") # User requested no new prints
                 worker_all_smiles[start_idx_local:end_idx_local] = [None] * (end_idx_local - start_idx_local)

            del z_batch_local, logits_batch
            if 'smiles_batch' in locals(): del smiles_batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        q_results.put((worker_id, worker_all_logits.cpu(), worker_all_smiles))
        # print(f"{log_prefix} Finished processing chunk.") # User requested no new prints

    except Exception as e:
        # print(f"{log_prefix} FATAL EXCEPTION: {e}") # User requested no new prints
        traceback.print_exc() 
        q_results.put((worker_id, None, None))
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


# --- MODIFIED make_dataset_parallel ---
def make_dataset_parallel(args: Args, main_device: torch.device, num_gpus: int, output_dir_path: Path):
    set_seed(args.seed)
    # print("Initiating parallel data generation...") # User requested no new prints

    # print("Initializing temporary MolGenInterface to get dimensions...") # User requested no new prints
    temp_gm_config = {}
    if args.molgen_config_json and os.path.exists(args.molgen_config_json):
        with open(args.molgen_config_json, 'r') as f:
            temp_gm_config = json.load(f)
        # print(f"Main thread MolGen config loaded from {args.molgen_config_json}") # User requested no new prints

    temp_molgen_interface = MolGenInterface(gm_config=temp_gm_config, device=main_device)
    molgen_latent_seq_len = temp_molgen_interface.latent_seq_len
    molgen_latent_embed_dim = temp_molgen_interface.latent_embed_dim
    molgen_output_seq_len = temp_molgen_interface.output_seq_len
    molgen_vocab_size = temp_molgen_interface.model_vocab_size
    del temp_molgen_interface
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect()
    # print(f"Obtained latent dimensions: seq_len={molgen_latent_seq_len}, embed_dim={molgen_latent_embed_dim}") # User requested no new prints
    # print(f"Obtained output dimensions: seq_len={molgen_output_seq_len}, vocab_size={molgen_vocab_size}") # User requested no new prints

    if molgen_latent_seq_len <= 0 or molgen_latent_embed_dim <=0 or molgen_output_seq_len <=0 or molgen_vocab_size <=0:
        raise ValueError("Failed to obtain valid MolGen dimensions.")

    # print(f"Generating {args.n} random latent sequences for MolGen...") # User requested no new prints
    z0_cpu = torch.randn(
        args.n,
        molgen_latent_seq_len,
        molgen_latent_embed_dim,
        device='cpu'
    )
    # print(f"Generated latent tensor z0 shape: {z0_cpu.shape} on CPU") # User requested no new prints

    if num_gpus == 0:
        available_devices = ['cpu']
        num_workers_mp = max(1, os.cpu_count() // 2 if os.cpu_count() else 1)
        # print(f"No GPUs detected for decoding. Using {num_workers_mp} CPU workers.") # User requested no new prints
    else:
        available_devices = [f'cuda:{i}' for i in range(num_gpus)]
        num_workers_mp = num_gpus
        # print(f"Using {num_workers_mp} GPU workers for decoding.") # User requested no new prints

    if args.n == 0:
        # print("Warning: args.n is 0. No data to generate.") # User requested no new prints
        return []
        
    samples_per_worker = args.n // num_workers_mp
    remainder = args.n % num_workers_mp
    split_sizes = [samples_per_worker + 1 if i < remainder else samples_per_worker for i in range(num_workers_mp)]
    split_sizes = [s for s in split_sizes if s > 0] 

    if not split_sizes or sum(split_sizes) != args.n:
         raise ValueError(f"Error in splitting tensor: split_sizes {split_sizes} do not sum to args.n {args.n}")

    z_chunks = torch.split(z0_cpu, split_sizes, dim=0)
    actual_num_workers = len(z_chunks)
    # print(f"Splitting {args.n} samples into {actual_num_workers} chunks for {actual_num_workers} workers. Chunk sizes: {split_sizes}") # User requested no new prints
    del z0_cpu 
    gc.collect()

    ctx = mp.get_context('spawn')
    q_results = ctx.Queue()
    processes = []

    # print("Launching worker processes...") # User requested no new prints
    for i in range(actual_num_workers):
        worker_device_name = available_devices[i % len(available_devices)]
        target_func = partial(decode_worker,
                              args=args,
                              molgen_latent_seq_len=molgen_latent_seq_len,
                              molgen_latent_embed_dim=molgen_latent_embed_dim,
                              molgen_output_seq_len=molgen_output_seq_len,
                              molgen_vocab_size=molgen_vocab_size,
                              device_name=worker_device_name)
        p = ctx.Process(target=target_func, args=(z_chunks[i].clone(), i, q_results))
        processes.append(p)
        p.start()
    del z_chunks 
    gc.collect()

    # print("Workers launched. Waiting for results...") # User requested no new prints
    results_logits = [None] * actual_num_workers
    results_smiles = [None] * actual_num_workers
    all_workers_succeeded = True
    
    for _ in range(actual_num_workers): # Use `trange` here if progress is desired and acceptable
        try:
            worker_id, logits_chunk, smiles_chunk = q_results.get(timeout=10800) # Increased timeout to 3 hours
        except queue.Empty:
            # print("Error: Timed out waiting for a worker result. A worker might have died.") # User requested no new prints
            all_workers_succeeded = False
            break 

        if logits_chunk is None or smiles_chunk is None:
            # print(f"Error: Worker {worker_id} failed and reported None results.") # User requested no new prints
            all_workers_succeeded = False
        else:
            # print(f"Received results from worker {worker_id} (Logits: {logits_chunk.shape}, SMILES: {len(smiles_chunk)})") # User requested no new prints
            results_logits[worker_id] = logits_chunk
            results_smiles[worker_id] = smiles_chunk

    for p_idx, p in enumerate(processes):
        p.join(timeout=60) 
        if p.is_alive():
            # print(f"Warning: Process {p_idx} (PID {p.pid}) did not terminate gracefully. Terminating.") # User requested no new prints
            p.terminate()
            p.join() 
            all_workers_succeeded = False 
            # if results_logits[p_idx] is None: # User requested no new prints
                 # print(f"Worker {p_idx} likely caused the earlier timeout or failure.")

    # print("Workers finished. Gathering results...") # User requested no new prints
    if not all_workers_succeeded or any(r is None for r in results_logits):
        failed_workers = [idx for idx, res in enumerate(results_logits) if res is None]
        # if failed_workers: # User requested no new prints
            # print(f"Data from workers {failed_workers} is missing.")
        raise RuntimeError("One or more worker processes failed or did not return data during data generation.")

    all_logits_final = torch.cat(results_logits, dim=0)
    all_smiles_final = [s for chunk in results_smiles if chunk is not None for s in chunk]

    # print(f"Final logits tensor shape: {all_logits_final.shape}") # User requested no new prints
    # print(f"Total SMILES generated: {len(all_smiles_final)}") # User requested no new prints

    if all_logits_final.shape[0] != len(all_smiles_final):
        # print(f"CRITICAL WARNING: Mismatch between final logits count ({all_logits_final.shape[0]}) and SMILES count ({len(all_smiles_final)}).") # User requested no new prints
        pass # Silenced warning

    logits_save_path = output_dir_path / f"{args.output_base_name}.pt"
    torch.save(all_logits_final, logits_save_path)
    # print(f"Saved combined decoded logits to {logits_save_path}") # User requested no new prints

    return all_smiles_final


# --- MODIFIED main ---
def main(args: Args, main_device_name: str, n_parallel_workers_props: int, output_dir_path: Path):
    # print("Starting main function...") # User requested no new prints
    if main_device_name.startswith('cuda'):
        num_gpus_for_decoding = torch.cuda.device_count()
        # if num_gpus_for_decoding == 0: # User requested no new prints
             # print("Warning: main_device is cuda, but no GPUs detected. Decoding will use CPU.")
    else: 
        num_gpus_for_decoding = 0
    # print(f"Decoding will use {num_gpus_for_decoding} GPUs (or CPU workers if 0).") # User requested no new prints

    smiles_list = make_dataset_parallel(args, torch.device(main_device_name), num_gpus_for_decoding, output_dir_path)

    if not smiles_list:
        # print("No SMILES strings were generated. Skipping property calculation.") # User requested no new prints
        return

    # print(f"Finished generating {len(smiles_list)} smiles. Starting property calculation...") # User requested no new prints
    pandarallel.initialize(
        nb_workers=n_parallel_workers_props,
        progress_bar=True,
        verbose=0, # Silenced pandarallel
    )

    def calculate_props(series: pd.Series):
        smiles_str = series["smiles"]

        if not isinstance(smiles_str, str) or pd.isna(smiles_str) or not smiles_str:
            props_to_set_nan = BINDING_AFFINITY_PROPS if args.binding_affinity else PROPS
            for prop_name_key in props_to_set_nan: series[prop_name_key] = np.nan
            return series

        mol = Chem.MolFromSmiles(smiles_str)
        if mol is None:
            props_to_set_nan = BINDING_AFFINITY_PROPS if args.binding_affinity else PROPS
            for prop_name_key in props_to_set_nan: series[prop_name_key] = np.nan
            return series

        if args.binding_affinity:
            pandas_worker_idx = series.name 
            
            num_gpus_for_props = torch.cuda.device_count() if torch.cuda.is_available() else 0
            worker_device_idx_for_props = 0 
            if num_gpus_for_props > 0 :
                worker_device_idx_for_props = pandas_worker_idx % num_gpus_for_props

            for name, file_path in PROTEIN_FILES.items():
                if not os.path.exists(file_path):
                     # print(f"Error: Protein file not found: {file_path}") # User requested no new prints
                     series[name] = np.nan
                     continue
                try:
                    series[name] = smiles2affinity(smiles_str, file_path, device_idx=worker_device_idx_for_props)
                except Exception as e:
                    series[name] = np.nan
        else: 
            for prop_name_key in PROPS:
                try:
                    series[prop_name_key] = PROP_FN[prop_name_key](smiles_str)
                except Exception as e:
                     series[prop_name_key] = np.nan
        return series

    df = pd.DataFrame(smiles_list, columns=["smiles"])
    if df.empty:
        # print("DataFrame from SMILES list is empty. Cannot calculate properties.") # User requested no new prints
        return
        
    df.reset_index(drop=True, inplace=True) 
    
    # print("Applying property calculations in parallel...") # User requested no new prints
    df = df.parallel_apply(calculate_props, axis=1)
    # print("Property calculation complete.") # User requested no new prints

    df_props_to_save = df.copy()
    
    if 'smiles' in df_props_to_save.columns:
         df_props_to_save = df_props_to_save.set_index('smiles', drop=True)
    
    df_props_to_save.dropna(how='all', inplace=True)

    if df_props_to_save.empty:
        # print("Warning: DataFrame is empty after dropping rows with all NaN properties. No properties will be saved.") # User requested no new prints
        pass
    else:
        props_save_path = output_dir_path / f"{args.output_base_name}.csv"
        df_props_to_save.to_csv(props_save_path)
        # print(f"Saved calculated properties ({df_props_to_save.shape[0]} valid rows) to {props_save_path}") # User requested no new prints


try:
    mp.set_sharing_strategy('file_system')
    # print("Multiprocessing sharing strategy set to 'file_system'.") # User requested no new prints
except RuntimeError:
    # print("Warning: Could not set multiprocessing sharing strategy (might already be set or not supported).") # User requested no new prints
    pass

if __name__ == "__main__":
    args = Args().parse_args()
    args.process_args() 

    n_workers_for_props = os.cpu_count() if os.cpu_count() else 1 
    
    # if args.binding_affinity: # User requested no new prints
    #     num_gpus_avail = torch.cuda.device_count()
    #     # print(f"Binding affinity selected. Property calculation will use {n_workers_for_props} parallel workers (pandarallel).")
    #     # if num_gpus_avail > 0:
    #         # print(f"  {num_gpus_avail} GPU(s) available, smiles2affinity may utilize them depending on its implementation.")
    #     # else:
    #         # print("  No GPUs available for potential acceleration of binding affinity calculation.")
    # else:
    #     # print(f"Using {n_workers_for_props} CPU workers for standard property calculation (pandarallel).")
    #     pass


    main_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Main script process using device: {main_device}") # User requested no new prints

    output_dir = Path("data/interim/props")
    output_dir.mkdir(parents=True, exist_ok=True)

    if main_device.type == 'cuda': 
        current_start_method = mp.get_start_method(allow_none=True)
        if current_start_method != 'spawn':
            try:
                mp.set_start_method('spawn', force=True)
                # print("Multiprocessing start method set to 'spawn'.") # User requested no new prints
            except RuntimeError as e:
                # print(f"Note: Could not set multiprocessing start method to 'spawn' (possibly already set or started): {e}") # User requested no new prints
                pass
        # else: # User requested no new prints
            # print(f"Multiprocessing start method already '{current_start_method}'.")


    try:
        main(args, str(main_device), n_workers_for_props, output_dir)
    except Exception as e:
        # print(f"An error occurred in the main function: {e}") # User requested no new prints
        traceback.print_exc()
    # finally: # User requested no new prints
        # print("Script execution finished.")