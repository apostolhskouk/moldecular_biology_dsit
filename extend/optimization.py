import torch
from torch import Tensor
import torch.nn as nn # For PropGenerator/VAEGenerator base
import torch.nn.functional as F # For MolGenTransformerWrapper
import numpy as np
import pandas as pd
from accelerate.utils import set_seed
import os
from pandarallel import pandarallel
from tap import Tap
from tqdm import trange
from typing import Literal
from pathlib import Path
import selfies as sf
from rdkit import Chem
import random
# --- Imports from your project structure ---
# Assuming these are correctly in PYTHONPATH or relative paths work
from ChemFlow.src.utils.scores import MINIMIZE_PROPS, PROP_FN # Adjust if WAVEPDE_IDX_MAP etc. are here too
from ChemFlow.src.pinn.pde import load_wavepde # Adjust path
# from ChemFlow.src.pinn import PropGenerator, VAEGenerator # Definitions will be here
from ChemFlow.src.predictor import Predictor # Adjust path
from extend.wrapper import MolGenTransformerWrapper # Your wrapper

# --- Helper Classes (SmilesTokenizerForWrapper, load_util, Traversal) ---
# It's cleaner if these are in separate utility files, but for a single script:

class SmilesTokenizerForWrapper:
    def __init__(self, wrapper_instance: MolGenTransformerWrapper):
        self.molgen_index = wrapper_instance.molgen_index
        self.char2ind = self.molgen_index.char2ind
        self.sos_idx = wrapper_instance.sos_idx 
        self.pad_idx = wrapper_instance.pad_idx 
        self.eos_idx = wrapper_instance.eos_idx 
        self.ind2char = self.molgen_index.ind2char
        self.padded_length = wrapper_instance.max_len 
        self.vocab_size = wrapper_instance.vocab_size

    def _preprocess_smiles_for_selfies(self, smiles_str: str) -> str:
        try:
            mol = Chem.MolFromSmiles(smiles_str)
            if mol is None: return ""
            hmol = Chem.AddHs(mol)
            k_smile = Chem.MolToSmiles(hmol, kekuleSmiles=True)
            return k_smile
        except Exception: return ""

    def encode(self, smiles_list: list[str], device: torch.device = None) -> torch.Tensor:
        target_device = device if device is not None else torch.device("cpu")
        token_ids_batch = []
        for smiles_str_original in smiles_list:
            processed_smiles = self._preprocess_smiles_for_selfies(smiles_str_original)
            selfies_str = ""
            if processed_smiles:
                try: selfies_str = sf.encoder(processed_smiles)
                except Exception: pass
            
            tokens = []
            if selfies_str:
                try: tokens = list(sf.split_selfies(selfies_str))
                except Exception: pass
            
            token_ids = [self.char2ind.get(t, self.pad_idx) for t in tokens]
            if len(token_ids) > self.padded_length - 1:
                token_ids = token_ids[:self.padded_length - 1]
            
            final_sequence = [self.sos_idx] + token_ids
            padding_needed = self.padded_length - len(final_sequence)
            final_sequence.extend([self.pad_idx] * padding_needed)
            token_ids_batch.append(final_sequence)
        return torch.tensor(token_ids_batch, dtype=torch.long).to(target_device)

    def decode_token_ids_to_smiles(self, batch_token_ids: list[list[int]]) -> list[str]:
        smiles_out = []
        for token_ids_single_content in batch_token_ids:
            selfie_chars = []
            for token_id_val in token_ids_single_content:
                char_token = self.ind2char.get(token_id_val)
                if char_token is None: selfie_chars.append(f"[UNK_{token_id_val}]"); continue
                selfie_chars.append(char_token)
            selfie_str = "".join(selfie_chars)
            decoded_smiles = f"RAW_SELFIES({selfie_str})"
            if selfie_str:
                try:
                    smiles_candidate = sf.decoder(selfie_str)
                    if smiles_candidate: decoded_smiles = smiles_candidate
                except Exception: pass
            smiles_out.append(decoded_smiles)
        return smiles_out

    # Keep this if PropGenerator needs it (if predictor was trained on parallel probs)
    def decode_parallel_log_probs_to_smiles(self, x_log_probs_flat: torch.Tensor) -> list[str]:
         batch_size = x_log_probs_flat.shape[0]
         token_ids_tensor = x_log_probs_flat.reshape(batch_size, self.padded_length, self.vocab_size).argmax(dim=2)
         # This reuses decode_token_ids_to_smiles by converting argmax output to list of lists
         return self.decode_token_ids_to_smiles(token_ids_tensor.tolist())


def load_molgen_transformer_wrapper_and_tokenizer(
    molgen_pretrained_path: str = None,
    device: torch.device = None
):
    effective_device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wrapper = MolGenTransformerWrapper(pretrained_model_path=molgen_pretrained_path, device=effective_device)
    wrapper.eval()
    tokenizer = SmilesTokenizerForWrapper(wrapper_instance=wrapper)
    return tokenizer, wrapper

# --- Generator classes (adapt based on your project structure) ---
# These need to be defined or imported correctly.
class BaseGenerator(nn.Module): # Example base
    def __init__(self): super().__init__(); self.latent_size = 0; self.reverse_size = 0
    def forward(self, z: Tensor) -> Tensor: raise NotImplementedError

class PropGenerator(BaseGenerator):
    def __init__(self, wrapper: MolGenTransformerWrapper, predictor_model: Predictor):
        super().__init__()
        self.wrapper = wrapper
        self.predictor_model = predictor_model
        self.latent_size = wrapper.latent_dim
        self.reverse_size = 1 

    def forward(self, z: Tensor) -> Tensor:
        # This assumes Predictor was trained on parallel log_probs output
        log_probs_flat = self.wrapper.decode_to_parallel_log_probs(z)
        probs_flat = log_probs_flat.exp() # Predictor expects probabilities
        return self.predictor_model(probs_flat)

class VAEGenerator(BaseGenerator): # Renamed from VAEGenerator for clarity if it's not a VAE
    def __init__(self, wrapper: MolGenTransformerWrapper):
        super().__init__()
        self.wrapper = wrapper
        self.latent_size = wrapper.latent_dim
        # Output of this generator for unsupervised PDE needs to be defined.
        # If PDE expects parallel log_probs (like old VAE):
        self.reverse_size = wrapper.max_len * wrapper.vocab_size 
        # If PDE works with token ID sequences, reverse_size is more complex or not fixed.

    def forward(self, z: Tensor) -> Tensor: # Or list[list[int]]
        # Option 1: If unsupervised PDE expects parallel log_probs
        return self.wrapper.decode_to_parallel_log_probs(z)
        
        # Option 2: If unsupervised PDE works with token ID sequences (more complex for PDE loss)
        # return self.wrapper.decode_to_token_ids_auto_regressive(z) # This would change reverse_size and PDE logic

# --- End Generator classes ---


def normalize(u_z, step_size, relative):
     if relative and u_z.norm().item() > 1e-6: # Avoid division by zero
         u_z = u_z / u_z.norm(dim=-1, keepdim=True) * step_size
     else: # Also handles case where norm is zero
         u_z = u_z * step_size
     return u_z

MODES = ["random", "random_1d", "fp", "limo", "chemspace", "wave_sup", "wave_unsup", "hj_sup", "hj_unsup"]
# Assume WAVEPDE_IDX_MAP and HJPDE_IDX_MAP are imported from ChemFlow.src.utils.scores

class Traversal:
    def __init__(
        self,
        method: str,
        prop: str,
        molgen_wrapper_path: str, # Made non-optional
        predictor_checkpoint_dir: str,
        pde_checkpoint_dir_base: str,
        step_size: float = 0.1,
        relative: bool = True,
        minimize: bool = False,
        k_idx: int | None = None,
        device: torch.device = None,
    ):
        self.method = method
        self.prop = prop
        self.molgen_wrapper_path = molgen_wrapper_path
        self.step_size = step_size
        self.relative = relative
        self.minimize = minimize
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        assert self.method in MODES

        self.dm, self.wrapper = load_molgen_transformer_wrapper_and_tokenizer(
            molgen_pretrained_path=self.molgen_wrapper_path, device=self.device
        )

        if self.method == "random":
            self.u_z = torch.randn(self.wrapper.latent_dim, device=self.device)
            return
        elif self.method == "random_1d":
            self.u_z = torch.zeros(self.wrapper.latent_dim, device=self.device)
            if self.wrapper.latent_dim > 0: # Avoid error if latent_dim is 0
                self.u_z[random.randint(0, self.wrapper.latent_dim - 1)] = (1 if random.random() < 0.5 else -1)
            return
        elif self.method == "chemspace": # Requires precomputed boundaries
            try:
                boundary_path_str = f"src/chemspace/boundaries_zmc/boundary_{self.prop}.npy" 
                boundary = np.load(boundary_path_str)
                if boundary.shape[-1] == self.wrapper.latent_dim:
                    self.u_z = torch.tensor(boundary, device=self.device, dtype=torch.float).squeeze(0)
                else: self.u_z = torch.randn(self.wrapper.latent_dim, device=self.device)
            except FileNotFoundError: self.u_z = torch.randn(self.wrapper.latent_dim, device=self.device)
            return

        predictor_input_size = self.wrapper.max_len * self.wrapper.vocab_size
        if self.method in {"limo", "fp", "wave_sup", "hj_sup"}:
            self.predictor = Predictor(predictor_input_size).to(self.device)
            predictor_path = Path(predictor_checkpoint_dir) / self.prop / "checkpoint.pt"
            if not predictor_path.exists(): raise FileNotFoundError(f"Predictor checkpoint not found: {predictor_path}")
            self.predictor.load_state_dict(torch.load(predictor_path, map_location=self.device))
            self.predictor.eval()
            for p_param in self.predictor.parameters(): p_param.requires_grad = False

        if self.method in {"limo", "fp"}:
            self.generator = PropGenerator(self.wrapper, self.predictor).to(self.device)
            return

        pde_name = self.method.split("_")[0]
        pde_checkpoint_path_str = ""
        if self.method in {"wave_sup", "hj_sup"}:
            self.generator = PropGenerator(self.wrapper, self.predictor).to(self.device)
            pde_checkpoint_path_str = str(Path(pde_checkpoint_dir_base) / f"{pde_name}pde_prop" / "zmc_transformer" / self.prop / "checkpoint.pt")
            self.idx = 0
            pde_k = 1
        else: # unsupervised
            self.generator = VAEGenerator(self.wrapper).to(self.device)
            pde_checkpoint_path_str = str(Path(pde_checkpoint_dir_base) / f"{pde_name}pde" / "zmc_transformer" / "checkpoint.pt")
            pde_k = 10
            if k_idx is not None: self.idx = k_idx
            elif pde_name == "wave": 0
            elif pde_name == "hj": 0
            else: raise ValueError(f"Unknown pde {pde_name}")
        
        if not Path(pde_checkpoint_path_str).exists(): raise FileNotFoundError(f"PDE checkpoint not found: {pde_checkpoint_path_str}")
        self.pde = load_wavepde(
            checkpoint=pde_checkpoint_path_str, generator=self.generator, k=pde_k,
            n_in=self.generator.latent_size, device=self.device
        )
        self.pde.eval()
        for p_param in self.pde.parameters(): p_param.requires_grad = False
        
        self.k = getattr(self.pde, 'k', pde_k) 
        self.half_range = getattr(self.pde, 'half_range', 10)

    def step(self, z: Tensor, t: int = 0, optimizer=None) -> Tensor:
        if t == 0: return torch.zeros_like(z)
        u_z_output = torch.zeros_like(z) # Default to no change

        if self.method in ["random", "random_1d", "chemspace"]:
            u_z_output = self.u_z.clone() # Use a clone
            if self.method == "chemspace" and u_z_output.ndim == 1 and z.ndim == 2:
                u_z_output = u_z_output.unsqueeze(0).expand_as(z)
            u_z_output = normalize(u_z_output, self.step_size, self.relative)
            if self.method == "chemspace" and self.minimize: u_z_output = -u_z_output
        elif self.method == "limo":
            if optimizer is not None: return self.limo_optimizer_step(optimizer, z) # LIMO updates z in-place
            z_req_grad = z.detach().clone().requires_grad_(True)
            grad_output = torch.autograd.grad(self.generator(z_req_grad).sum(), z_req_grad)[0]
            u_z_output = normalize(grad_output, self.step_size, self.relative)
            if self.minimize: u_z_output = -u_z_output
        elif self.method == "fp":
            z_req_grad = z.detach().clone().requires_grad_(True)
            grad_output = torch.autograd.grad(self.generator(z_req_grad).sum(), z_req_grad)[0]
            noise_term = torch.randn_like(grad_output) * np.sqrt(2 * self.step_size) * 0.1
            u_z_output = grad_output * self.step_size + noise_term
            if self.minimize: u_z_output = -u_z_output 
        else: # PDE methods
            _, u_z_pred = self.pde.inference(self.idx, z, t % self.half_range)
            u_z_output = normalize(u_z_pred, self.step_size, self.relative)
        return u_z_output

    def limo_optimizer_step(self, optimizer, z_tensor_with_grad): 
        optimizer.zero_grad()
        loss_val = -self.generator(z_tensor_with_grad).sum() 
        if self.minimize: loss_val = -loss_val
        loss_val.backward()
        optimizer.step()
        return torch.zeros_like(z_tensor_with_grad) # Indicates z was updated in-place

# --- Main Script Arguments and Execution ---
class Args(Tap):
    prop: str = "plogp"
    n: int = 100 
    steps: int = 100 
    method: Literal[
        "random", "random_1d", "fp", "limo", "chemspace",
        "wave_sup", "wave_unsup", "hj_sup", "hj_unsup",
    ] = "random"
    step_size: float = 0.1
    relative: bool = False
    
    # Paths now part of Traversal constructor, but can be exposed via CLI if needed
    molgen_wrapper_path: str = "/data/hdd1/users/akouk/moldecular_biology_dsit/MolTransformer_repo/MolTransformer/model/models/best_models/SS_model/Best_SS_GPU.pt"
    predictor_checkpoint_dir: str = "/data/hdd1/users/akouk/moldecular_biology_dsit/extend/checkpoints/prop_predictor_transformer"
    pde_checkpoint_dir_base: str = "extend/checkpoints_transformer_pde"
    initial_smiles_csv: str = "/data/hdd1/users/akouk/moldecular_biology_dsit/ChemFlow/data/interim/props/zinc250k.csv"
    output_dir: str = "extend/assets/optimization_results_transformer"


    def process_args(self): # model_name is now just for the output file
        self.model_name = f"{self.prop}_{self.method}_transformer"
        self.model_name += f"_{self.step_size}"
        self.model_name += "_relative" if self.relative else "_absolute"

if __name__ == "__main__":
    pandarallel.initialize(nb_workers=os.cpu_count(), progress_bar=True, verbose=2)
    args = Args().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    traversal_instance = Traversal(
        method=args.method,
        prop=args.prop,
        molgen_wrapper_path=args.molgen_wrapper_path,
        predictor_checkpoint_dir=args.predictor_checkpoint_dir,
        pde_checkpoint_dir_base=args.pde_checkpoint_dir_base,
        step_size=args.step_size,
        relative=args.relative,
        minimize=args.prop in MINIMIZE_PROPS,
        device=device,
    )

    df_initial_smiles = pd.read_csv(args.initial_smiles_csv)
    if args.prop not in df_initial_smiles.columns:
        raise ValueError(f"Property '{args.prop}' not found in {args.initial_smiles_csv}")
    df_initial_smiles = df_initial_smiles[["smiles", args.prop]]
    df_initial_smiles = df_initial_smiles.sort_values(args.prop, ascending=args.prop not in MINIMIZE_PROPS)
    initial_smiles_list = df_initial_smiles["smiles"].values[: args.n]
    
    x_tokens = traversal_instance.dm.encode(initial_smiles_list, device=device)
    z_initial, _, _ = traversal_instance.wrapper.encode(x_tokens)
    current_z = z_initial.clone().detach()

    # --- Optional: Debug initial z decoding ---
    # print("\n--- Debugging initial_z decoding in optimization.py ---")
    # num_debug_samples = min(5, args.n)
    # if num_debug_samples > 0:
    #     debug_z_batch = z_initial[:num_debug_samples]
    #     debug_smiles_original = initial_smiles_list[:num_debug_samples]
    #     print(f"Decoding {num_debug_samples} samples from z_initial...")
    #     debug_token_ids_lists = traversal_instance.wrapper.decode_to_token_ids_auto_regressive(debug_z_batch, debug_sample_idx=-1)
    #     debug_reconstructed_smiles = traversal_instance.dm.decode_token_ids_to_smiles(debug_token_ids_lists)
    #     for i_debug in range(num_debug_samples):
    #         print(f"  Original SMILES: {debug_smiles_original[i_debug]}")
    #         print(f"  Decoded from z_initial: {debug_reconstructed_smiles[i_debug]}")
    # print("--- END TEMP DEBUG FOR INITIAL Z ---\n")
    # --- End Optional Debug ---

    optimizer_limo = None
    if args.method == "limo":
        current_z.requires_grad_(True)
        optimizer_limo = torch.optim.Adam([current_z], lr=args.step_size)

    results_list = []
    for t_step in trange(args.steps, desc=f"Optimizing {args.prop} with {args.method}"):
        if args.method == "limo":
            traversal_instance.step(current_z, t_step + 1, optimizer=optimizer_limo) 
        else:
            u_z_delta = traversal_instance.step(current_z, t_step + 1) 
            current_z.add_(u_z_delta) # In-place addition
        
        batch_of_token_id_lists = traversal_instance.wrapper.decode_to_token_ids_auto_regressive(current_z, debug_sample_idx=-1) # No debug prints in loop
        decoded_smiles_batch = traversal_instance.dm.decode_token_ids_to_smiles(batch_of_token_id_lists)
        
        for i_mol, s_mol in enumerate(decoded_smiles_batch): 
            results_list.append({"idx": i_mol, "t": t_step, "smiles": s_mol})

    df_results = pd.DataFrame(results_list)
    df_unique_smiles = df_results[["smiles"]].drop_duplicates("smiles").copy() # Use .copy() to avoid SettingWithCopyWarning

    def calculate_prop_fn(series_row: pd.Series):
        # Ensure PROP_FN is accessible and contains args.prop
        if args.prop not in PROP_FN: raise KeyError(f"Property function for '{args.prop}' not found in PROP_FN.")
        series_row[args.prop] = PROP_FN[args.prop](series_row["smiles"])
        return series_row

    df_unique_smiles = df_unique_smiles.parallel_apply(calculate_prop_fn, axis=1)
    df_final_results = df_results.merge(df_unique_smiles, on="smiles", how="left")

    output_path_dir = Path(args.output_dir) 
    output_path_dir.mkdir(parents=True, exist_ok=True)
    output_csv_file = output_path_dir / f"{args.model_name}.csv"
    df_final_results.to_csv(output_csv_file, index=False)
    print(f"Results saved to {output_csv_file}")