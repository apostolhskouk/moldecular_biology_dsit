import torch
from torch import Tensor
import numpy as np
import random
import warnings # Added

from cd2root import cd2root

cd2root()
# --- ChemFlow Imports ---
from ChemFlow.src.utils.scores import * # Contains normalize, MINIMIZE_PROPS, PROTEIN_FILES etc.
# from src.vae import load_vae # Removed
from ChemFlow.src.pinn.pde import load_wavepde # Keep for structure, but needs retraining
from custom.pinn.pde.generator import MolGenVAEGenerator, MolGenPropGenerator # Use new MolGen generators
from ChemFlow.src.pinn.generator import Predictor
# --- MolGen Interface Import ---
from custom.molgen_interface import MolGenInterface

MODES = [
    "random",
    "random_1d",
    "fp",
    "limo",
    # "chemspace", # Disabled - Incompatible with sequence latent space without retraining boundary
    "wave_sup",
    "wave_unsup",
    "hj_sup",
    "hj_unsup",
]

# --- Index Maps - Need Re-evaluation after Retraining with MolGen ---
# Keep structure but values are likely incorrect for MolGen latent space
WAVEPDE_IDX_MAP = {
    "drd2": 2, "gsk3b": 8, "jnk3": 0, "plogp": 9, "qed": 6, "sa": 9, "uplogp": 1,
    "1err": 2, "2iik": 4, # Added potential binding affinity keys
}
HJPDE_IDX_MAP = {
    "drd2": 3, "gsk3b": 1, "jnk3": 3, "plogp": 1, "qed": 7, "sa": 1, "uplogp": 1,
    "1err": 6, "2iik": 3, # Added potential binding affinity keys
}
# --- End Index Maps ---


# --- Updated Normalize Function ---
# Moved here for clarity or ensure src.utils.scores.normalize handles batches correctly.
# This version handles sequence tensors by normalizing the flattened view.
def normalize_sequence_update(x: Tensor, step_size: float | None = None, relative: bool = False):
    """ Normalizes the update tensor x. Handles sequence shapes (B, Seq, Emb)."""
    if step_size is None:
        return x

    if relative:
        return x * step_size # Element-wise scaling

    # For absolute step size, normalize based on overall magnitude (Frobenius norm)
    # Calculate norm across sequence and embedding dimensions for each batch element
    # Keep batch dim (dim=0) separate
    norm = torch.norm(x.view(x.shape[0], -1), p=2, dim=1, keepdim=True) # Shape (B, 1)
    # Reshape norm to match x's dimensions for broadcasting: (B, 1, 1)
    norm = norm.unsqueeze(-1).expand_as(x)
    # Avoid division by zero
    normalized_x = torch.where(norm > 1e-8, x / norm, torch.zeros_like(x))
    return normalized_x * step_size

class Traversal:
    """
    Uniformed class to perform 1 step of traversal in latent space,
    adapted for MolGenInterface and its sequence latent space.
    """

    method: str
    prop: str
    data_name: str # Used mainly for checkpoint paths, might need adjustment
    step_size: float
    relative: bool
    minimize: bool
    device: torch.device
    molgen_interface: MolGenInterface # Holds MolGen encoder/decoder interface

    def __init__(
        self,
        method: str,
        prop: str,
        data_name: str = "zmc", # Keep for checkpoint path consistency, may need updates
        step_size: float = 0.1,
        relative: bool = True,
        minimize: bool = False,
        k_idx: int | None = None,  # the index of unsupervised pde to use
        molgen_gm_config: dict | None = None, # Config for MolGen's GenerateMethods
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        self.method = method
        self.prop = prop
        self.data_name = data_name # Primarily affects checkpoint loading paths
        self.step_size = step_size
        self.relative = relative
        self.minimize = minimize
        self.device = device

        if method == "chemspace":
             raise NotImplementedError("ChemSpace method is not compatible with MolGen's sequence latent space without retraining boundaries.")
        assert self.method in MODES, f"mode must be one of {MODES}"

        # --- Initialize MolGen Interface ---
        print(f"Initializing MolGenInterface for Traversal (device: {self.device})...")
        self.molgen_interface = MolGenInterface(gm_config=molgen_gm_config, device=self.device)
        self.latent_seq_len = self.molgen_interface.latent_seq_len
        self.latent_embed_dim = self.molgen_interface.latent_embed_dim
        print("MolGenInterface initialized in Traversal.")

        # --- Generate u_z for random methods ---
        if self.method == "random":
            # Random update across the entire sequence embedding space
            self.u_z = torch.randn(
                self.latent_seq_len, self.latent_embed_dim, device=self.device
            )
            # Note: This generates ONE direction applied to the whole batch in step()
            # For per-sample random directions, generate inside step()
        elif self.method == "random_1d":
            # Modify only one position in the sequence
            self.u_z = torch.zeros(
                 self.latent_seq_len, self.latent_embed_dim, device=self.device
            )
            rand_seq_idx = random.randint(0, self.latent_seq_len - 1)
            direction = 1 if random.random() < 0.5 else -1
            # Apply +/- 1 to all embedding dims at the chosen sequence position
            self.u_z[rand_seq_idx, :] = direction
            # Again, this is ONE direction applied to the whole batch
        # elif self.method == "chemspace": # Disabled
        #     pass

        # --- Load Predictor if needed ---
        if self.method in {"limo", "fp", "wave_sup", "hj_sup"}:
            print("Loading Predictor model...")
            predictor_input_dim = self.molgen_interface.output_seq_len * self.molgen_interface.vocab_size
            self.predictor = Predictor(predictor_input_dim).to(self.device) # Use correct input dim
            try:
                predictor_ckpt_path = f"checkpoints/prop_predictor/{self.prop}/checkpoint.pt"
                self.predictor.load_state_dict(
                    torch.load(
                        predictor_ckpt_path,
                        map_location=self.device,
                        # weights_only=True, # Use if PyTorch version supports/requires
                    )
                )
                print(f"Predictor loaded from {predictor_ckpt_path}")
            except FileNotFoundError:
                 raise FileNotFoundError(f"Predictor checkpoint not found at {predictor_ckpt_path}. Please train the predictor for property '{self.prop}' using MolGen outputs.")
            self.predictor.eval() # Set to evaluation mode
            for p in self.predictor.parameters():
                p.requires_grad = False

        # --- Instantiate Generator ---
        # LIMO and FP need a generator for gradient calculation
        if self.method in {"limo", "fp"}:
            self.generator = MolGenPropGenerator(self.molgen_interface, self.predictor).to(self.device)
            print("Using MolGenPropGenerator for LIMO/FP.")
            return # Done initializing for these modes

        # --- Load PDE models if needed ---
        # Note: These paths point to models trained with VAE. They NEED retraining with MolGen.
        # This section will likely fail until retraining is done.
        if self.method not in ["random", "random_1d"]: # PDE modes remain
             pde_name = self.method.split("_")[0] # 'wave' or 'hj'
             is_supervised = "_sup" in self.method

             print(f"Attempting to load PDE model: {pde_name}, supervised={is_supervised}")
             print("WARNING: PDE models must be retrained using MolGenInterface to function correctly.")

             if is_supervised:
                 self.generator = MolGenPropGenerator(self.molgen_interface, self.predictor).to(self.device)
                 checkpoint_path = f"checkpoints/{pde_name}pde_prop/{self.data_name}/{self.prop}/checkpoint.pt"
                 k_pde = 1
                 self.idx = 0 # Supervised typically uses index 0
                 print(f"Using MolGenPropGenerator for supervised {pde_name} PDE.")
             else: # Unsupervised
                 self.generator = MolGenVAEGenerator(self.molgen_interface).to(self.device)
                 checkpoint_path = f"checkpoints/{pde_name}pde/{self.data_name}/checkpoint.pt"
                 k_pde = 10 # Default k for unsupervised

                 if k_idx is not None:
                     self.idx = k_idx
                 elif pde_name == "wave":
                     self.idx = WAVEPDE_IDX_MAP.get(self.prop, 0) # Default to 0 if prop not in map
                     print(f"Using WAVEPDE_IDX_MAP index: {self.idx} for property {self.prop}")
                 elif pde_name == "hj":
                     self.idx = HJPDE_IDX_MAP.get(self.prop, 0) # Default to 0
                     print(f"Using HJPDE_IDX_MAP index: {self.idx} for property {self.prop}")
                 else:
                     raise ValueError(f"Unknown unsupervised PDE type {pde_name}")
                 print(f"Using MolGenVAEGenerator for unsupervised {pde_name} PDE.")

             try:
                 self.pde = load_wavepde( # Assuming load_wavepde can load HJ models too if structured similarly
                     checkpoint=checkpoint_path,
                     generator=self.generator,
                     k=k_pde,
                     # Pass MolGen dimensions to ensure MLP inside PDE is correct IF load_wavepde uses them
                     n_in=self.latent_embed_dim, # Feature dim
                     # Need to modify load_wavepde if it needs sequence dim
                     device=self.device,
                 )
                 print(f"PDE model loaded from {checkpoint_path}")
                 self.pde.eval() # Set to evaluation mode
                 for p in self.pde.parameters():
                     p.requires_grad = False
                 self.k = self.pde.k
                 self.half_range = self.pde.half_range
             except FileNotFoundError:
                 warnings.warn(f"PDE checkpoint not found at {checkpoint_path}. Method '{self.method}' will not work until the PDE model is retrained with MolGenInterface.", RuntimeWarning)
                 self.pde = None # Mark as unavailable


    def step(self, z: Tensor, t: int = 0, optimizer=None) -> Tensor:
        """
        Perform 1 step of traversal in latent space z (B, SeqLen, EmbDim), return u_z.

        When t=0, return 0 tensor.
        """
        if t == 0:
            return torch.zeros_like(z)

        # Ensure z is on the correct device
        z = z.to(self.device)

        if self.method in ["random", "random_1d"]:
            # Expand the single direction u_z stored in self to the batch size
            u_z_batch = self.u_z.unsqueeze(0).expand(z.shape[0], -1, -1)
            u_z = normalize_sequence_update(u_z_batch, self.step_size, self.relative)
            # Note: for per-sample random directions, generate u_z here inside step()
            # e.g., u_z = torch.randn_like(z) for 'random'

        # elif self.method == "chemspace": # Disabled
        #     pass

        elif self.method == "limo":
            if optimizer is not None: # LIMO uses external optimizer steps
                return self.limo_optimizer_step(optimizer, z)

            # Calculate gradient for standard update step
            # Requires z to track gradients
            z_detached = z.detach().requires_grad_(True)
            # self.generator is MolGenPropGenerator
            property_pred = self.generator(z_detached)
            # Calculate gradient of summed property w.r.t latent sequence z
            grad_outputs = torch.ones_like(property_pred) # For scalar output
            u_z_grad = torch.autograd.grad(outputs=property_pred, inputs=z_detached, grad_outputs=grad_outputs)[0]
            u_z = normalize_sequence_update(u_z_grad, self.step_size, self.relative)
            if self.minimize:
                u_z = -u_z

        elif self.method == "fp": # Fokker-Planck (Langevin Dynamics)
             # Requires z to track gradients
            z_detached = z.detach().requires_grad_(True)
            # self.generator is MolGenPropGenerator
            property_pred = self.generator(z_detached)
            # Calculate gradient for drift term
            grad_outputs = torch.ones_like(property_pred)
            drift_grad = torch.autograd.grad(outputs=property_pred, inputs=z_detached, grad_outputs=grad_outputs)[0]
            # Apply drift term (scaled by step_size)
            drift_term = drift_grad * self.step_size # Assuming relative=True interpretation for drift
            if self.minimize:
                drift_term = -drift_term
            # Apply diffusion term
            # Check dimensions and sqrt(2*step_size) factor carefully
            noise_scale = np.sqrt(2 * self.step_size) * 0.1 # Match original code's scaling
            diffusion_term = torch.randn_like(z) * noise_scale
            # Combine terms
            u_z = drift_term + diffusion_term

        else: # PDE-based methods
            if self.pde is None:
                 warnings.warn(f"PDE model for method '{self.method}' not loaded (likely needs retraining). Returning zero update.", RuntimeWarning)
                 return torch.zeros_like(z)

            # self.pde should be loaded WavePDE/HJPDE model
            # It takes (k_idx, z, t_idx)
            # Ensure z does not require grad for PDE inference if not needed
            with torch.no_grad(): # PDE inference usually doesn't require grad w.r.t input z
                 _, u_z_pde = self.pde.inference(self.idx, z, t % self.half_range)

            u_z = normalize_sequence_update(u_z_pde, self.step_size, self.relative)
            # Note: PDE models internally handle maximizing/minimizing JVP term during training.
            # We assume the inference direction is correctly learned.
            # If explicit minimization is needed at traversal time (unlikely), add a sign flip here.

        return u_z.detach() # Return detached update vector

    def limo_optimizer_step(self, optimizer, z):
        """ Handles the backward pass and optimizer step for LIMO """
        if not z.requires_grad:
             raise ValueError("Input z must require gradients for LIMO optimizer step.")
        optimizer.zero_grad()
        # self.generator is MolGenPropGenerator
        property_pred = self.generator(z)
        # Objective is to MAXIMIZE property, so loss is NEGATIVE property
        loss = -property_pred.sum()
        if self.minimize:
            loss = -loss # Minimize negative property = Maximize property
        loss.backward()
        optimizer.step()
        # Return the gradient that was just applied (optional, might not be needed by caller)
        grad_applied = z.grad.clone() if z.grad is not None else torch.zeros_like(z)
        z.grad = None # Clear grad for next iteration
        return grad_applied