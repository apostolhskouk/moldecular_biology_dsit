import torch
from torch import Tensor
import numpy as np
import random

from cd2root import cd2root
cd2root()

from src.utils.scores import *
from src.vae import load_vae
from src.pinn.pde import load_wavepde, NeuralODEFunc
from src.pinn import PropGenerator, VAEGenerator
from src.predictor import Predictor
from torchdiffeq import odeint_adjoint as odeint
from experiments.train_latent_stepper import LatentStepperMLP
from src.rl_policy import PolicyNetwork

MODES = [
    "random",
    "random_1d",
    "fp",
    "limo",
    "chemspace",
    "wave_sup",
    "wave_unsup",
    "hj_sup",
    "hj_unsup",
    "neural_ode",
    "neural_ode_unsup",
    "latent_stepper",
    "hybrid_sup_unsup",
    "rl_policy"
]

WAVEPDE_IDX_MAP = {
    "plogp": 1, "sa": 1, "qed": 1, "drd2": 9, "jnk3": 4, "gsk3b": 0, "uplogp": 1, "1err": 2, "2iik": 4,
}
HJPDE_IDX_MAP = {
    "plogp": 0, "sa": 0, "qed": 9, "drd2": 2, "jnk3": 3, "gsk3b": 8, "uplogp": 0, "1err": 6, "2iik": 3,
}


def normalize_dz(x: Tensor, step_size=None, relative=False):
    if step_size is None:
        return x
    norm_val = torch.norm(x, dim=-1, keepdim=True)
    if torch.any(norm_val == 0):
        return x 
    if relative:
        return x * step_size
    return x / norm_val * step_size


class Traversal:
    method: str
    prop: str
    data_name: str
    step_size: float
    relative: bool
    minimize: bool
    device: torch.device
    idx: int # for k_idx or pde_idx
    policy_net_rl: PolicyNetwork = None # For RL policy
    
    
    def __init__(
        self,
        method: str,
        prop: str,
        data_name: str = "zmc",
        step_size: float = 0.1,
        relative: bool = True,
        minimize: bool = False,
        k_idx: int | None = None,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        alpha_hybrid: float = 0.5, # Weight for supervised component in hybrid mode
        hybrid_unsup_pde_type: str = "wave", \
        hybrid_unsup_k_idx: int | None = None, 
    ):
        self.rl_action_scale = 0.1
        self.method = method
        self.prop = prop
        self.data_name = data_name
        self.step_size = step_size
        self.relative = relative
        self.minimize = minimize
        self.device = device
        self.idx = 0 # Default
        self.alpha_hybrid = alpha_hybrid
        self.hybrid_unsup_pde_type = hybrid_unsup_pde_type
        self.unsupervised_k_idx_for_hybrid = 0 # Default, will be set below
        
        assert self.method in MODES, f"mode must be one of {MODES}"

        self.dm, self.vae = load_vae(
            file_path=f"ChemFlow/data/processed/{self.data_name}.smi",
            model_path=f"ChemFlow/checkpoints/vae/{self.data_name}/checkpoint.pt",
            latent_dim=1024,
            embedding_dim=128,
            device=self.device,
        )
        for p_vae in self.vae.parameters():
            p_vae.requires_grad = False
        self.vae.eval()
        if self.method == "rl_policy":
            self.policy_net_rl = PolicyNetwork(
                latent_dim=self.vae.latent_dim, 
                action_dim=self.vae.latent_dim
            ).to(self.device)
            policy_checkpoint_path = f"ChemFlow/checkpoints/rl_policy/{self.data_name}/{self.prop}/policy_network_final.pt"
            try:
                self.policy_net_rl.load_state_dict(
                    torch.load(policy_checkpoint_path, map_location=self.device)
                )
            except FileNotFoundError:
                print(f"ERROR: RL Policy checkpoint not found at {policy_checkpoint_path}")
                raise
            for p_policy in self.policy_net_rl.parameters(): p_policy.requires_grad = False
            self.policy_net_rl.eval()
        elif self.method == "random":
            self.rand_u_z_base = torch.randn(self.vae.latent_dim, device=self.device)
        elif self.method == "random_1d":
            self.rand_u_z_base = torch.zeros(self.vae.latent_dim, device=self.device)
            self.rand_u_z_base[random.randint(0, self.vae.latent_dim - 1)] = (1 if random.random() < 0.5 else -1)
        elif self.method == "chemspace":
            boundary_np = np.load(f"src/chemspace/boundaries_{self.data_name}/boundary_{self.prop}.npy")
            self.chemspace_u_z_base = torch.tensor(boundary_np, device=self.device).squeeze()
        elif self.method == "neural_ode":
            self.neural_ode_func = NeuralODEFunc(latent_dim=self.vae.latent_dim).to(self.device)
            checkpoint_path = f"ChemFlow/checkpoints/neural_ode/{self.data_name}/{self.prop}/checkpoint.pt"
            self.neural_ode_func.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            for p_node in self.neural_ode_func.parameters():
                p_node.requires_grad = False
            self.neural_ode_func.eval()
            return
        elif self.method == "neural_ode_unsup":
            self.neural_ode_func = NeuralODEFunc(latent_dim=self.vae.latent_dim).to(self.device)
            checkpoint_path = f"ChemFlow/checkpoints/neural_ode_unsup/{self.data_name}/{self.prop}/checkpoint.pt"
            self.neural_ode_func.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            for p_node in self.neural_ode_func.parameters():
                p_node.requires_grad = False
            self.neural_ode_func.eval()
            return
        elif self.method == "latent_stepper":
            self.stepper_mlp = LatentStepperMLP(latent_dim=self.vae.latent_dim) # Add hidden_dims, dropout from args if needed
            stepper_checkpoint_path = f"ChemFlow/checkpoints/latent_stepper/{self.data_name}/{self.prop}/stepper_mlp.pt"
            self.stepper_mlp.load_state_dict(torch.load(stepper_checkpoint_path, map_location=self.device))
            self.stepper_mlp.to(self.device)
            self.stepper_mlp.eval()
            
        if self.method in {"limo", "fp", "wave_sup", "hj_sup", "hybrid_sup_unsup"}:
            self.predictor = Predictor(self.dm.max_len * self.dm.vocab_size).to(self.device)
            self.predictor.load_state_dict(
                torch.load(f"ChemFlow/checkpoints/prop_predictor/{self.prop}/checkpoint.pt", map_location=self.device)
            )
            for p_pred in self.predictor.parameters():
                p_pred.requires_grad = False
            self.predictor.eval()

        if self.method in {"limo", "fp"}:
            self.generator = PropGenerator(self.vae, self.predictor).to(self.device)
        elif self.method in {"wave_sup", "hj_sup"}:
            self.generator = PropGenerator(self.vae, self.predictor).to(self.device)
            pde_name = self.method.split("_")[0]
            self.pde = load_wavepde(
                checkpoint=f"ChemFlow/checkpoints/{pde_name}pde_prop/{self.data_name}/{self.prop}/checkpoint.pt",
                generator=self.generator, k=1, device=self.device
            )
            self.idx = 0 # Supervised usually has k=1
        elif self.method in {"wave_unsup", "hj_unsup"}:
            self.generator = VAEGenerator(self.vae).to(self.device)
            pde_name = self.method.split("_")[0]
            self.pde = load_wavepde(
                checkpoint=f"ChemFlow/checkpoints/{pde_name}pde/{self.data_name}/checkpoint.pt",
                generator=self.generator, k=10, device=self.device
            )
            if k_idx is not None:
                self.idx = k_idx
            elif pde_name == "wave":
                self.idx = WAVEPDE_IDX_MAP.get(self.prop, 0)
            elif pde_name == "hj":
                self.idx = HJPDE_IDX_MAP.get(self.prop, 0)
        elif self.method == "hybrid_sup_unsup":
            # Supervised part (LIMO-like gradient flow for simplicity, or could be wave_sup/hj_sup)
            self.generator_sup_hybrid = PropGenerator(self.vae, self.predictor).to(self.device)
            for p_gen in self.generator_sup_hybrid.parameters(): p_gen.requires_grad = False
            self.generator_sup_hybrid.eval()
            
            # Unsupervised part (e.g., wave_unsup)
            self.generator_unsup_hybrid = VAEGenerator(self.vae).to(self.device)
            unsup_pde_checkpoint = f"ChemFlow/checkpoints/{self.hybrid_unsup_pde_type}pde/{self.data_name}/checkpoint.pt"
            self.pde_unsup_hybrid = load_wavepde(
                checkpoint=unsup_pde_checkpoint,
                generator=self.generator_unsup_hybrid, k=10, device=self.device
            )
            for p_pde in self.pde_unsup_hybrid.parameters(): p_pde.requires_grad = False
            self.pde_unsup_hybrid.eval()
            self.k_unsup_hybrid = self.pde_unsup_hybrid.k
            self.half_range_unsup_hybrid = self.pde_unsup_hybrid.half_range
            
            # Determine k_idx for the unsupervised part of hybrid
            if hybrid_unsup_k_idx is not None:
                 self.unsupervised_k_idx_for_hybrid = hybrid_unsup_k_idx
            elif self.hybrid_unsup_pde_type == "wave":
                self.unsupervised_k_idx_for_hybrid = WAVEPDE_IDX_MAP.get(self.prop, 0) 
            elif self.hybrid_unsup_pde_type == "hj":
                self.unsupervised_k_idx_for_hybrid = HJPDE_IDX_MAP.get(self.prop, 0)
            else: # Default if type is unknown, though should be validated
                self.unsupervised_k_idx_for_hybrid = 0
            print(f"Hybrid mode using unsupervised k_idx: {self.unsupervised_k_idx_for_hybrid} for {self.hybrid_unsup_pde_type} PDE.")
            
        if hasattr(self, 'generator'):
            for p_gen in self.generator.parameters():
                p_gen.requires_grad = False
            self.generator.eval()
        if hasattr(self, 'pde'):
            for p_pde in self.pde.parameters():
                p_pde.requires_grad = False
            self.pde.eval()
            self.k = self.pde.k
            self.half_range = self.pde.half_range


    def step(self, z: Tensor, t: int = 0, optimizer=None) -> Tensor:
        if t == 0 and self.method != "neural_ode" and self.method != "latent_stepper": 
            return torch.zeros_like(z)
        

        u_z_final = torch.zeros_like(z)
        if self.method == "rl_policy":
            if t == 0: # No action at initial step for consistency, or could take one
                return torch.zeros_like(z)
            with torch.no_grad():
                mean_action_raw, _ = self.policy_net_rl(z) # Get mean and std
                
                action_u_z_scaled = torch.tanh(mean_action_raw) * self.rl_action_scale 
            u_z_final = action_u_z_scaled
            return u_z_final
        
        elif self.method == "random":
            u_z_final = normalize_dz(self.rand_u_z_base.clone(), self.step_size, self.relative)
        elif self.method == "random_1d":
            u_z_final = normalize_dz(self.rand_u_z_base.clone(), self.step_size, self.relative)
        elif self.method == "chemspace":
            u_z_intermediate = self.chemspace_u_z_base.clone()
            if self.minimize:
                u_z_intermediate = -u_z_intermediate
            u_z_final = normalize_dz(u_z_intermediate, self.step_size, self.relative)
        elif self.method == "limo":
            z_req_grad = z.detach().clone().requires_grad_(True)
            prop_val = self.generator(z_req_grad).sum()
            grad_z = torch.autograd.grad(prop_val, z_req_grad)[0]
            
            u_z_intermediate = grad_z
            if self.minimize:
                u_z_intermediate = -u_z_intermediate
            u_z_final = normalize_dz(u_z_intermediate, self.step_size, self.relative)
        elif self.method == "latent_stepper":
            with torch.no_grad():
                delta_z = self.stepper_mlp(z)
            u_z_intermediate = delta_z
            if self.minimize:
                pass 

            u_z_final = normalize_dz(u_z_intermediate, self.step_size, self.relative)
        elif self.method == "fp":
            z_req_grad = z.detach().clone().requires_grad_(True)
            potential_energy = self.generator(z_req_grad).sum()
            grad_potential = torch.autograd.grad(potential_energy, z_req_grad)[0]
            
            drift = grad_potential
            if self.minimize:
                drift = -grad_potential
            
            drift_scaled = drift * self.step_size
            noise = torch.randn_like(z) * np.sqrt(2 * self.step_size) * 0.1 
            u_z_final = drift_scaled + noise
        elif self.method == "neural_ode" or self.method == "neural_ode_unsup":
            if t == 0: # For Neural ODE, t=0 means initial state, no step taken yet.
                 return torch.zeros_like(z)
            t_span = torch.tensor([0.0, self.step_size], device=self.device, dtype=z.dtype)
           
            current_z = z.detach().clone() 
            with torch.no_grad():
                # Integrate the ODE for a duration of self.step_size
                # The trajectory will have 2 points: z(0) and z(self.step_size)
                z_next_trajectory = odeint(self.neural_ode_func, current_z, t_span, method='rk4', options={'step_size': self.step_size/5.0})
            z_next = z_next_trajectory[-1]
            u_z_final = z_next - current_z
        elif self.method == "hybrid_sup_unsup":
            # Supervised component (LIMO-like gradient)
            z_req_grad_sup = z.detach().clone().requires_grad_(True)
            prop_val_sup = self.generator_sup_hybrid(z_req_grad_sup).sum()
            u_z_sup = torch.autograd.grad(prop_val_sup, z_req_grad_sup, retain_graph=False)[0]
            if self.minimize: u_z_sup = -u_z_sup
            u_z_sup_normalized = normalize_dz(u_z_sup, self.step_size, self.relative)

            # Unsupervised component (e.g., wave_unsup)
            time_for_unsup_pde = t % self.half_range_unsup_hybrid
            # Use the k_idx determined for the hybrid's unsupervised part
            _, u_z_unsup_raw = self.pde_unsup_hybrid.inference(self.unsupervised_k_idx_for_hybrid, z, time_for_unsup_pde)
            u_z_unsup_normalized = normalize_dz(u_z_unsup_raw, self.step_size, self.relative)
            
            # Combine: alpha for supervised, (1-alpha) for unsupervised
            u_z_final = self.alpha_hybrid * u_z_sup_normalized + (1.0 - self.alpha_hybrid) * u_z_unsup_normalized
            # Final normalization of the combined vector (optional, but good for consistent step magnitude)
            u_z_final = normalize_dz(u_z_final, self.step_size, self.relative)
        else: # PDE methods
            # self.idx is set in __init__ for unsupervised, or is 0 for supervised
            # t is the discrete step count for the traversal
            time_for_pde = t % self.half_range 
            _, u_z_pde = self.pde.inference(self.idx, z, time_for_pde)
            u_z_intermediate = u_z_pde
            # PDE training should handle minimization/maximization direction
            u_z_final = normalize_dz(u_z_intermediate, self.step_size, self.relative)
            
        return u_z_final