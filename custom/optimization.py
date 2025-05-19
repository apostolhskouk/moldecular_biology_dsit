import torch
from torch import nn, Tensor
import pandas as pd
from accelerate.utils import set_seed
import os
from tap import Tap
from tqdm import trange, tqdm
from typing import Literal
from pathlib import Path
import random
import numpy as np

from cd2root import cd2root

cd2root()

from ChemFlow.src.predictor import Predictor
from ChemFlow.src.pinn.pde import WavePDE
from ChemFlow.src.utils.scores import *

from MolTransformer_repo.MolTransformer.generative.generative_method import GenerateMethods


MODES_MOLGEN = [
    "random",
    "random_1d",
    "fp",
    "limo",
    "wave_sup",
    "wave_unsup",
    "hj_sup",
    "hj_unsup",
]

WAVEPDE_IDX_MAP_MOLGEN = {
    "drd2": 2, "gsk3b": 8, "jnk3": 0, "plogp": 9, "qed": 6, "sa": 9, "uplogp": 1
}
HJPDE_IDX_MAP_MOLGEN = {
    "drd2": 3, "gsk3b": 1, "jnk3": 3, "plogp": 1, "qed": 7, "sa": 1, "uplogp": 1
}

from abc import ABC
class BaseGenerator(nn.Module, ABC):
    latent_size: int
    reverse_size: int
    def forward(self, z: Tensor) -> Tensor: ...

class MolgenPropGenerator(BaseGenerator):
    def __init__(self, predictor_model: Predictor, latent_dim: int = 12030):
        super().__init__()
        self.model = predictor_model
        self.latent_size = latent_dim
        self.reverse_size = 1
    def forward(self, z_molgen_flat: Tensor) -> Tensor:
        return self.model(z_molgen_flat)

def normalize_molgen(u_z: Tensor, step_size: float, relative: bool, z: Tensor = None):
    if relative and z is not None:
        return u_z / (torch.norm(u_z, dim=-1, keepdim=True) + 1e-8) * torch.norm(z, dim=-1, keepdim=True) * step_size
    return u_z / (torch.norm(u_z, dim=-1, keepdim=True) + 1e-8) * step_size


class MolgenTraversal:
    method: str
    prop: str
    step_size: float
    relative: bool
    minimize: bool
    device: torch.device
    molgen_latent_dim: int

    def __init__(
        self,
        method: str,
        prop: str,
        step_size: float = 0.01,
        relative: bool = False,
        minimize: bool = False,
        k_idx: int | None = None,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        molgen_latent_dim: int = 12030,
        predictor_checkpoint_base: str = "custom/checkpoints/prop_predictor_moltransformer",
        wavepde_checkpoint_base: str = "custom/checkpoints/wavepde_molgen",
    ):
        self.method = method
        self.prop = prop
        self.step_size = step_size
        self.relative = relative
        self.minimize = minimize
        self.device = device
        self.molgen_latent_dim = molgen_latent_dim

        assert self.method in MODES_MOLGEN, f"mode must be one of {MODES_MOLGEN}"

        self.gm = GenerateMethods(gpu_mode=torch.cuda.is_available())

        if self.method == "random":
            self.u_z_static = torch.randn(self.molgen_latent_dim, device=self.device)
            return
        elif self.method == "random_1d":
            self.u_z_static = torch.zeros(self.molgen_latent_dim, device=self.device)
            self.u_z_static[random.randint(0, self.molgen_latent_dim - 1)] = (1 if random.random() < 0.5 else -1)
            return

        self.predictor = Predictor(self.molgen_latent_dim).to(self.device)
        predictor_path = Path(predictor_checkpoint_base) / self.prop / "predictor_best_weights.pt"
        self.predictor.load_state_dict(torch.load(predictor_path, map_location=self.device))
        for p in self.predictor.parameters():
            p.requires_grad = False
        self.generator = MolgenPropGenerator(self.predictor, self.molgen_latent_dim).to(self.device)

        if self.method in {"limo", "fp"}:
            return

        pde_name = self.method.split("_")[0]
        pde_checkpoint_path = Path(wavepde_checkpoint_base) / pde_name / self.prop / "checkpoint.pt"
        
        # These are example defaults, WavePDE ideally saves its config or you pass them
        k_pde = 1 if "_sup" in self.method else 10
        time_steps_pde = 20 
        normalize_pde = None 

        self.pde = WavePDE(
            k=k_pde,
            generator=self.generator,
            time_steps=time_steps_pde,
            n_in=self.molgen_latent_dim,
            pde_function=pde_name,
            normalize=normalize_pde,
            minimize_jvp=self.minimize
        ).to(self.device)
        self.pde.load_state_dict(torch.load(pde_checkpoint_path, map_location=self.device))
        for p in self.pde.parameters():
            p.requires_grad = False

        if "_sup" in self.method:
            self.idx = 0
        else:
            if k_idx is not None:
                self.idx = k_idx
            elif pde_name == "wave":
                self.idx = WAVEPDE_IDX_MAP_MOLGEN.get(self.prop, 0)
            elif pde_name == "hj":
                self.idx = HJPDE_IDX_MAP_MOLGEN.get(self.prop, 0)
            else:
                raise ValueError(f"Unknown pde {pde_name} for unsupervised index mapping")
        
        self.k = self.pde.k
        self.half_range = self.pde.half_range


    def decode_to_smiles(self, z_flat_batch: Tensor) -> list[str]:
        if z_flat_batch.ndim == 2:
            z_reshaped = z_flat_batch.view(z_flat_batch.size(0), 401, 30)
        else:
            z_reshaped = z_flat_batch
        z_np = z_reshaped.detach().cpu().numpy()
        smiles_dict = self.gm.latent_space_2_strings(z_np)
        return smiles_dict['SMILES']

    def step(self, z_flat: Tensor, t: int = 0, optimizer=None) -> Tensor:
        if t == 0 and not (self.method == "limo" and optimizer is not None): # LIMO needs to run step 0 for grad
             return torch.zeros_like(z_flat)

        if self.method in ["random", "random_1d"]:
            u_z = self.u_z_static.unsqueeze(0).repeat(z_flat.shape[0], 1) # Make batch consistent
            u_z = normalize_molgen(u_z, self.step_size, self.relative, z_flat if self.relative else None)
        elif self.method == "limo":
            if optimizer is not None:
                return self.limo_optimizer_step(optimizer, z_flat)
            z_flat_g = z_flat.detach().requires_grad_(True)
            u_z = torch.autograd.grad(self.generator(z_flat_g).sum(), z_flat_g)[0]
            u_z = normalize_molgen(u_z, self.step_size, self.relative, z_flat_g if self.relative else None)
            if self.minimize:
                u_z = -u_z
        elif self.method == "fp":
            z_flat_g = z_flat.detach().requires_grad_(True)
            grad_val = torch.autograd.grad(self.generator(z_flat_g).sum(), z_flat_g)[0]
            u_z = grad_val * self.step_size + torch.randn_like(z_flat_g) * np.sqrt(2 * self.step_size) * 0.1
            if self.minimize:
                u_z = -u_z
        else: # PDE methods
            # WavePDE.inference returns (z_next, u_z_like_velocity_or_delta)
            # We are interested in the u_z part for explicit step
            _, u_z_from_pde = self.pde.inference(self.idx, z_flat, t % self.half_range)
            u_z = normalize_molgen(u_z_from_pde, self.step_size, self.relative, z_flat if self.relative else None)
            # The PDE's u_z should already incorporate minimization/maximization direction if trained for it.
            # If not, an explicit sign flip might be needed based on self.minimize,
            # but typically supervised PDEs learn the correct direction.
            # For unsupervised, it depends on the learned dynamics.
            # Let's assume PDE inference provides u_z in the "optimizing" direction.
            # If PDE's u_z is just a "dynamic" and not "optimizing", then:
            # pred_after_pde_step = self.generator(z_flat + u_z_from_pde)
            # pred_before = self.generator(z_flat)
            # if (self.minimize and (pred_after_pde_step > pred_before).any()) or \
            #    (not self.minimize and (pred_after_pde_step < pred_before).any()):
            #    u_z = -u_z # Flip if PDE step goes wrong direction (crude)

        return u_z.detach()

    def limo_optimizer_step(self, optimizer, z_flat_with_grad):
        optimizer.zero_grad()
        # z_flat_with_grad must have requires_grad=True
        loss = -self.generator(z_flat_with_grad).sum() # Maximize by minimizing negative
        if self.minimize:
            loss = -loss # Minimize by minimizing positive
        loss.backward()
        optimizer.step()
        return z_flat_with_grad.grad # Return grad for potential logging, not used for update


class Args(Tap):
    prop: str = "plogp"
    n: int = 10
    steps: int = 1000
    method: Literal[
        "random", "random_1d", "fp", "limo",
        "wave_sup", "wave_unsup", "hj_sup", "hj_unsup",
    ] = "fp"
    step_size: float = 0.01
    relative_step: bool = False # New arg for clarity

    molgen_latent_file: str = "/data/hdd1/users/akouk/moldecular_biology_dsit/custom/assets/molgen.pt"
    predictor_checkpoint_base: str = "/data/hdd1/users/akouk/moldecular_biology_dsit/custom/checkpoints/prop_predictor_molgen"
    wavepde_checkpoint_base: str = "/data/hdd1/users/akouk/moldecular_biology_dsit/custom/checkpoints/wavepde_molgen"
    output_dir: str = "custom/optimization_results_molgen"

    def process_args(self):
        self.model_name = f"{self.prop}_{self.method}_s{self.step_size}"
        self.model_name += "_rel" if self.relative_step else "_abs"

if __name__ == "__main__":
    args = Args().parse_args()
    args.process_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    traversal = MolgenTraversal(
        method=args.method,
        prop=args.prop,
        step_size=args.step_size,
        relative=args.relative_step,
        minimize=args.prop in MINIMIZE_PROPS,
        device=device,
        predictor_checkpoint_base=args.predictor_checkpoint_base,
        wavepde_checkpoint_base=args.wavepde_checkpoint_base,
    )

    all_latents_full_shape = torch.load(args.molgen_latent_file, map_location=device)
    all_latents_flat = all_latents_full_shape.view(all_latents_full_shape.size(0), -1)

    selected_indices = None
    if args.n > 0 and args.n < len(all_latents_flat):
        selected_indices = torch.randperm(len(all_latents_flat), generator=torch.Generator().manual_seed(42))[:args.n]
        z_current_flat = all_latents_flat[selected_indices].clone().detach().to(device)
    else:
        z_current_flat = all_latents_flat.clone().detach().to(device)

    optimizer_limo = None
    if args.method == "limo":
        z_current_flat.requires_grad_(True)
        optimizer_limo = torch.optim.Adam([z_current_flat], lr=args.step_size)

    results_data = []
    pbar_steps = trange(args.steps, desc=f"Optimizing {args.prop} via {args.method}")

    for t_step in pbar_steps:
        if args.method == "limo":
            traversal.step(z_current_flat, t_step, optimizer=optimizer_limo)
        else:
            delta_z = traversal.step(z_current_flat, t_step)
            z_current_flat.add_(delta_z) # In-place update

        if t_step % 10 == 0 or t_step == args.steps - 1:
            current_smiles_batch = traversal.decode_to_smiles(z_current_flat)
            for i, s_mol in enumerate(current_smiles_batch):
                original_idx_val = selected_indices[i].item() if selected_indices is not None else i
                results_data.append({
                    "original_idx": original_idx_val,
                    "batch_idx": i,
                    "step": t_step,
                    "smiles": s_mol,
                })
        pbar_steps.set_postfix({"prop": args.prop})

    df_results = pd.DataFrame(results_data)
    df_unique_smiles = df_results[["smiles"]].drop_duplicates("smiles").copy()

    calculated_props_list = []
    for sml_val in tqdm(df_unique_smiles["smiles"], desc="Calculating properties"):
        prop_val_calc = None
        if sml_val and PROP_FN.get(args.prop):
            prop_val_calc = PROP_FN[args.prop](sml_val)
        calculated_props_list.append(prop_val_calc)
    df_unique_smiles[args.prop] = calculated_props_list

    df_results = df_results.merge(df_unique_smiles, on="smiles", how="left")

    output_path_dir = Path(args.output_dir)
    output_path_dir.mkdir(parents=True, exist_ok=True)
    csv_filename_path = output_path_dir / f"{args.model_name}.csv"
    df_results.to_csv(csv_filename_path, index=False)