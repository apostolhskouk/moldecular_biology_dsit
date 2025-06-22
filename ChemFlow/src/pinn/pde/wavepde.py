import torch
from torch import nn, Tensor
from lightning.pytorch import LightningModule
from torch.autograd import grad
from torch.autograd.functional import jvp

from attrs import define, asdict

import random

from ChemFlow.src.pinn import AuxClassifier, Generator
from ChemFlow.src.pinn.pde import PDE, MLP


@define
class WavePDEResult:
    loss: Tensor
    energy: Tensor
    latent1: Tensor
    latent2: Tensor
    loss_ic: Tensor
    loss_pde: Tensor
    loss_jvp: Tensor
    loss_cls: Tensor = None


class WavePDE(PDE):
    def __init__(
        self,
        k: int,
        generator: Generator = None,
        time_steps: int = 20,
        n_in: int = 1024,
        pde_function: str = "wave",
        normalize: float | None = None,
        minimize_jvp: bool = False,
    ):
        assert time_steps > 0
        assert pde_function in {"wave", "hj"}

        super().__init__()

        self.k = k
        self.time_steps = time_steps
        self.half_range = time_steps // 2
        self.pde_function = pde_function
        self.normalize = normalize
        self.minimize_jvp = minimize_jvp

        self.generator = generator

        self.mlp = nn.ModuleList([MLP(n_in=n_in, n_out=1) for _ in range(k)])
        self.c = nn.Parameter(torch.ones(k))

    def forward(self, idx: int, z_input: Tensor, t_target: int) -> WavePDEResult: # Renamed args for clarity
        assert t_target < self.half_range, f"t_target={t_target} must be less than {self.half_range}"

        c_param = self.c[idx] # Current wave speed parameter

        # z_loop is the tensor that evolves through time steps i
        # z_input is assumed to be prepared by WavePDEModel.step (leaf, requires_grad=True)
        z_loop = z_input.clone().requires_grad_(True) # This creates a new leaf node for this forward pass

        loss_ic = torch.tensor(0.0, device=z_loop.device)
        loss_pde_sum = torch.tensor(0.0, device=z_loop.device)
        loss_jvp = torch.tensor(0.0, device=z_loop.device)
        
        energy, latent1, latent2 = None, None, None
        pde_terms_count = 0

        for i in range(self.half_range + 2): # Loop over discrete time points
            if i >= self.half_range and latent2 is not None:
                break

            # Force gradient calculation context for this block
            with torch.enable_grad():  
                _t_tensor = torch.full((1,), float(i), dtype=z_loop.dtype, device=z_loop.device, requires_grad=True)
                
                # Ensure z_loop still requires grad (belt-and-suspenders, z_loop.requires_grad_(True) above should suffice)
                if not z_loop.requires_grad:
                    print(f"WARNING: z_loop lost requires_grad before MLP call at i={i}. Forcing it back.")
                    z_loop.requires_grad_(True)

                # --- Start Enhanced Debugging ---
                if i == 0: # Only print for the first iteration, or where it typically fails
                    print(f"\n--- Debug Info: WavePDE.forward (iter {i}) ---")
                    print(f"z_input: id={id(z_input)}, device={z_input.device}, requires_grad={z_input.requires_grad}, is_leaf={z_input.is_leaf}")
                    print(f"z_loop (after clone().requires_grad_(True)): id={id(z_loop)}, device={z_loop.device}, requires_grad={z_loop.requires_grad}, is_leaf={z_loop.is_leaf}, grad_fn={z_loop.grad_fn}")
                    print(f"_t_tensor: requires_grad={_t_tensor.requires_grad}")
                    print(f"Global grad enabled: torch.is_grad_enabled() = {torch.is_grad_enabled()}")
                    try:
                        is_inference = torch._C._is_inference_mode_enabled()
                        print(f"Global inference mode: torch._C._is_inference_mode_enabled() = {is_inference}")
                        if is_inference:
                            print("CRITICAL: inference_mode is ENABLED. This will prevent requires_grad_() from working as expected and grad() calls.")
                    except AttributeError:
                        print("Could not check inference_mode via torch._C (might be version specific).")

                u = self.mlp[idx](z_loop, _t_tensor)  # Potential energy u(z,t)
                
                if i == 0: # More debug after MLP call
                    print(f"u (after mlp call): requires_grad={u.requires_grad}, grad_fn={u.grad_fn.name() if u.grad_fn else 'None'}")


                # Gradient of u w.r.t. z (spatial derivative component)
                # This is line 70 (or equivalent) where the error occurs
                u_z = grad(u.sum(), z_loop, create_graph=True)[0]

                if i == 0: # Initial condition
                    loss_ic = u_z.square().mean()

                if i < self.half_range: # PDE residual loss for t < T/2
                    pde_terms_count +=1
                    u_t = grad(u.sum(), _t_tensor, create_graph=True)[0]
                    
                    if self.pde_function == "wave":
                        u_tt = grad(u_t.sum(), _t_tensor, create_graph=True)[0]
                        u_zz = grad(u_z.sum(), z_loop, create_graph=True)[0]
                        pde_residual = u_tt - (c_param**2) * u_zz
                        loss_pde_sum += pde_residual.square().mean()

                    elif self.pde_function == "hj":
                        hamiltonian = 0.5 * u_z.square().sum(dim=-1, keepdim=True)
                        pde_residual = u_t.unsqueeze(0) + hamiltonian
                        loss_pde_sum += pde_residual.square().mean()
                    else:
                        raise NotImplementedError

                # JVP loss calculation
                if i == t_target + 1:
                    energy = u.detach() 
                    latent1 = z_loop.detach().clone()
                    _, jvp_value = jvp(self.generator, z_loop, v=u_z, create_graph=True)
                    if jvp_value.shape[1] == 1:
                        loss_jvp = (jvp_value.sign() * jvp_value.square()).mean()
                        if self.minimize_jvp:
                            loss_jvp = -loss_jvp
                    else:
                        loss_jvp = jvp_value.square().mean()

                elif i == t_target + 2:
                    latent2 = z_loop.detach().clone()

                # Update z_loop: z_{i+1} = z_i + step_vector
                # This step must also be within the `enable_grad` context if subsequent u_z relies on it.
                if self.normalize is not None:
                    norm_u_z = u_z.norm(dim=1, keepdim=True)
                    # Ensure u_z has a grad_fn if z_loop is to continue tracking grads correctly
                    # print(f"WavePDE iter {i} u_z: requires_grad={u_z.requires_grad}, grad_fn name: {u_z.grad_fn.name() if u_z.grad_fn else None}")
                    z_loop = z_loop + u_z / (norm_u_z + 1e-8) * self.normalize
                else:
                    z_loop = z_loop + u_z
                # After this update, z_loop is no longer a leaf, it will have a grad_fn.
                # print(f"WavePDE iter {i} z_loop (post-update): requires_grad={z_loop.requires_grad}, is_leaf={z_loop.is_leaf}, grad_fn name: {z_loop.grad_fn.name() if z_loop.grad_fn else None}")
        
        avg_loss_pde = loss_pde_sum / pde_terms_count if pde_terms_count > 0 else torch.tensor(0.0, device=z_input.device)
        total_loss = loss_ic + avg_loss_pde - loss_jvp

        return WavePDEResult(
            loss=total_loss,
            energy=energy,
            latent1=latent1,
            latent2=latent2,
            loss_ic=loss_ic,
            loss_pde=avg_loss_pde,
            loss_jvp=loss_jvp,
        )
        
        
    def inference(self, idx: int, z: Tensor, t: Tensor | int) -> tuple[Tensor, Tensor]:
        z = z.clone().requires_grad_()

        if isinstance(t, int):
            t = torch.full((1,), t, dtype=z.dtype, device=z.device)

        u = self.mlp[idx](z, t)  # (b, 1)
        u_z = grad(u.sum(), z, create_graph=True)[0]  # (b, n_in)

        return u, u_z


class WavePDEModel(LightningModule):
    def __init__(
        self,
        pde_function: str = "wave",
        learning_rate: float = 1e-3,
        normalize: float | None = None,
        # CLI controllable args
    ):
        super().__init__()
        self.save_hyperparameters(
            "pde_function",
            "learning_rate",
            "normalize",
        )
        self.pde_function_val = pde_function
        self.learning_rate_val = learning_rate
        self.normalize_val = normalize
        self.generator_instance = None
        self.k_val = None
        self.minimize_jvp_val = None
        self.n_in_val = None
        self.pde = None

    def setup_core_logic(self, generator, k: int, minimize_jvp: bool, n_in: int, time_steps: int = 20):
        self.generator_instance = generator
        self.k_val = k
        self.minimize_jvp_val = minimize_jvp
        self.n_in_val = n_in
        self.pde = WavePDE(
            k=self.k_val,
            generator=self.generator_instance,
            time_steps=time_steps,
            n_in=self.n_in_val,
            pde_function=self.pde_function_val,
            normalize=self.normalize_val,
            minimize_jvp=self.minimize_jvp_val,
        )
        if self.k_val > 1 and hasattr(self.generator_instance, "reverse_size"):
            self.aux_cls = AuxClassifier(self.generator_instance.reverse_size, self.k_val)
            self.loss_fn_aux = nn.CrossEntropyLoss()

    def forward(self, idx: int, z: Tensor, t: int, positive: bool = True):
        if self.pde is None:
            raise RuntimeError("WavePDEModel not fully set up. Call setup_core_logic.")
        result = self.pde(idx, z, t)
        if self.k_val > 1 and hasattr(self, 'aux_cls'):
            if positive:
                mol_shifted = self.generator_instance(result.latent1)
                mol_shifted2 = self.generator_instance(result.latent2)
            else:
                mol_shifted = self.generator_instance(2 * z - result.latent2)
                mol_shifted2 = self.generator_instance(2 * z - result.latent1)
            pred = self.aux_cls(torch.cat([mol_shifted, mol_shifted2], dim=1))
            result.loss_cls = self.loss_fn_aux(
                pred,
                torch.full((z.shape[0],), idx, device=self.device)
            )
            result.loss += result.loss_cls
        return result

    def step(self, batch: tuple[Tensor], batch_idx: int, deterministic: bool = False, stage: str = None):
        if self.pde is None:
            raise RuntimeError("WavePDEModel not fully set up. Call setup_core_logic.")
        (z_from_batch,) = batch # Rename to avoid confusion

        # Ensure z is a leaf and requires grad before entering the PDE logic
        # This z will be passed to self.pde
        z = z_from_batch.clone().detach().requires_grad_(True)

        if deterministic:
            idx = batch_idx % self.k_val
            t = batch_idx % self.pde.half_range
            positive = batch_idx % 2 == 0
        else:
            idx = random.randint(0, self.k_val - 1)
            t = random.randint(0, self.pde.half_range - 1)
            positive = random.random() < 0.5
        
        # self() calls WavePDEModel.forward, which calls self.pde.forward(idx, z, t)
        results = self(idx, z, t, positive=positive) 
        
        log_dict_payload = {
            f"{stage}/loss": results.loss,
            f"{stage}/loss_ic": results.loss_ic,
            f"{stage}/loss_pde": results.loss_pde,
            f"{stage}/loss_jvp": results.loss_jvp,
        }
        if hasattr(results, 'loss_cls') and results.loss_cls is not None: # Check for None
            log_dict_payload[f"{stage}/loss_cls"] = results.loss_cls
        self.log_dict(
            log_dict_payload,
            on_step=(stage=="train"),
            on_epoch=True,
            prog_bar=True,
            sync_dist=True # Removed if not DDP/FSDP, but harmless if single GPU
        )
        return results.loss

    def training_step(self, batch: tuple[Tensor], batch_idx: int):
        return self.step(batch, batch_idx, stage="train")

    def validation_step(self, batch: tuple[Tensor], batch_idx: int):
        return self.step(batch, batch_idx, deterministic=True, stage="val")

    def on_validation_epoch_start(self) -> None:
        if self.training:
            torch.set_grad_enabled(True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate_val)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=30, T_mult=2, eta_min=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }


def load_wavepde(
    checkpoint: str = "ChemFlow/checkpoints/wavepde/zinc250k/checkpoint.pt",
    generator: Generator = None,
    k: int = 10,
    time_steps: int = 20,
    n_in: int = 1024,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> WavePDE:
    model = WavePDE(k, generator, time_steps, n_in).to(device)
    model.load_state_dict(torch.load(checkpoint))

    for param in model.parameters():
        param.requires_grad = False

    return model


if __name__ == "__main__":
    # wavepde = WavePDE(1)

    from cd2root import cd2root

    cd2root()

    from ChemFlow.src.vae.vae import VAE
    from ChemFlow.src.vae.datamodule import MolDataModule
    from ChemFlow.src.pinn.generator import VAEGenerator

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dm = MolDataModule()
    dm.prepare_data()
    dm.setup()

    obj = torch.load("ChemFlow/checkpoints/vae/zinc250k/checkpoint.pt")
    vae = VAE(dm.max_len, dm.vocab_size).to(device)
    vae.load_state_dict(torch.load("ChemFlow/checkpoints/vae/zinc250k/checkpoint.pt"))
    vae.eval()

    generator = VAEGenerator(vae).to(device)

    wavepde = WavePDE(1, generator).to(device)

    z = torch.randn(10, 1024, device=device)

    print(wavepde(0, z, 7))
    print(wavepde.inference(0, z, 7))
