import torch
from torch import nn, Tensor
import lightning as L
from lightning.pytorch import LightningDataModule
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.cli import LightningArgumentParser
from torch.utils.data import DataLoader, TensorDataset, random_split
from pathlib import Path
import random


from src.vae import load_vae, VAE
from src.pinn.generator import PropGenerator, VAEGenerator
from src.predictor import Predictor
from src.pinn.pde.neural_ode import NeuralODEFunc
from torchdiffeq import odeint_adjoint as odeint


class NeuralODEDataModule(LightningDataModule):
    def __init__(self, vae: VAE = None, n: int = 10_000, batch_size: int = 100): # Made vae optional
        super().__init__()
        self.save_hyperparameters(ignore="vae")
        self.n = n
        self.batch_size = batch_size
        self.dataset = TensorDataset(torch.randn(n, vae.latent_dim))
        self.train_data, self.val_data = random_split(
            self.dataset, [0.9, 0.1], generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size)


class NeuralODELightning(L.LightningModule):
    def __init__(
        self,
        latent_dim: int = 1024,
        guidance_generator: nn.Module = None,
        learning_rate: float = 1e-3,
        integration_time_step: float = 0.1,
        num_loss_integration_steps: int = 2,
        is_supervised: bool = False,
        minimize_property: bool = False,
        flow_reg_lambda: float = 0.01,
        time_embedding_dim: int = 128,
        hidden_dim_nde: int = 512,
        num_layers_nde: int = 3,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["guidance_generator"])
        self.neural_ode_func = NeuralODEFunc(latent_dim, time_embedding_dim, hidden_dim_nde, num_layers_nde)
        self.guidance_generator = guidance_generator 
        self.learning_rate = learning_rate
        self.integration_time_step = integration_time_step
        self.num_loss_integration_steps = num_loss_integration_steps
        self.is_supervised = is_supervised
        self.minimize_property = minimize_property
        self.flow_reg_lambda = flow_reg_lambda

    def forward_trajectory(self, z_initial: Tensor) -> Tensor:
        t_span = torch.linspace(0, self.integration_time_step, self.num_loss_integration_steps, device=z_initial.device)
        trajectory = odeint(self.neural_ode_func, z_initial, t_span, method='rk4', options={'step_size': self.integration_time_step / (self.num_loss_integration_steps * 2.0)})
        return trajectory

    def calculate_loss(self, trajectory: Tensor, z_initial: Tensor):
        loss = torch.tensor(0.0, device=z_initial.device)
        
        z_start_traj = trajectory[0]
        z_end_traj = trajectory[-1]

        t_start_val = 0.0
        t_end_val = self.integration_time_step

        if self.is_supervised:
            prop_start = self.guidance_generator(z_start_traj)
            prop_end = self.guidance_generator(z_end_traj)
            
            property_change = prop_end - prop_start
            if not self.minimize_property:
                loss = -property_change.mean()
            else:
                loss = property_change.mean()
        else:
            decoded_start = self.guidance_generator(z_start_traj)
            decoded_end = self.guidance_generator(z_end_traj)
            structural_change_loss = -torch.nn.functional.mse_loss(decoded_end, decoded_start.detach())
            loss = structural_change_loss
        
        if self.flow_reg_lambda > 0:
            t_start_tensor = torch.full((z_initial.size(0),), t_start_val, device=z_initial.device, dtype=z_initial.dtype)
            t_end_tensor = torch.full((z_initial.size(0),), t_end_val, device=z_initial.device, dtype=z_initial.dtype)

            dz_dt_at_start = self.neural_ode_func(t_start_tensor, z_start_traj)
            dz_dt_at_end = self.neural_ode_func(t_end_tensor, z_end_traj)
            
            flow_reg = (torch.norm(dz_dt_at_start, p=2, dim=-1).mean() + \
                        torch.norm(dz_dt_at_end, p=2, dim=-1).mean()) / 2.0
            loss += self.flow_reg_lambda * flow_reg
            
        return loss

    def training_step(self, batch: tuple[Tensor], batch_idx: int):
        (z_initial,) = batch
        trajectory = self.forward_trajectory(z_initial)
        loss = self.calculate_loss(trajectory, z_initial)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch: tuple[Tensor], batch_idx: int):
        (z_initial,) = batch
        trajectory = self.forward_trajectory(z_initial)
        val_loss = self.calculate_loss(trajectory, z_initial)
        self.log("val_loss", val_loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.neural_ode_func.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2, eta_min=1e-7)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

def parse_args_train_neural_ode():
    parser = LightningArgumentParser()
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("-e", "--epochs", type=int, default=40)
    parser.add_argument("-o", "--output_dir_base", type=str, default="ChemFlow/checkpoints/neural_ode")
    parser.add_argument("--prop_name", type=str, default="qed")
    parser.add_argument("--data_name", type=str, default="zmc")
    parser.add_argument("--is_supervised", action="store_true", default=False)
    parser.add_argument("--minimize_property", action="store_true", default=False)
    
    parser.add_lightning_class_args(NeuralODELightning, "model_params")
    parser.add_lightning_class_args(NeuralODEDataModule, "data_params")
    
    args = parser.parse_args()
    return args

def main_train_neural_ode():
    args = parse_args_train_neural_ode()
    L.seed_everything(args.seed)

    output_path = Path(args.output_dir_base) / args.data_name / args.prop_name
    output_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dm_vae, vae = load_vae(
        file_path=f"ChemFlow/data/processed/{args.data_name}.smi",
        model_path=f"ChemFlow/checkpoints/vae/{args.data_name}/checkpoint.pt", device=device
    )
    for p_vae in vae.parameters(): p_vae.requires_grad = False
    vae.eval()
    args.data_params.vae = vae
    data_module_neural_ode = NeuralODEDataModule(vae=vae)

    guidance_gen: nn.Module
    if args.is_supervised:
        predictor = Predictor(dm_vae.max_len * dm_vae.vocab_size)
        print(f"Loading predictor from checkpoints/prop_predictor/{args.prop_name}/checkpoint.pt")
        predictor.load_state_dict(
            torch.load(
                f"ChemFlow/checkpoints/prop_predictor/{args.prop_name}/checkpoint.pt",
                map_location=device,
                weights_only=False
            )
        )
        for p_pred in predictor.parameters(): p_pred.requires_grad = False
        predictor.eval()
        guidance_gen = PropGenerator(vae, predictor).to(device)
    else:
        guidance_gen = VAEGenerator(vae).to(device)
    
    for p_guidance in guidance_gen.parameters(): p_guidance.requires_grad = False
    guidance_gen.eval()

    model_specific_params = vars(args.model_params).copy() 

    explicitly_passed_keys = ['latent_dim', 'guidance_generator', 'is_supervised', 'minimize_property']
    for key in explicitly_passed_keys:
        if key in model_specific_params:
            del model_specific_params[key]

    lightning_model = NeuralODELightning(
        latent_dim=vae.latent_dim,
        guidance_generator=guidance_gen,
        is_supervised=args.is_supervised,
        minimize_property=args.minimize_property,
        **model_specific_params 
    ).to(device)
    
    trainer = L.Trainer(
        max_epochs=args.epochs,
        logger=[WandbLogger(project=f"soc_neural_ode_{args.data_name}", name=f"{args.prop_name}_{'sup' if args.is_supervised else 'unsup'}", entity="lakhs")],
        callbacks=[
            LearningRateMonitor(logging_interval="epoch"),
            ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, dirpath=output_path, save_last=True),
        ],
        accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1
    )
    trainer.fit(lightning_model, datamodule=data_module_neural_ode)

    best_model_path_ckpt = trainer.checkpoint_callback.best_model_path
    if best_model_path_ckpt:
        loaded_model = NeuralODELightning.load_from_checkpoint(
            best_model_path_ckpt,
            latent_dim=vae.latent_dim, 
            guidance_generator=guidance_gen,
        )
        torch.save(loaded_model.neural_ode_func.state_dict(), output_path / "checkpoint.pt")
    else: 
        torch.save(lightning_model.neural_ode_func.state_dict(), output_path / "checkpoint.pt")


if __name__ == "__main__":
    main_train_neural_ode()
