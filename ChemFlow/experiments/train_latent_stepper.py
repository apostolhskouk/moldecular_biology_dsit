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

from ChemFlow.src.vae import load_vae, VAE
from ChemFlow.src.pinn.generator import PropGenerator, VAEGenerator
from ChemFlow.src.predictor import Predictor
from ChemFlow.src.utils.scores import MINIMIZE_PROPS # For checking minimization

# Re-using the DataModule from Neural ODE as it provides random z vectors
class LatentStepperDataModule(LightningDataModule):
    def __init__(self, vae: VAE = None, n: int = 10_000, batch_size: int = 100):
        super().__init__()
        self.save_hyperparameters(ignore="vae")
        self.n = n
        self.batch_size = batch_size
        # latent_dim will be available from vae.latent_dim after vae is passed
        self.latent_dim = vae.latent_dim if vae else None 
        if self.latent_dim:
            self.dataset = TensorDataset(torch.randn(n, self.latent_dim))
            self.train_data, self.val_data = random_split(
                self.dataset, [0.9, 0.1], generator=torch.Generator().manual_seed(42)
            )
        else:
            # Placeholder, will be properly initialized in setup or when vae is available
            self.dataset = None
            self.train_data = None
            self.val_data = None


    def setup_with_vae(self, vae: VAE):
        self.latent_dim = vae.latent_dim
        self.dataset = TensorDataset(torch.randn(self.n, self.latent_dim))
        self.train_data, self.val_data = random_split(
            self.dataset, [0.9, 0.1], generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        if self.train_data is None:
            raise RuntimeError("LatentStepperDataModule not fully set up. Call setup_with_vae(vae) first.")
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, drop_last=True)

    def val_dataloader(self):
        if self.val_data is None:
            raise RuntimeError("LatentStepperDataModule not fully set up. Call setup_with_vae(vae) first.")
        return DataLoader(self.val_data, batch_size=self.batch_size)

class LatentStepperMLP(nn.Module):
    def __init__(self, latent_dim: int, hidden_dims: list[int] = None, dropout_rate: float = 0.1):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [latent_dim, latent_dim // 2, latent_dim // 4]

        layers = []
        current_dim = latent_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.Mish())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = h_dim
        layers.append(nn.Linear(current_dim, latent_dim))
        # Optional: Tanh to bound delta_z, but might limit step magnitude.
        # layers.append(nn.Tanh()) # Output between -1 and 1, then scale externally if needed.

        self.mlp = nn.Sequential(*layers)

    def forward(self, z: Tensor) -> Tensor:
        return self.mlp(z)

class LatentStepperLightning(L.LightningModule):
    def __init__(
        self,
        latent_dim: int, 
        guidance_generator: nn.Module, 
        learning_rate: float = 1e-3,
        is_supervised: bool = True, 
        minimize_property: bool = False, 
        delta_z_reg_lambda: float = 0.001, 
        mlp_hidden_dims: list[int] = None,
        mlp_dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["guidance_generator"]) # guidance_generator is not serializable well
        self.stepper_mlp = LatentStepperMLP(latent_dim, mlp_hidden_dims, mlp_dropout_rate)
        self.guidance_generator = guidance_generator 
        self.learning_rate = learning_rate
        self.is_supervised = is_supervised
        self.minimize_property = minimize_property
        self.delta_z_reg_lambda = delta_z_reg_lambda

    def calculate_loss(self, z_initial: Tensor):
        loss = torch.tensor(0.0, device=z_initial.device)
        
        delta_z = self.stepper_mlp(z_initial)
        z_final = z_initial + delta_z

        if self.is_supervised:
            prop_initial = self.guidance_generator(z_initial.detach()) 
            prop_final = self.guidance_generator(z_final)
            
            property_improvement = prop_final - prop_initial
            if self.minimize_property:
                loss = property_improvement.mean() 
            else:
                loss = -property_improvement.mean() 
        else:
            decoded_initial = self.guidance_generator(z_initial.detach())
            decoded_final = self.guidance_generator(z_final)
            structural_change_loss = -torch.nn.functional.mse_loss(decoded_final, decoded_initial)
            loss = structural_change_loss

        if self.delta_z_reg_lambda > 0:
            delta_z_norm = torch.norm(delta_z, p=2, dim=-1).mean()
            loss += self.delta_z_reg_lambda * delta_z_norm
            
        return loss, delta_z 

    def training_step(self, batch: tuple[Tensor], batch_idx: int):
        (z_initial,) = batch
        loss, delta_z = self.calculate_loss(z_initial)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("train_delta_z_norm_mean", torch.norm(delta_z, p=2, dim=-1).mean(), on_step=True, on_epoch=False, sync_dist=True)
        return loss

    def validation_step(self, batch: tuple[Tensor], batch_idx: int):
        (z_initial,) = batch
        val_loss, delta_z = self.calculate_loss(z_initial)
        self.log("val_loss", val_loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val_delta_z_norm_mean", torch.norm(delta_z, p=2, dim=-1).mean(), on_epoch=True, sync_dist=True)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.stepper_mlp.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}

def parse_args_train_latent_stepper():
    parser = LightningArgumentParser()
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("-e", "--epochs", type=int, default=50)
    parser.add_argument("-o", "--output_dir_base", type=str, default="checkpoints/latent_stepper")
    parser.add_argument("--prop_name", type=str, default="qed", help="Property to optimize.")
    parser.add_argument("--data_name", type=str, default="zmc", help="Dataset name for VAE.")
    
    # Add only the CLI-configurable model parameters
    parser.add_argument("--model_params.learning_rate", type=float, default=1e-3)
    parser.add_argument("--model_params.is_supervised", type=bool, default=True) # Or use action='store_true'/'store_false'
    parser.add_argument("--model_params.delta_z_reg_lambda", type=float, default=0.001)
    parser.add_argument("--model_params.mlp_hidden_dims", type=list[int], default=None) # Will be parsed as string, handle in main
    parser.add_argument("--model_params.mlp_dropout_rate", type=float, default=0.1)
    
    # Add CLI-configurable data parameters
    parser.add_argument("--data_params.n", type=int, default=10000)
    parser.add_argument("--data_params.batch_size", type=int, default=100)
    
    args = parser.parse_args()

    # Post-process mlp_hidden_dims if provided as a string
    if args.model_params.mlp_hidden_dims is not None and isinstance(args.model_params.mlp_hidden_dims, str):
        try:
            import json
            args.model_params.mlp_hidden_dims = json.loads(args.model_params.mlp_hidden_dims)
        except json.JSONDecodeError:
            raise ValueError("mlp_hidden_dims must be a valid JSON string representing a list of integers, e.g., \"[1024,512,256]\"")

    return args

def main_train_latent_stepper():
    args = parse_args_train_latent_stepper()
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
    
    # Create data_params dict from args
    data_params_dict = {
        "n": args.data_params.n,
        "batch_size": args.data_params.batch_size
    }
    data_module_stepper = LatentStepperDataModule(vae=vae, **data_params_dict)

    guidance_gen: nn.Module
    is_supervised_actual = args.model_params.is_supervised 

    if is_supervised_actual:
        predictor = Predictor(dm_vae.max_len * dm_vae.vocab_size)
        predictor_path = f"ChemFlow/checkpoints/prop_predictor/{args.prop_name}/checkpoint.pt"
        print(f"Loading predictor from {predictor_path}")
        predictor.load_state_dict(
            torch.load(predictor_path, map_location=device, weights_only=False)
        )
        for p_pred in predictor.parameters(): p_pred.requires_grad = False
        predictor.eval()
        guidance_gen = PropGenerator(vae, predictor).to(device)
    else:
        guidance_gen = VAEGenerator(vae).to(device)
    
    for p_guidance in guidance_gen.parameters(): p_guidance.requires_grad = False
    guidance_gen.eval()

    minimize_prop_actual = args.prop_name in MINIMIZE_PROPS

    # Create model_init_kwargs from args.model_params namespace
    model_init_kwargs = {
        "learning_rate": args.model_params.learning_rate,
        "is_supervised": args.model_params.is_supervised,
        "delta_z_reg_lambda": args.model_params.delta_z_reg_lambda,
        "mlp_hidden_dims": args.model_params.mlp_hidden_dims,
        "mlp_dropout_rate": args.model_params.mlp_dropout_rate,
        # Programmatically set parameters
        "latent_dim": vae.latent_dim,
        "guidance_generator": guidance_gen,
        "minimize_property": minimize_prop_actual,
    }
    
    lightning_model = LatentStepperLightning(**model_init_kwargs).to(device)
    
    trainer = L.Trainer(
        max_epochs=args.epochs,
        logger=[WandbLogger(project=f"soc_latent_stepper_{args.data_name}", name=f"{args.prop_name}_{'sup' if is_supervised_actual else 'unsup'}", entity="lakhs")], 
        callbacks=[
            LearningRateMonitor(logging_interval="epoch"),
            ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, dirpath=output_path, save_last=True),
        ],
        accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1
    )
    trainer.fit(lightning_model, datamodule=data_module_stepper)

    best_model_path_ckpt = trainer.checkpoint_callback.best_model_path
    save_target_path = output_path / "stepper_mlp.pt"
    if best_model_path_ckpt:
        loaded_lightning_model = LatentStepperLightning.load_from_checkpoint(
            best_model_path_ckpt,
            latent_dim=vae.latent_dim, 
            guidance_generator=guidance_gen 
        )
        torch.save(loaded_lightning_model.stepper_mlp.state_dict(), save_target_path)
        print(f"Saved best LatentStepperMLP state_dict to {save_target_path}")
    else: 
        torch.save(lightning_model.stepper_mlp.state_dict(), save_target_path)
        print(f"Saved last LatentStepperMLP state_dict to {save_target_path}")

if __name__ == "__main__":
    main_train_latent_stepper()
