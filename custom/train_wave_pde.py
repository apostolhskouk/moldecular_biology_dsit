import torch
from torch import nn, Tensor # Added nn, Tensor
import lightning as L
from lightning.pytorch import LightningDataModule, LightningModule # Added LightningModule
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
# from lightning.pytorch.cli import LightningArgumentParser # Removed, not parsing args

from torch.utils.data import DataLoader, TensorDataset, random_split
import random # Added for WavePDEModel step
from pathlib import Path

from cd2root import cd2root

cd2root()

from ChemFlow.src.predictor import Predictor # Assuming this is your Predictor class
from ChemFlow.src.utils.scores import *
from ChemFlow.src.pinn.pde import WavePDEModel

# --- Generator Abstract Class (from your WavePDEModel file) ---
from abc import ABC
class Generator(nn.Module, ABC):
    latent_size: int
    reverse_size: int
    def forward(self, z: Tensor) -> Tensor: ...

# --- MolgenPropGenerator ---
class MolgenPropGenerator(Generator):
    def __init__(self, predictor_model: Predictor, latent_dim: int = 12030):
        super().__init__()
        self.model = predictor_model
        self.latent_size = latent_dim # This is the flattened Molgen latent dim
        self.reverse_size = 1 # Output of predictor is a single property value

    def forward(self, z_molgen_flat: Tensor) -> Tensor:
        return self.model(z_molgen_flat)


class DataModule(LightningDataModule):
    def __init__(self, latent_file_path: str = "custom/assets/molgen.pt", n_samples_to_use: int = -1, batch_size: int = 100, seed: int = 42):
        super().__init__()
        self.save_hyperparameters()
        self.latent_file_path = latent_file_path
        self.n_samples_to_use = n_samples_to_use # -1 means use all
        self.batch_size = batch_size
        self.seed = seed
        self.latent_dim = 12030 # MolTransformer flattened latent dim

    def setup(self, stage=None):
        all_latents = torch.load(self.latent_file_path) # Shape (N, 401, 30)
        
        # Flatten the latents for the WavePDE (which expects 1D latent vectors per sample)
        # The WavePDE's internal 'z' will be these flattened vectors.
        all_latents_flat = all_latents.view(all_latents.size(0), -1) # Shape (N, 12030)

        if self.n_samples_to_use > 0 and self.n_samples_to_use < len(all_latents_flat):
            # If you want to use a subset for faster testing/debugging
            indices = torch.randperm(len(all_latents_flat), generator=torch.Generator().manual_seed(self.seed))[:self.n_samples_to_use]
            selected_latents = all_latents_flat[indices]
        else:
            selected_latents = all_latents_flat

        self.dataset = TensorDataset(selected_latents)
        
        num_total = len(self.dataset)
        num_train = int(num_total * 0.9)
        num_val = num_total - num_train
        
        self.train_data, self.val_data = random_split( # Changed test_data to val_data
            self.dataset, [num_train, num_val], generator=torch.Generator().manual_seed(self.seed)
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=4, pin_memory=True)


def main():
    seed = 42
    epochs = 30 # Example, adjust as needed
    output_base = "custom/checkpoints/wavepde_molgen"
    predictor_checkpoint_base = "custom/checkpoints/prop_predictor_moltransformer"
    
    
    for wave_pde_model in ["hj"]:
        wave_pde_model_args = {
            "k": 10,
            "time_steps": 20,
            "pde_function": wave_pde_model, # or "burgers", "reaction_diffusion"
            "normalize": None, # or a float value e.g. 1.0
            "learning_rate": 1e-3,
        }
        wandb_entity = "lakhs" # Your W&B entity

        L.seed_everything(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        datamodel = DataModule()
        # datamodel.prepare_data() # Not strictly needed if setup does all
        datamodel.setup()

        properties_to_train = ['plogp']

        for prop_name in properties_to_train:
            output_path = Path(output_base) / wave_pde_model_args["pde_function"] / prop_name
            output_path.mkdir(parents=True, exist_ok=True)

            # 1. Load the pre-trained Molgen-based property predictor
            molgen_latent_dim = 12030 # Flattened Molgen latent dim
            predictor = Predictor(molgen_latent_dim).to(device)
            predictor_checkpoint_path = Path(predictor_checkpoint_base) / prop_name / "checkpoint.pt"
            
            if not predictor_checkpoint_path.exists():
                continue # Skip if predictor checkpoint doesn't exist

            predictor.load_state_dict(torch.load(predictor_checkpoint_path, map_location=device))
            predictor.eval()
            for p in predictor.parameters():
                p.requires_grad = False

            # 2. Create the MolgenPropGenerator
            molgen_generator = MolgenPropGenerator(predictor_model=predictor, latent_dim=molgen_latent_dim).to(device)

            # 3. Initialize WavePDEModel
            current_wave_pde_args = wave_pde_model_args.copy()
            current_wave_pde_args["minimize_jvp"] = prop_name in MINIMIZE_PROPS or prop_name in PROTEIN_FILES
            current_wave_pde_args["n_in"] = molgen_generator.latent_size # Ensure n_in matches generator

            model = WavePDEModel(generator=molgen_generator, **current_wave_pde_args).to(device)

            # 4. Setup Trainer
            trainer_callbacks = [
                LearningRateMonitor(logging_interval="epoch"),
                ModelCheckpoint(
                    monitor="val/loss",
                    mode="min",
                    save_top_k=1,
                    dirpath=output_path,
                    save_last=True,
                    filename=f"{prop_name}-{{epoch:02d}}-{{val/loss:.2f}}"
                ),
            ]
            
            logger_name = f"{prop_name}_norm_{model.pde.normalize}"
            if model.pde.normalize is None:
                logger_name = f"{prop_name}_norm_False"


            trainer_config = {
                "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
                "devices": -1 if torch.cuda.is_available() else 1,
                "max_epochs": epochs,
                "callbacks": trainer_callbacks,
                "strategy": "ddp_find_unused_parameters_true" if torch.cuda.device_count() > 1 else "auto",
            }

            if wandb_entity:
                trainer_config["logger"] = WandbLogger(
                    project=f"soc_molgen_{model.pde.pde_function}pde_prop",
                    entity=wandb_entity,
                    name=logger_name,
                )
            
            trainer = L.Trainer(**trainer_config)

            trainer.fit(model, datamodel)

            # Load best checkpoint for saving the WavePDE part
            best_model_path = trainer.checkpoint_callback.best_model_path
            if best_model_path and Path(best_model_path).exists():
                model_loaded = WavePDEModel.load_from_checkpoint(
                    best_model_path, generator=molgen_generator, **current_wave_pde_args
                ).to(device)
                wavepde_to_save = model_loaded.pde
                torch.save(wavepde_to_save.state_dict(), output_path / "checkpoint.pt")
            else: # Save last if best not found (e.g. training interrupted)
                last_ckpt_path = output_path / "last.ckpt"
                if last_ckpt_path.exists():
                    model_loaded = WavePDEModel.load_from_checkpoint(
                        last_ckpt_path, generator=molgen_generator, **current_wave_pde_args
                    ).to(device)
                    wavepde_to_save = model_loaded.pde
                    torch.save(wavepde_to_save.state_dict(), output_path / "checkpoint.pt")
                else: # Fallback to saving current model's PDE state
                    torch.save(model.pde.state_dict(), output_path / "checkpoint.pt")


if __name__ == "__main__":
    main()