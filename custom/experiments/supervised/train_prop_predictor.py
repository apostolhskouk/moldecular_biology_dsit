from dotenv import load_dotenv
import os

load_dotenv()  # Loads variables from .env into os.environ

# Change working directory
project_path = os.getenv("PROJECT_PATH") or "."
os.chdir(project_path)
import os
import pandas as pd
import numpy as np
# from pandarallel import pandarallel # Not needed here
from lightning.pytorch.callbacks import ModelCheckpoint
from torchmetrics import PearsonCorrCoef

import torch
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader, TensorDataset, random_split # Keep random_split
from lightning import LightningModule, LightningDataModule
import lightning as L
from lightning.pytorch.cli import LightningArgumentParser
from lightning.pytorch.tuner.tuning import Tuner
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)
from tqdm import tqdm

from pathlib import Path
from typing import Optional,List

# from dict_hash import sha256 # Not used

from scipy.stats import linregress # Keep for validation metric

from cd2root import cd2root

cd2root()
# --- ChemFlow Imports ---
# from src.vae import load_vae, VAE, MolDataModule # Removed VAE imports
from ChemFlow.src.utils.scores import * # Keep for constants like PROTEIN_FILES if needed
from ChemFlow.src.predictor import Predictor
# --- MolGen Interface Import ---
# Import indirectly only to get dimensions if needed, but prefer passing them directly
from custom.molgen_interface import MolGenInterface # Maybe needed just for default dims
# --- Determine MolGen Dimensions (Ideally passed via config or inferred) ---
# These should match the MolGen model used in prepare_random_data
_temp_interface = MolGenInterface()
DEFAULT_MOLGEN_OUTPUT_SEQ_LEN = _temp_interface.output_seq_len
DEFAULT_MOLGEN_VOCAB_SIZE = _temp_interface.model_vocab_size
del _temp_interface # Clean up temporary instance

class Model(LightningModule):
    """ Lightning Module for training the property predictor. """
    def __init__(
        self,
        # Removed vae argument
        predictor_input_dim: int, # Explicitly require the input dimension
        optimizer: str = "sgd",
        learning_rate: float = 1e-3,
        hidden_sizes: Optional[List[int]] = None, # Allow passing hidden sizes for Predictor
    ):
        super().__init__()
        # self.vae = vae # Removed
        self.save_hyperparameters(ignore=["predictor_input_dim"]) # Save args except dim

        self.optimizer_name = optimizer
        self.learning_rate = learning_rate

        # Initialize Predictor with the correct input dimension
        print(f"Initializing Predictor with input dimension: {predictor_input_dim}")
        self.cls = Predictor(predictor_input_dim, hidden_sizes=hidden_sizes)
        self.loss_fn = nn.MSELoss()

        self.val_pearson = PearsonCorrCoef()

    def forward(self, x: Tensor):
        # Direct prediction from input x (which are the decoder logits)
        return self.cls(x)

    def training_step(self, batch, batch_idx):
        x, y = batch # x: logits, y: normalized property
        y_hat = self(x).squeeze() # Predict property
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, batch_size=x.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        loss = self.loss_fn(y_hat, y)

        # Store predictions and targets for epoch-end calculation
        self.val_pearson.update(y_hat.detach(), y)


        self.log("val_loss", loss, prog_bar=True, on_epoch=True, batch_size=x.size(0))
        return loss

    def on_validation_epoch_start(self) -> None:
        self.val_pearson.reset()

    def on_validation_epoch_end(self) -> None:
        r = self.val_pearson.compute()

        if not torch.is_tensor(r) or not torch.isfinite(r):
            r = 0.0
        else:
            r = r.item()

        self.log("lin_reg_r", r, prog_bar=True)

    def configure_optimizers(self):
        if self.optimizer_name == "adam":
            optimizer = optim.AdamW(self.cls.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == "sgd":
            optimizer = optim.SGD(
                self.cls.parameters(), lr=self.learning_rate, weight_decay=1e-2
            )
        else:
            raise ValueError(f"Optimizer {self.optimizer_name} not supported")

        # Using MultiStepLR as in original code
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[10, 15], gamma=0.1 # Milestones might need tuning
        )

        return [optimizer], [scheduler]


class PropDataset(LightningDataModule):
    """ Lightning DataModule for loading pre-generated logits and properties. """
    def __init__(
        self,
        # Removed vae, dm args
        prop: str = "plogp",
        n: int = 110, # Should match the 'n' used in prepare_random_data
        seed: int = 42, # Should match the 'seed' used in prepare_random_data
        binding_affinity: bool = False,
        batch_size: int = 1_000,
        # device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), # Not needed
        data_base_path: str = "data/interim/props", # Base path for generated data
        # output_dir: str = "data/processed/prop", # Not used here
    ):
        super().__init__()
        # Store necessary parameters
        self.prop = prop
        self.n = n
        self.seed = seed
        self.binding_affinity = binding_affinity
        self.batch_size = batch_size
        self.data_base_path = Path(data_base_path)

        # Construct the expected filenames based on arguments and conventions
        # from prepare_random_data.py
        self.file_base = f"prop_predictor_{self.n}_seed{self.seed}_molgen"
        if self.binding_affinity:
            self.file_base += "_binding_affinity"

        self.logits_file = self.data_base_path / f"{self.file_base}.pt"
        self.props_file = self.data_base_path / f"{self.file_base}.csv"

        # Store normalization parameters
        self.y_mean = None
        self.y_std = None

        # Save hyperparameters used for this DataModule
        self.save_hyperparameters(ignore=["data_base_path"])


    def setup(self, stage: Optional[str] = None):
        """ Loads data and prepares datasets. """
        print(f"Loading predictor training data for property: {self.prop}")
        print(f"  Loading logits from: {self.logits_file}")
        print(f"  Loading properties from: {self.props_file}")

        if not self.logits_file.exists():
            raise FileNotFoundError(f"Logits file not found: {self.logits_file}. "
                                    "Did you run prepare_random_data.py with matching settings?")
        if not self.props_file.exists():
             raise FileNotFoundError(f"Properties file not found: {self.props_file}. "
                                     "Did you run prepare_random_data.py with matching settings?")

        # Load logits (decoder outputs)
        x = torch.load(self.logits_file, map_location='cpu') # Load to CPU initially
        # Load properties
        df = pd.read_csv(self.props_file)

        # --- Data Alignment and Cleaning ---
        # Ensure data corresponds (e.g., using SMILES index if available)
        # The current props file is indexed by SMILES
        if 'smiles' not in df.columns:
             # If props file doesn't have smiles column, assume direct correspondence (risky)
             print("Properties file does not have 'smiles' column for alignment. Assuming direct row correspondence with logits.")
             if len(df) != x.shape[0]:
                 raise ValueError(f"Mismatch between number of logits ({x.shape[0]}) and properties ({len(df)}).")
             props_series = df[self.prop]
        else:
            # If using SMILES index, need to ensure logits tensor x is ordered correctly.
            # The prepare_random_data script saves properties indexed by SMILES, but saves logits sequentially.
            # This alignment is complex without saving SMILES alongside logits.
            # Easiest Fix: Modify prepare_random_data to save props sequentially *before* setting SMILES index.
            # For now, assume prepare_random_data was modified or row order matches.
            print("Assuming sequential row order in properties CSV matches logits tensor.")
            if len(df) != x.shape[0]:
                 raise ValueError(f"Mismatch between number of logits ({x.shape[0]}) and properties ({len(df)}).")
            props_series = df[self.prop]


        # Handle potential NaNs from property calculation failures
        valid_mask = ~props_series.isna()
        x = x[valid_mask]
        props_values = props_series[valid_mask].values
        print(f"Removed {len(valid_mask) - valid_mask.sum()} rows with NaN properties.")

        # Special handling for binding affinity (clamp positive values)
        if self.binding_affinity:
            print(f"Clamping positive binding affinity values for {self.prop}")
            props_values = np.minimum(props_values, 0)

        # --- Normalize Target Property ---
        self.y_mean = np.mean(props_values)
        self.y_std = np.std(props_values)
        # Avoid division by zero if standard deviation is very small
        if self.y_std < 1e-8: self.y_std = 1.0

        print(f"Property '{self.prop}': Mean={self.y_mean:.4f}, Std={self.y_std:.4f}")
        y = (props_values - self.y_mean) / self.y_std
        y_tensor = torch.tensor(y).float()
        print("\n--- Target (y) Tensor Statistics ---")
        if torch.isnan(y_tensor).any():
            print(f"!!! WARNING: Found {torch.isnan(y_tensor).sum().item()} NaN values in target y_tensor !!!")
        if torch.isinf(y_tensor).any():
            print(f"!!! WARNING: Found {torch.isinf(y_tensor).sum().item()} Inf values in target y_tensor !!!")
        print(f"  Min y_tensor: {torch.min(y_tensor).item()}")
        print(f"  Max y_tensor: {torch.max(y_tensor).item()}")
        print(f"  Mean y_tensor: {torch.mean(y_tensor).item()}")
        print(f"  Std y_tensor: {torch.std(y_tensor).item()}")
        # --- Create Datasets ---
        self.dataset = TensorDataset(x, y_tensor)
        # Use fixed generator seed for reproducibility
        generator = torch.Generator().manual_seed(self.seed)
        self.train_dataset, self.val_dataset = random_split(
            self.dataset, [0.9, 0.1], generator=generator # Use 90/10 split
        )
        print(f"Setup complete. Train size: {len(self.train_dataset)}, Val size: {len(self.val_dataset)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,          # ← add this
            pin_memory=True,        # ← if using GPU and CPU tensors
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=0,          # ← and here
            pin_memory=True,
        )


def parse_args():
    parser = LightningArgumentParser()

    # --- Training Arguments ---
    parser.add_argument( "-e", "--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument( "-lb", "--load-best", action="store_true", help="Load best checkpoint after training for validation.")
    parser.add_argument( "-tlr", "--tune-learning-rate", action="store_true", help="Run learning rate finder before training.")
    parser.add_argument("--entity", type=str, default="lakhs", help="WandB entity name.")
    # --- Model Arguments (Pass predictor_input_dim explicitly) ---
    # Calculate default based on constants/analysis
    default_predictor_input_dim = DEFAULT_MOLGEN_OUTPUT_SEQ_LEN * DEFAULT_MOLGEN_VOCAB_SIZE
    parser.add_lightning_class_args(Model, "model")
    parser.set_defaults({"model.predictor_input_dim": default_predictor_input_dim})
    parser.set_defaults({"model.hidden_sizes": [1024, 1024]})


    # --- Data Arguments ---
    parser.add_lightning_class_args(PropDataset, "data")
    # Override default for prop if needed via command line
    parser.set_defaults({"data.prop": "plogp"})


    args = parser.parse_args()

    # Remove args that are handled internally or not needed by Lightning classes anymore
    # del args.model.vae # Already removed
    # del args.data.vae, args.data.dm # Already removed

    return args


def main():
    args = parse_args()

    print("Initializing DataModule...")
    dm_prop = PropDataset(**args.data)
    dm_prop.setup()
    print("DataModule setup complete.")


    # Seed everything for reproducibility after DataModule setup (which uses seed for split)
    L.seed_everything(dm_prop.seed)

    # Instantiate Model, passing the required predictor input dimension
    print("Initializing Model...")
    model = Model(**args.model)
    print("Model initialized.")


    # --- Trainer Setup ---
    # Define checkpoint directory based on property being trained
    checkpoint_dir = Path("custom/checkpoints/prop_predictor_molgen") / dm_prop.prop
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints will be saved in: {checkpoint_dir}")

    # WandB Logger setup
    run_name = f"{dm_prop.prop}-lr{args.model.learning_rate}-bs{dm_prop.batch_size}-molgen"
    wandb_logger = WandbLogger(project="soc_prop_molgen", entity=args.entity, name=run_name) # Use your entity

    checkpoint_callback = ModelCheckpoint(
        monitor="lin_reg_r",
        mode="max",
        save_top_k=1,
        save_last=True,
        filename="best-{epoch}-{lin_reg_r:.4f}",  # ← no literal "epoch=" or "lin_reg_r="
        dirpath=checkpoint_dir,
    )

    trainer = L.Trainer(
    max_epochs=args.epochs,
    logger=wandb_logger,
    callbacks=[
        LearningRateMonitor(logging_interval="epoch"),
        checkpoint_callback,   # <-- use this one
    ],
    deterministic=True,
    gradient_clip_val=1.0,
    gradient_clip_algorithm="norm",
)

    # --- Optional Learning Rate Finder ---
    if args.tune_learning_rate:
        print("Running learning rate finder...")
        tuner = Tuner(trainer)
        # Run learning rate finder
        lr_finder = tuner.lr_find(model, datamodule=dm_prop)
        # Plot results
        fig = lr_finder.plot(suggest=True)
        fig.show()
        # Update hparams and model lr
        new_lr = lr_finder.suggestion()
        model.hparams.learning_rate = new_lr
        model.learning_rate = new_lr # Make sure instance variable is updated too
        print(f"Suggested LR: {new_lr}. Updated model learning rate.")
        # Stop execution after LR finding if you just want the suggestion
        # return

    # --- Training ---
    print("Starting training...")
    trainer.fit(model, datamodule=dm_prop)
    print("Training finished.")

    # --- Validation (Optional: Load Best) ---
    best_ckpt_path = checkpoint_callback.best_model_path
    print(f"Best checkpoint path: {best_ckpt_path}")

    if args.load_best and best_ckpt_path:
        print("Loading best model for final validation…")
        model = Model.load_from_checkpoint(
            best_ckpt_path,
            predictor_input_dim=args.model.predictor_input_dim,
            optimizer=args.model.optimizer,
            learning_rate=args.model.learning_rate,
            hidden_sizes=args.model.hidden_sizes,
        )
    trainer.validate(model, dm_prop.val_dataloader())
    print("Validation finished.")

    # --- Save Final Predictor Weights ---
    # Save only the predictor state_dict (cls) from the final model state
    final_predictor_weights_path = checkpoint_dir / "predictor_final_weights.pt"
    torch.save(model.cls.state_dict(), final_predictor_weights_path)
    print(f"Saved final predictor weights to {final_predictor_weights_path}")

    # Optionally save weights from best checkpoint too
    if args.load_best and best_ckpt_path:
         best_predictor_weights_path = checkpoint_dir / "predictor_best_weights.pt"
         # Need to load the best model again to save its specific predictor weights
         best_model = Model.load_from_checkpoint(
             best_ckpt_path,
             predictor_input_dim=args.model.predictor_input_dim,
             optimizer=args.model.optimizer, learning_rate=args.model.learning_rate,
             hidden_sizes=args.model.hidden_sizes)
         torch.save(best_model.cls.state_dict(), best_predictor_weights_path)
         print(f"Saved best predictor weights to {best_predictor_weights_path}")


if __name__ == "__main__":
    main()