import os
#set only one gpu visble
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import pandas as pd
import numpy as np
from pandarallel import pandarallel

import torch
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader, TensorDataset
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

from dict_hash import sha256

from scipy.stats import linregress, pearsonr, spearmanr

from cd2root import cd2root

cd2root()

from src.vae import load_vae, VAE, MolDataModule
from src.utils.scores import *
from src.predictor import Predictor


class Model(LightningModule):
    def __init__(
        self,
        # vae: VAE = None, # VAE no longer directly needed here if predictor input size is fixed
        predictor_input_dim: int = None, # New argument
        optimizer: str = "sgd",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4, # Added weight decay
    ):
        super().__init__()
        # self.vae = vae # Not storing vae anymore
        self.predictor_input_dim = predictor_input_dim
        self.optimizer_name = optimizer # Renamed to avoid conflict
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.cls = Predictor(self.predictor_input_dim) # Use predictor_input_dim
        self.loss_fn = nn.MSELoss()

        self.val_y_hat = []
        self.val_y = []

    def forward(self, z: Tensor): # Input is now z
        return self.cls(z)

    def training_step(self, batch, batch_idx):
        z, y = batch # Batch contains z and y
        loss = self.loss_fn(self(z).squeeze(), y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        z, y = batch # Batch contains z and y
        y_hat = self(z).squeeze()
        loss = self.loss_fn(y_hat, y)
        self.val_y_hat.append(y_hat)
        self.val_y.append(y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        self.val_y_hat.clear()
        self.val_y.clear()

    def on_validation_epoch_end(self) -> None:
        val_y_hat = torch.cat(self.val_y_hat).flatten().detach().cpu().numpy()
        val_y = torch.cat(self.val_y).flatten().detach().cpu().numpy()
        try:
            r = linregress(val_y_hat, val_y).rvalue
        except ValueError:
            r = 0.0
        self.log("lin_reg_r", r, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        if self.optimizer_name == "adamw": # Changed to adamw
            optimizer = optim.AdamW(self.cls.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_name == "sgd":
            optimizer = optim.SGD(
                self.cls.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay, momentum=0.9
            )
        else:
            raise ValueError(f"Optimizer {self.optimizer_name} not supported")

        # Using a more common scheduler like ReduceLROnPlateau or CosineAnnealingLR
        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15], gamma=0.1)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=3, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "lin_reg_r"}}


class PropDataset(LightningDataModule):
    def __init__(
        self,
        # vae: VAE = None, # No longer needed
        # dm: MolDataModule = None, # No longer needed
        prop: str = "plogp",
        n: int = 110_000,
        batch_size: int = 1000, # Consider reducing if memory is an issue with z
        # device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), # Not used here
        # output_dir: str = "data/processed/prop", # Not used here
        seed: int = 42,
        binding_affinity: bool = False,
        latent_dim: int = 1024 # Needed to construct correct file name
    ):
        super().__init__()
        # self.vae = vae
        # self.dm = dm
        self.prop = prop
        self.n = n
        self.batch_size = batch_size
        # self.device = device
        # self.output_dir = Path(output_dir)
        self.seed = seed
        self.binding_affinity = binding_affinity
        self.latent_dim = latent_dim # Store for file naming

        self.save_hyperparameters(ignore=["vae", "dm", "device", "prop_fn", "output_dir"])

    def setup(self, stage=None):
        # Adjusted file name to match the new output from prepare_random_data.py
        file_name_base = f"data/interim/props/prop_predictor_latent_z_{self.n}_seed{self.seed}"
        if self.binding_affinity:
            file_name_base += "_binding_affinity"

        z_vectors = torch.load(f"{file_name_base}.pt")
        df_props = pd.read_csv(f"{file_name_base}.csv")

        if self.binding_affinity:
            df_props[self.prop] = df_props[self.prop].apply(lambda e: min(e, 0) if pd.notna(e) else np.nan)
        
        # Handle potential NaNs from property calculation if molecules were invalid
        valid_indices = df_props[self.prop].notna()
        z_vectors = z_vectors[valid_indices]
        df_props = df_props[valid_indices].reset_index(drop=True)


        self.y_mean = df_props[self.prop].mean()
        self.y_std = df_props[self.prop].std()
        if self.y_std == 0: self.y_std = 1.0 # Avoid division by zero if all values are same

        print(f"Property: {self.prop}, Original Mean: {self.y_mean:.4f}, Std: {self.y_std:.4f} (from {len(df_props)} valid samples)")
        y_normalized = (df_props[self.prop].values - self.y_mean) / self.y_std
        
        self.dataset = TensorDataset(z_vectors, torch.tensor(y_normalized).float())
        
        # Ensure there's enough data for split after NaN removal
        num_val_samples = max(1, int(len(self.dataset) * (1/11))) # Ensure at least 1 val sample
        num_train_samples = len(self.dataset) - num_val_samples

        if num_train_samples <=0 or num_val_samples <=0:
            raise ValueError(f"Not enough valid data for property {self.prop} to split into train/val. Train: {num_train_samples}, Val: {num_val_samples}")

        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            self.dataset,
            [num_train_samples, num_val_samples],
            generator=torch.Generator().manual_seed(self.seed),
        )
        print(f"Train samples: {len(self.train_dataset)}, Validation samples: {len(self.val_dataset)}")


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)



def parse_args():
    parser = LightningArgumentParser()
    parser.add_argument("-e", "--epochs", type=int, default=50) # Increased epochs
    parser.add_argument("-lb", "--load-best", action="store_true")
    parser.add_argument("-tlr", "--tune-learning-rate", action="store_true")
    parser.add_argument("--wandb-entity", type=str, default=None)
    
    # Add predictor_input_dim to model args, it will be set programmatically
    parser.add_lightning_class_args(Model, "model")
    parser.add_lightning_class_args(PropDataset, "data")
    
    args = parser.parse_args()
    # del args.model.vae # VAE not part of Model.__init__ anymore
    return args


def main():
    args = parse_args()

    # Load VAE just to get its latent_dim for the Model and PropDataset
    _, vae_for_dim = load_vae(
        file_path="data/processed/zmc.smi", # Or args.data.smiles_file if you add it
        model_path="checkpoints/vae/zmc/checkpoint.pt", # Or args.data.vae_path
    )
    predictor_input_dim = vae_for_dim.latent_dim
    # Pass latent_dim to PropDataset if it's needed for filename construction
    args.model.predictor_input_dim = predictor_input_dim

    data_kwargs = vars(args.data)

    # overwrite whatever the CLI gave (or the default) with the real VAE dimension:
    data_kwargs["latent_dim"] = predictor_input_dim

    # now there's only one latent_dim in the dict:
    dm_prop = PropDataset(**data_kwargs)
    # dm_prop.prepare_data() # No prepare_data step in this version
    dm_prop.setup()
    
    L.seed_everything(dm_prop.seed)

    model_params_from_cli = vars(args.model)
    model = Model(**vars(args.model))

    name = f"{dm_prop.prop}_latent_z_lr{args.model.learning_rate}_wd{args.model.weight_decay}"
    
    logger = None
    if args.wandb_entity:
        logger = WandbLogger(project="soc_prop_latent_z", entity=args.wandb_entity, name=name)

    trainer = L.Trainer(
        max_epochs=args.epochs,
        logger=logger,
        callbacks=[
            LearningRateMonitor(logging_interval="epoch"), # Changed to epoch
            ModelCheckpoint(
                monitor="lin_reg_r", # Monitor lin_reg_r
                mode="max",
                save_top_k=1,
                save_last=True,
                save_weights_only=True,
                dirpath=f"checkpoints/prop_predictor_latent_z/{dm_prop.prop}", # New path
                filename='best-{epoch}-{lin_reg_r:.3f}'
            ),
        ],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1
    )

    if args.tune_learning_rate:
        tuner = Tuner(trainer)
        lr_finder_result = tuner.lr_find(model, dm_prop)
        print(lr_finder_result)
        args.model.learning_rate = lr_finder_result.suggestion()
        model.learning_rate = lr_finder_result.suggestion() # Update model's lr

    trainer.fit(model, datamodule=dm_prop)

    best_model_path = trainer.checkpoint_callback.best_model_path
    if args.load_best and best_model_path:
        print(f"Loading best model from: {best_model_path}")
        # When loading, ensure predictor_input_dim is passed correctly
        model = Model.load_from_checkpoint(
            best_model_path,
            predictor_input_dim=predictor_input_dim,
            # Strict=False might be needed if hparams saved are slightly different
        )
    trainer.validate(model, dm_prop.val_dataloader())

    output_dir = Path("checkpoints/prop_predictor_latent_z") / dm_prop.prop # New path
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.cls.state_dict(), output_dir / "checkpoint.pt")
    print(f"Saved final model to {output_dir / 'checkpoint.pt'}")


if __name__ == "__main__":
    main()