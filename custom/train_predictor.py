import os
import pandas as pd
import numpy as np
# from pandarallel import pandarallel # Not strictly needed for this script's core logic

import torch
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader, TensorDataset
from lightning import LightningModule, LightningDataModule
import lightning as L
from lightning.pytorch.cli import LightningArgumentParser
# from lightning.pytorch.tuner.tuning import Tuner # Keep if lr_find is used
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)
# from tqdm import tqdm # Not strictly needed

from pathlib import Path

# from dict_hash import sha256 # Not used in the core logic shown

from scipy.stats import linregress #, pearsonr, spearmanr # Keep linregress

from cd2root import cd2root

cd2root()
from ChemFlow.src.predictor import Predictor

class Model(LightningModule):
    def __init__(
        self,
        latent_input_dim: int = 12030, # Changed from vae; 401*30=12030
        optimizer: str = "sgd",
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.latent_input_dim = latent_input_dim

        self.cls = Predictor(self.latent_input_dim)
        self.loss_fn = nn.MSELoss()

        self.val_y_hat = []
        self.val_y = []

    def forward(self, x: Tensor):
        # x is expected to be (batch_size, 401, 30) from MolTransformer latents
        x_flat = x.view(x.size(0), -1) # Flatten to (batch_size, 401*30)
        return self.cls(x_flat)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fn(self(x).squeeze(), y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, batch_size=x.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        loss = self.loss_fn(y_hat, y)
        self.val_y_hat.append(y_hat)
        self.val_y.append(y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, batch_size=x.size(0))
        return loss

    def on_validation_epoch_start(self) -> None:
        self.val_y_hat.clear()
        self.val_y.clear()

    def on_validation_epoch_end(self) -> None:
        if not self.val_y_hat or not self.val_y:
            self.log("lin_reg_r", 0.0, prog_bar=True)
            return

        val_y_hat = torch.cat(self.val_y_hat).flatten().detach().cpu().numpy()
        val_y = torch.cat(self.val_y).flatten().detach().cpu().numpy()
        
        r = 0.0
        if len(val_y_hat) > 1 and len(val_y) > 1 and np.std(val_y_hat) > 1e-6 and np.std(val_y) > 1e-6:
             # linregress needs at least 2 points and non-zero variance
            try:
                r = linregress(val_y_hat, val_y).rvalue
            except ValueError: # Handles cases like all identical values
                r = 0.0
        self.log("lin_reg_r", r, prog_bar=True)


    def configure_optimizers(self):
        if self.optimizer == "adam":
            optimizer = optim.AdamW(self.cls.parameters(), lr=self.learning_rate)
        elif self.optimizer == "sgd":
            optimizer = optim.SGD(
                self.cls.parameters(), lr=self.learning_rate, weight_decay=1e-2
            )
        else:
            raise ValueError(f"Optimizer {self.optimizer} not supported")
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[10, 15], gamma=0.1 # Consider making milestones configurable
        )
        return [optimizer], [scheduler]


class PropDataset(LightningDataModule):
    def __init__(
        self,
        prop: str = "plogp",
        n: int = 110_000,
        batch_size: int = 1_000,
        seed: int = 42,
        binding_affinity: bool = False,
        train_val_split_ratio: float = 10/11,
    ):
        super().__init__()
        self.prop = prop
        self.n = n
        self.batch_size = batch_size
        self.seed = seed
        self.binding_affinity = binding_affinity
        self.train_val_split_ratio = train_val_split_ratio

        self.save_hyperparameters()


    def setup(self, stage=None):
        
        latent_file_path = "custom/assets/molgen.pt"
        csv_file_path = "ChemFlow/data/interim/props/prop_predictor_110000_seed42_vae_pde.csv"

        x = torch.load(latent_file_path,weights_only=False) # MolTransformer latents (N, 401, 30)
        df = pd.read_csv(csv_file_path)

        if self.binding_affinity:
            df[self.prop] = df[self.prop].apply(lambda e: min(e, 0))

        self.y_mean = df[self.prop].mean()
        self.y_std = df[self.prop].std()
        
        # Handle cases where std might be zero (e.g., all y values are the same)
        if self.y_std < 1e-6: # Check for very small std dev
            y_normalized = (df[self.prop].values - self.y_mean)
        else:
            y_normalized = (df[self.prop].values - self.y_mean) / self.y_std
        
        self.dataset = TensorDataset(x, torch.tensor(y_normalized).float())
        
        num_samples = len(self.dataset)
        num_train = int(num_samples * self.train_val_split_ratio)
        num_val = num_samples - num_train

        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            self.dataset,
            [num_train, num_val],
            generator=torch.Generator().manual_seed(self.seed),
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4, pin_memory=True)




def main():

    for prop in ['plogp', 'qed', 'sa', 'jnk3', 'drd2', 'gsk3b', 'uplogp'] :
        dm_prop = PropDataset(prop=prop) # Pass data args
        dm_prop.setup()

        L.seed_everything(dm_prop.seed)

        model = Model() # Pass model args

        model_type_tag = "moltransformer"
        name = f"{dm_prop.prop}-{model_type_tag}"
        project_name = f"soc_prop_{model_type_tag}"
        checkpoint_dir_base = f"custom/checkpoints/prop_predictor_{model_type_tag}"

        trainer = L.Trainer(
            max_epochs=20,
            logger=WandbLogger(project=project_name, entity="lakhs", name=name), # entity is example
            callbacks=[
                LearningRateMonitor(logging_interval="step"),
                ModelCheckpoint(
                    monitor="lin_reg_r",
                    mode="max",
                    save_top_k=1,
                    save_last=True,
                    save_weights_only=True,
                    save_on_train_epoch_end=False, # Validate before saving checkpoint
                    dirpath=f"{checkpoint_dir_base}/{dm_prop.prop}",
                ),
            ],
        )

        #from lightning.pytorch.tuner.tuning import Tuner 
        #tuner = Tuner(trainer)
        #tuner.lr_find(model, datamodule=dm_prop)

        trainer.fit(model, datamodule=dm_prop)

        if trainer.checkpoint_callback.best_model_path:
            model = Model.load_from_checkpoint(
                trainer.checkpoint_callback.best_model_path)
        trainer.validate(model, dm_prop.val_dataloader())

        output_dir = Path(checkpoint_dir_base) / dm_prop.prop
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.cls.state_dict(), output_dir / "checkpoint.pt")


if __name__ == "__main__":
    main()