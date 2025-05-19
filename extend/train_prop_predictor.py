import torch
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader, TensorDataset
from lightning import LightningModule, LightningDataModule
import lightning as L
from lightning.pytorch.cli import LightningArgumentParser
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
import pandas as pd
from pathlib import Path
from scipy.stats import linregress
from ChemFlow.src.predictor import Predictor
from extend.wrapper import MolGenTransformerWrapper
import os

class PropPredictorModel(LightningModule):
    def __init__(
        self, 
        input_feature_size: int, 
        optimizer: str = "sgd", 
        learning_rate: float = 1e-3
    ):
        super().__init__()
        self.optimizer_name = optimizer 
        self.learning_rate = learning_rate
        self.save_hyperparameters()

        self.predictor_net = Predictor(input_feature_size)
        self.loss_fn = nn.MSELoss()

        self.val_y_hat = []
        self.val_y = []

    def forward(self, x: Tensor):
        return self.predictor_net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fn(self(x).squeeze(), y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        loss = self.loss_fn(y_hat, y)
        self.val_y_hat.append(y_hat)
        self.val_y.append(y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        self.val_y_hat.clear()
        self.val_y.clear()

    def on_validation_epoch_end(self) -> None:
        if not self.val_y_hat or not self.val_y: return
        val_y_hat_all = torch.cat(self.val_y_hat).flatten().detach().cpu().numpy()
        val_y_all = torch.cat(self.val_y).flatten().detach().cpu().numpy()
        
        r_value = 0.0
        if len(val_y_hat_all) > 1 and len(val_y_all) > 1:
            try:
                r_value = linregress(val_y_hat_all, val_y_all).rvalue
            except ValueError:
                 r_value = 0.0
        self.log("lin_reg_r", r_value, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        if self.optimizer_name == "adam":
            opt = optim.AdamW(self.predictor_net.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == "sgd":
            opt = optim.SGD(self.predictor_net.parameters(), lr=self.learning_rate, weight_decay=1e-2)
        else:
            raise ValueError(f"Optimizer {self.optimizer_name} not supported")
        
        scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[10, 15], gamma=0.1)
        return [opt], [scheduler]


class PropDataset(LightningDataModule):
    def __init__(
        self,
        prop_pt_file: str, 
        prop_csv_file: str, 
        prop_column_name: str = "plogp",
        batch_size: int = 1000,
        seed: int = 42,
        binding_affinity: bool = False,
    ):
        super().__init__()
        self.prop_pt_file = prop_pt_file
        self.prop_csv_file = prop_csv_file
        self.prop_column_name = prop_column_name
        self.batch_size = batch_size
        self.seed = seed
        self.binding_affinity = binding_affinity
        self.save_hyperparameters()

    def setup(self, stage=None):
        x = torch.load(self.prop_pt_file)
        df = pd.read_csv(self.prop_csv_file)

        if self.binding_affinity:
            df[self.prop_column_name] = df[self.prop_column_name].apply(lambda e: min(e, 0.0) if pd.notnull(e) else 0.0)
        
        self.y_mean = df[self.prop_column_name].mean()
        self.y_std = df[self.prop_column_name].std()
        
        y_values = df[self.prop_column_name].fillna(self.y_mean).values
        y_normalized = (y_values - self.y_mean) / (self.y_std if self.y_std != 0 else 1.0)
        
        self.dataset = TensorDataset(x, torch.tensor(y_normalized).float())
        
        n_samples = len(self.dataset)
        n_train = int(n_samples * (10 / 11))
        n_val = n_samples - n_train
        
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            self.dataset, [n_train, n_val], generator=torch.Generator().manual_seed(self.seed)
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)


def parse_args_predictor():
    parser = LightningArgumentParser()
    parser.add_argument("-e", "--epochs", type=int, default=20)
    parser.add_argument("-lb", "--load_best", action="store_true")

    parser.add_argument("--prop_pt_file", type=str, default="/data/hdd1/users/akouk/moldecular_biology_dsit/extend/assets/encoded_smiles_molgen.pt",help="Path to the .pt data file.")
    parser.add_argument("--prop_csv_file", type=str, default="/data/hdd1/users/akouk/moldecular_biology_dsit/ChemFlow/data/interim/props/prop_predictor_110000_seed42_vae_pde.csv", help="Path to the .csv data file.")
    parser.add_argument("--prop_column_name", type=str, default="plogp", help="Name of the property column in CSV.")
    parser.add_argument("--prop_batch_size", type=int, default=1024)
    parser.add_argument("--prop_seed", type=int, default=42)
    parser.add_argument("--binding_affinity", action="store_true", default=False)

    #parser.add_lightning_class_args(PropPredictorModel, "model")
    
    args = parser.parse_args()
    return args


def main_train_predictor():
    args = parse_args_predictor()
    
    temp_wrapper_config = MolGenTransformerWrapper()
    transformer_max_len = temp_wrapper_config.max_len
    transformer_vocab_size = temp_wrapper_config.vocab_size
    del temp_wrapper_config 

    dm_prop = PropDataset(
        prop_pt_file=args.prop_pt_file,
        prop_csv_file=args.prop_csv_file,
        prop_column_name=args.prop_column_name,
        batch_size=args.prop_batch_size,
        seed=args.prop_seed,
        binding_affinity=args.binding_affinity
    )
    dm_prop.setup() # Call setup explicitly if not relying on trainer to call it.

    L.seed_everything(dm_prop.seed)

    model_input_feature_size = transformer_max_len * transformer_vocab_size
    prop_model = PropPredictorModel(
        input_feature_size=model_input_feature_size,
    )

    model_name_suffix = f"{dm_prop.prop_column_name}"
    
    # Ensure checkpoint directory exists
    checkpoint_dir = Path(f"extend/checkpoints/prop_predictor_transformer/{dm_prop.prop_column_name}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    trainer = L.Trainer(
        max_epochs=args.epochs,
        logger=WandbLogger(project="soc_prop_transformer", entity="lakhs", name=model_name_suffix),
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(
                monitor="lin_reg_r", mode="max", save_top_k=1, save_last=True,
                save_weights_only=True, 
                dirpath=str(checkpoint_dir),
            ),
        ],
        # --- ADD/MODIFY THESE LINES FOR LIGHTNING DDP ---
        strategy="ddp_find_unused_parameters_true" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "auto", # Use DDP if multiple GPUs
        devices="auto", # Auto-select GPUs
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        # --- END ADD/MODIFY ---
    )

    trainer.fit(prop_model, datamodule=dm_prop)

    best_model_path_to_load = trainer.checkpoint_callback.best_model_path
    final_model_for_validation = prop_model # Default to last trained model

    if args.load_best and best_model_path_to_load and os.path.exists(best_model_path_to_load):
        final_model_for_validation = PropPredictorModel.load_from_checkpoint(
            best_model_path_to_load,
            input_feature_size=model_input_feature_size 
        )
    
    trainer.validate(final_model_for_validation, datamodule=dm_prop)

    torch.save(final_model_for_validation.predictor_net.state_dict(), checkpoint_dir / "checkpoint.pt")


if __name__ == "__main__":
    main_train_predictor()