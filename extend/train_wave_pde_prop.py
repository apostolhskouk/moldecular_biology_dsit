import torch
import lightning as L
from lightning.pytorch import LightningDataModule
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.cli import LightningArgumentParser
from torch.utils.data import DataLoader, TensorDataset, random_split
from pathlib import Path
import os # For path checks

# User's existing modules - ASSUME THESE ARE CORRECTLY IN PYTHONPATH
from ChemFlow.src.predictor import Predictor # Or your updated path
from extend.wrapper import MolGenTransformerWrapper # The wrapper we created

from ChemFlow.src.pinn.pde import WavePDEModel # Or your updated path
from ChemFlow.src.utils.scores import MINIMIZE_PROPS, PROTEIN_FILES # Or your updated path
from torch import nn
from torch import Tensor
from abc import ABC


class Generator(nn.Module, ABC):
    latent_size: int
    reverse_size: int

    def forward(self, z: Tensor) -> Tensor: ...
    
class PropGenerator(Generator): # Generator is your nn.Module, ABC
    def __init__(self, model_wrapper, predictor_model): # model_wrapper is VAE or MolGenTransformerWrapper
        super().__init__()
        self.model_wrapper = model_wrapper
        self.predictor_model = predictor_model
        self.latent_size = self.model_wrapper.latent_dim # Used by WavePDEModel
        self.reverse_size = 1 # Output of predictor is scalar

    def forward(self, z: Tensor) -> Tensor:
        return self.predictor_model(self.model_wrapper.decode(z).exp())
    
class DataModule(LightningDataModule):
    # wrapper_latent_dim will be passed to a setup method or directly during instantiation if not using CLI for it
    def __init__(self, n: int = 100_000, batch_size: int = 100): # Removed wrapper_latent_dim from here
        super().__init__()
        self.save_hyperparameters("n", "batch_size") # These are the direct CLI hyperparams

        self.n = n
        self.batch_size = batch_size
        
        # These will be set by a setup method or by passing wrapper_latent_dim to __init__ in main
        self.wrapper_latent_dim = None 
        self.dataset = None
        self.train_data = None
        self.test_data = None

    def setup_data(self, wrapper_latent_dim: int, seed: int = 42): # Added seed here
        self.wrapper_latent_dim = wrapper_latent_dim
        self.dataset = TensorDataset(
            torch.randn(self.n, self.wrapper_latent_dim),
        )
        self.train_data, self.test_data = random_split(
            self.dataset, [0.9, 0.1], generator=torch.Generator().manual_seed(seed)
        )

    # train_dataloader and val_dataloader remain the same, but ensure self.train_data/test_data are set
    def train_dataloader(self):
        if self.train_data is None:
            raise RuntimeError("DataModule not setup. Call setup_data() first.")
        return DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=4
        )

    def val_dataloader(self):
        if self.test_data is None:
            raise RuntimeError("DataModule not setup. Call setup_data() first.")
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=4)


def parse_args_wavepde():
    parser = LightningArgumentParser()

    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("-e", "--epochs", type=int, default=30)
    parser.add_argument("-o", "--output_dir_base", type=str, default="extend/checkpoints_transformer_pde")
    parser.add_argument("-p", "--prop", type=str, default="qed")
    parser.add_argument("--molgen_pretrained_path", type=str, default="/data/hdd1/users/akouk/moldecular_biology_dsit/MolTransformer_repo/MolTransformer/model/models/best_models/SS_model/Best_SS_GPU.pt",
                        help="Path to MolGenTransformer pretrained model.")
    parser.add_argument("--predictor_checkpoint_dir", type=str, 
                        default="/data/hdd1/users/akouk/moldecular_biology_dsit/extend/checkpoints/prop_predictor_transformer",
                        help="Base directory for property predictor checkpoints.")


    # For WavePDEModel: generator, k, minimize_jvp are set in main()
    # For DataModule: wrapper_latent_dim is set in main(), n and batch_size are parsed
    parser.add_lightning_class_args(WavePDEModel, "model") 
    parser.add_lightning_class_args(DataModule, "data") 
    
    args = parser.parse_args()
    return args


def main_train_wavepde():
    args = parse_args_wavepde()
    L.seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wrapper = MolGenTransformerWrapper(pretrained_model_path=args.molgen_pretrained_path, device=device)
    wrapper.eval()

    datamodel = DataModule(**vars(args.data))
    datamodel.setup_data(wrapper_latent_dim=wrapper.latent_dim, seed=args.seed)

    predictor_input_size = wrapper.max_len * wrapper.vocab_size
    predictor = Predictor(predictor_input_size).to(device)
    predictor_checkpoint_path = Path(args.predictor_checkpoint_dir) / args.prop / "checkpoint.pt"
    if not predictor_checkpoint_path.exists():
        raise FileNotFoundError(f"Predictor checkpoint not found at {predictor_checkpoint_path}")
    predictor.load_state_dict(torch.load(predictor_checkpoint_path, map_location=device))
    predictor.eval()
    for p_param in predictor.parameters(): p_param.requires_grad = False

    prop_generator = PropGenerator(wrapper, predictor).to(device)

    wave_pde_model = WavePDEModel(**vars(args.model)).to(device)
    wave_pde_model.setup_core_logic(
        generator=prop_generator,
        k=1,
        minimize_jvp=(args.prop in MINIMIZE_PROPS or args.prop in PROTEIN_FILES),
        n_in=prop_generator.latent_size
    )

    pde_func_name = wave_pde_model.pde_function_val
    normalize_val = wave_pde_model.normalize_val
    output_path_final = Path(args.output_dir_base) / f"{pde_func_name}pde_prop" / "zmc_transformer" / args.prop
    output_path_final.mkdir(parents=True, exist_ok=True)

    logger_name = f"{args.prop}_norm_{normalize_val}"
    trainer = L.Trainer(
        max_epochs=args.epochs,
        logger=[
            WandbLogger(
                project=f"soc_{pde_func_name}pde_prop_transformer",
                entity="lakhs",
                name=logger_name,
            )
        ],
        callbacks=[
            LearningRateMonitor(logging_interval="epoch"),
            ModelCheckpoint(
                monitor="val/loss",
                mode="min",
                save_top_k=1,
                dirpath=output_path_final,
                save_last=True,
            ),
        ],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None,
    )
    trainer.fit(wave_pde_model, datamodule=datamodel)

    best_model_path = trainer.checkpoint_callback.best_model_path
    wavepde_to_save = wave_pde_model.pde
    if best_model_path and os.path.exists(best_model_path):
        loaded_wave_pde_model = WavePDEModel(**vars(args.model)).to(device)
        loaded_wave_pde_model.setup_core_logic(
            generator=prop_generator,
            k=1,
            minimize_jvp=(args.prop in MINIMIZE_PROPS or args.prop in PROTEIN_FILES),
            n_in=prop_generator.latent_size
        )
        checkpoint_data = torch.load(best_model_path, map_location=loaded_wave_pde_model.device)
        loaded_wave_pde_model.load_state_dict(checkpoint_data['state_dict'])
        wavepde_to_save = loaded_wave_pde_model.pde

    torch.save(wavepde_to_save.state_dict(), output_path_final / "checkpoint.pt")


if __name__ == "__main__":
    # Ensure ChemFlow.src.predictor.Predictor, extend.wrapper.MolGenTransformerWrapper,
    # ChemFlow.src.pinn.PropGenerator, ChemFlow.src.pinn.pde.WavePDEModel,
    # and ChemFlow.src.utils.scores are correctly pathed and importable.
    main_train_wavepde()