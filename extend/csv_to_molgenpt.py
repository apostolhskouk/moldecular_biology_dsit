import torch
import torch.nn.functional as F
from pathlib import Path
import pandas as pd
import selfies as sf # For MolDataModule tokenization
import numpy as np
import os # For _get_default_ss_path
import torch.nn as nn
from MolTransformer_repo.MolTransformer.model import BuildModel
from MolTransformer_repo.MolTransformer.model.utils import LoadIndex
from torch import Tensor
from tqdm import tqdm

class MolGenTransformerWrapper(nn.Module):
    def __init__(self, pretrained_model_path="/data/hdd1/users/akouk/moldecular_biology_dsit/MolTransformer_repo/MolTransformer/model/models/best_models/SS_model/Best_SS_GPU.pt", device='cuda'):
        super().__init__()

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.molgen_index = LoadIndex()
        self.sos_idx = self.molgen_index.sos_indx
        self.eos_idx = self.molgen_index.eos_indx
        self.pad_idx = self.molgen_index.pad_indx
        
        actual_pretrained_path = pretrained_model_path

        builder = BuildModel(device=self.device, model_mode='SS', 
                             gpu_mode=torch.cuda.is_available() and self.device.type == 'cuda',
                             pretrain_model_file=actual_pretrained_path)
        
        self.model_ddp_wrapped = builder.model.to(self.device)

        if isinstance(self.model_ddp_wrapped, nn.parallel.DistributedDataParallel):
            self.model_actual = self.model_ddp_wrapped.module
        else:
            self.model_actual = self.model_ddp_wrapped
        
        self.model_actual.eval()

        self._internal_max_seq_len = self.model_actual.max_sequence_length 
        self._internal_embedding_size = self.model_actual.embedding_size
        self._internal_vocab_size = self.model_actual.vocab_size

        self.max_len = self._internal_max_seq_len 
        self.vocab_size = self._internal_vocab_size
        self.latent_dim = self._internal_max_seq_len * self._internal_embedding_size


    def encode(self, token_ids: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch_size = token_ids.shape[0]
        with torch.no_grad():
            memory = self.model_actual.encoder(token_ids)
        z = memory.permute(1, 0, 2).reshape(batch_size, self.latent_dim)
        return z, z, torch.zeros_like(z)

    def decode(self, z: Tensor) -> Tensor:
        batch_size = z.shape[0]
        memory_reshaped = z.reshape(batch_size, self._internal_max_seq_len, self._internal_embedding_size)
        logits = self.model_actual.outputs2vocab(memory_reshaped)
        log_probs = F.log_softmax(logits, dim=2)
        return log_probs.reshape(batch_size, -1)


class ArgsProcessExistingCsv:
    # User MUST configure these paths
    existing_csv_path: str = "/data/hdd1/users/akouk/moldecular_biology_dsit/ChemFlow/data/interim/props/prop_predictor_110000_seed42_vae_pde.csv" # INPUT CSV
    pretrained_molgen_path: str = "/data/hdd1/users/akouk/moldecular_biology_dsit/MolTransformer_repo/MolTransformer/model/models/best_models/SS_model/Best_SS_GPU.pt"
    output_pt_file: str = "/data/hdd1/users/akouk/moldecular_biology_dsit/extend/assets/encoded_smiles_molgen.pt"
    
    smiles_column_name: str = "smiles" # Column name in CSV containing SMILES strings
    batch_size: int = 32
    seed: int = 42


# Simplified DataModule-like functionality for tokenizing SMILES for the wrapper
class SmilesTokenizerForWrapper:
    def __init__(self, wrapper_instance):
        self.molgen_index = wrapper_instance.molgen_index
        self.char2ind = self.molgen_index.char2ind
        self.sos_idx = self.molgen_index.sos_indx
        self.pad_idx = self.molgen_index.pad_indx
        # max_len for tokenized sequence including SOS and padding
        self.padded_length = wrapper_instance.max_len # This is molgen_settings.max_sequence_length + 1

    def tokenize_smiles_batch(self, smiles_list: list[str], device: torch.device) -> torch.Tensor:
        token_ids_batch = []
        for smiles_str in smiles_list:
            try:
                selfies_str = sf.encoder(smiles_str)
                tokens = list(sf.split_selfies(selfies_str))
            except Exception: # Handle cases where SMILES to SELFIES conversion fails
                tokens = [] 

            token_ids = [self.char2ind.get(t, self.pad_idx) for t in tokens] # Use pad_idx for unknown tokens

            # Prepare input for ChemTransformer.encoder: [SOS] + tokens + [PAD]
            # Truncate if longer than self.padded_length - 1 (to make space for SOS)
            if len(token_ids) > self.padded_length - 1:
                token_ids = token_ids[:self.padded_length - 1]
            
            # Final sequence: SOS + token_ids + padding
            final_sequence = [self.sos_idx] + token_ids
            padding_needed = self.padded_length - len(final_sequence)
            final_sequence.extend([self.pad_idx] * padding_needed)
            
            token_ids_batch.append(final_sequence)
        
        return torch.tensor(token_ids_batch, dtype=torch.long).to(device)


def main_process_existing_csv():
    args = ArgsProcessExistingCsv()

    if not args.pretrained_molgen_path or not os.path.exists(args.pretrained_molgen_path):
        # print(f"Error: Pretrained MolGenTransformer model path not found: {args.pretrained_molgen_path}")
        return
    if not args.existing_csv_path or not os.path.exists(args.existing_csv_path):
        # print(f"Error: Existing CSV path not found: {args.existing_csv_path}")
        return

    output_pt_path = Path(args.output_pt_file)
    output_pt_path.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    wrapper = MolGenTransformerWrapper(pretrained_model_path=args.pretrained_molgen_path, device=device)
    smiles_tokenizer = SmilesTokenizerForWrapper(wrapper)

    df = pd.read_csv(args.existing_csv_path)
    all_smiles = df[args.smiles_column_name].tolist()

    all_x0_probs_list = []
    num_molecules = len(all_smiles)
    num_batches = (num_molecules + args.batch_size - 1) // args.batch_size

    for i in tqdm(range(num_batches)):
        start_idx = i * args.batch_size
        end_idx = min((i + 1) * args.batch_size, num_molecules)
        smiles_batch = all_smiles[start_idx:end_idx]
        
        if not smiles_batch: continue

        # Tokenize SMILES for the encoder
        token_ids_for_encoder = smiles_tokenizer.tokenize_smiles_batch(smiles_batch, device)
        
        with torch.no_grad():
            # 1. Encode SMILES to latent z
            z_batch, _, _ = wrapper.encode(token_ids_for_encoder)
            # 2. Decode z to log_probs (this is the representation we want to save)
            x_log_probs_flat_batch = wrapper.decode(z_batch)
            # 3. Convert to probabilities
            x_probs_flat_batch = x_log_probs_flat_batch.exp()
        
        all_x0_probs_list.append(x_probs_flat_batch.cpu())

    final_x0_probs_tensor = torch.cat(all_x0_probs_list, dim=0)
    torch.save(final_x0_probs_tensor, output_pt_path)


if __name__ == "__main__":
    main_process_existing_csv()