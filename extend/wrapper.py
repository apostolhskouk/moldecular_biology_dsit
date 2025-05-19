import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os # For _get_default_ss_path
from MolTransformer_repo.MolTransformer.model import BuildModel
from MolTransformer_repo.MolTransformer.model.utils import LoadIndex
#from MolTransformer_repo.MolTransformer.model import settings as molgen_settings
from MolTransformer_repo import MolTransformer
from torch import Tensor
class MolGenTransformerWrapper(nn.Module):
    def __init__(self, pretrained_model_path="/data/hdd1/users/akouk/moldecular_biology_dsit/MolTransformer_repo/MolTransformer/model/models/best_models/SS_model/Best_SS_GPU.pt", device='cuda'):
        super().__init__()
        if isinstance(device, str): self.device = torch.device(device)
        else: self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.molgen_index = LoadIndex() # Actual
        builder = BuildModel(device=self.device, model_mode='SS', 
                                gpu_mode=(self.device.type == 'cuda'),
                                pretrain_model_file=pretrained_model_path)
        self.model_ddp_wrapped = builder.model.to(self.device)
        self.model_actual = self.model_ddp_wrapped.module if isinstance(self.model_ddp_wrapped, nn.parallel.DistributedDataParallel) else self.model_ddp_wrapped
        
        self.model_actual.eval()

        self.sos_idx = self.molgen_index.sos_indx
        self.eos_idx = self.molgen_index.eos_indx
        self.pad_idx = self.molgen_index.pad_indx
        
        self._internal_max_seq_len = self.model_actual.max_sequence_length 
        self._internal_embedding_size = self.model_actual.embedding_size
        self._internal_vocab_size = self.model_actual.vocab_size
        self.max_len = self._internal_max_seq_len 
        self.vocab_size = self._internal_vocab_size
        self.latent_dim = self._internal_max_seq_len * self._internal_embedding_size

    # _get_default_ss_path can be removed if pretrained_model_path is always provided

    def encode(self, token_ids: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch_size = token_ids.shape[0]
        token_ids_on_device = token_ids.to(self.device)
        with torch.no_grad():
            memory = self.model_actual.encoder(token_ids_on_device) 
        z = memory.permute(1, 0, 2).reshape(batch_size, self.latent_dim)
        return z, z, torch.zeros_like(z)

    def decode_to_parallel_log_probs(self, z: Tensor) -> Tensor:
        batch_size = z.shape[0]
        memory_reshaped = z.reshape(batch_size, self._internal_max_seq_len, self._internal_embedding_size)
        logits = self.model_actual.outputs2vocab(memory_reshaped.to(self.device))
        log_probs = F.log_softmax(logits, dim=2)
        return log_probs.reshape(batch_size, -1)

    def decode_to_token_ids_auto_regressive(self, z: Tensor, max_output_len: int = None, debug_sample_idx: int = -1) -> list[list[int]]:
        self.model_actual.eval()
        batch_size = z.shape[0]
        z_on_device = z.to(self.device)
        memory_for_decoder = z_on_device.reshape(
            batch_size, self._internal_max_seq_len, self._internal_embedding_size
        ).permute(1, 0, 2).contiguous()

        if max_output_len is None: max_output_len = self._internal_max_seq_len -1 
        generated_sequences_ids = []

        for i in range(batch_size):
            memory_sample = memory_for_decoder[:, i:i+1, :] 
            current_sequence_ids = torch.tensor([[self.sos_idx]], dtype=torch.long, device=self.device)
            sequence_token_ids_for_sample = [] 
            if i == debug_sample_idx: print(f"\n--- Debugging Auto-Regressive Decode for Sample {i} (Wrapper) ---")

            for step in range(max_output_len):
                input_embeddings = self.model_actual.embedding(current_sequence_ids).permute(1, 0, 2)
                input_embeddings = self.model_actual.position_encoding(input_embeddings)
                tgt_mask = self.model_actual.get_tgt_mask(current_sequence_ids.size(1)).to(self.device)
                
                with torch.no_grad():
                    transformer_decoder_output = self.model_actual.tansformer_decoder(
                        tgt=input_embeddings.float(), memory=memory_sample.float(), tgt_mask=tgt_mask.float()
                    )
                last_token_output_embedding = transformer_decoder_output[-1:, :, :]
                logits_next_token = self.model_actual.outputs2vocab(last_token_output_embedding)
                
                if i == debug_sample_idx and step < 5:
                    top_k_logits, top_k_ids = torch.topk(logits_next_token.squeeze(), 5)
                    print(f"  Step {step}: Prev Token ID: {current_sequence_ids[0,-1].item()}")
                    print(f"    Top 5 Logits: {np.round(top_k_logits.cpu().detach().numpy(), 2)}")
                    print(f"    Top 5 IDs: {top_k_ids.cpu().tolist()}")
                    id_to_char_list = [self.molgen_index.ind2char.get(tkid, f'UNK({tkid})') for tkid in top_k_ids.cpu().tolist()]
                    print(f"    Top 5 Chars: {id_to_char_list}")
                
                next_token_id_tensor = torch.argmax(logits_next_token, dim=2)
                next_token_id_item = next_token_id_tensor.item()
                
                if i == debug_sample_idx and step < 5:
                     print(f"    Predicted Next ID: {next_token_id_item} ({self.molgen_index.ind2char.get(next_token_id_item, f'UNK({next_token_id_item})')})")

                if next_token_id_item == self.eos_idx: break 
                sequence_token_ids_for_sample.append(next_token_id_item)
                current_sequence_ids = torch.cat((current_sequence_ids, next_token_id_tensor.view(1,1)), dim=1)
            
            generated_sequences_ids.append(sequence_token_ids_for_sample)
            if i == debug_sample_idx: print(f"  Generated content IDs for sample {i} (Wrapper): {sequence_token_ids_for_sample}")
        return generated_sequences_ids