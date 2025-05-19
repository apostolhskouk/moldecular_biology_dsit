import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import selfies as sf # For SmilesTokenizerForWrapper and GenerateMethods
from torch import Tensor
from rdkit import Chem # type: ignore
# Attempt to import from MolTransformer_repo
# These paths assume 'MolTransformer_repo' is in your PYTHONPATH
# or the script is run from a location where this relative import works.
try:
    from MolTransformer_repo.MolTransformer.model import BuildModel
    from MolTransformer_repo.MolTransformer.model.utils import LoadIndex
    from MolTransformer_repo import MolTransformer # Used by _get_default_ss_path
    MOLTRANSFORMER_LIB_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import MolGenTransformer library components: {e}")
    print("Using placeholder definitions. This script will not function correctly without the actual library.")
    MOLTRANSFORMER_LIB_AVAILABLE = False
    # Placeholder definitions if the library is not found (for script structure only)
    class LoadIndex:
        def __init__(self): self.sos_indx=1; self.eos_indx=2; self.pad_indx=0; self.vocab_size=50; self.char2ind={'G':1, '[H]':21, '[C]':18, '[O]':4, 'E':17}; self.ind2char={v:k for k,v in self.char2ind.items()}
    class ChemTransformerPlaceholder(nn.Module):
        def __init__(self, *args, **kwargs): super().__init__(); self.max_sequence_length=401; self.embedding_size=30; self.vocab_size=50; self.embedding=nn.Embedding(50,30); self.position_encoding=lambda x: x; self.tansformer_encoder=lambda x, **kw: torch.randn(x.shape[1], x.shape[0], 30, device=x.device); self.tansformer_decoder=lambda tgt,mem,**kw: tgt; self.outputs2vocab=nn.Linear(30,50); self.get_tgt_mask = lambda size: torch.ones(size,size)
    class BuildModel:
        def __init__(self, *args, **kwargs): self.model = ChemTransformerPlaceholder()
    class MolTransformer: __file__ = '.' # Dummy for path


# --- MolGenTransformerWrapper CLASS DEFINITION ---
class MolGenTransformerWrapper(nn.Module):
    def __init__(self, pretrained_model_path="/data/hdd1/users/akouk/moldecular_biology_dsit/MolTransformer_repo/MolTransformer/model/models/best_models/SS_model/Best_SS_GPU.pt", device='cuda'):
        super().__init__()
        if isinstance(device, str): self.device = torch.device(device)
        else: self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not MOLTRANSFORMER_LIB_AVAILABLE:
            print("MolGenTransformer library not available, wrapper cannot be properly initialized.")
            # Initialize with dummy values if library is not found so script doesn't crash immediately
            self.molgen_index = LoadIndex() # Dummy
            self.model_actual = BuildModel().model # Dummy
        else:
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

# --- SmilesTokenizerForWrapper CLASS DEFINITION ---
class SmilesTokenizerForWrapper:
    def __init__(self, wrapper_instance):
        self.molgen_index = wrapper_instance.molgen_index
        self.char2ind = self.molgen_index.char2ind
        self.sos_idx = wrapper_instance.sos_idx 
        self.pad_idx = wrapper_instance.pad_idx 
        self.eos_idx = wrapper_instance.eos_idx 
        self.ind2char = self.molgen_index.ind2char
        self.padded_length = wrapper_instance.max_len 
        self.vocab_size = wrapper_instance.vocab_size

    def _preprocess_smiles_for_selfies(self, smiles_str: str) -> str:
        """Replicates the preprocessing from GenerateMethods._process_smile."""
        try:
            mol = Chem.MolFromSmiles(smiles_str)
            if mol is None: return "" # Or raise error
            hmol = Chem.AddHs(mol)
            # Kekulization might be needed if the model was trained on Kekulized SELFIES
            # Chem.Kekulize(hmol) # Optional: test with and without if unsure
            k_smile = Chem.MolToSmiles(hmol, kekuleSmiles=True) # kekuleSmiles=True is important
            return k_smile
        except Exception:
            return "" # Fallback for RDKit errors

    def encode(self, smiles_list: list[str], device: torch.device = None) -> torch.Tensor:
        target_device = device if device is not None else torch.device("cpu")
        token_ids_batch = []
        for smiles_str_original in smiles_list:
            # --- ADDED PREPROCESSING ---
            processed_smiles_for_selfies = self._preprocess_smiles_for_selfies(smiles_str_original)
            if not processed_smiles_for_selfies: # If preprocessing failed
                selfies_str = "" # Will lead to empty tokens
            else:
                try:
                    selfies_str = sf.encoder(processed_smiles_for_selfies)
                except Exception: 
                    selfies_str = ""
            # --- END ADDED PREPROCESSING ---
            
            try: # sf.split_selfies can also fail on empty/bad string
                tokens = list(sf.split_selfies(selfies_str))
            except Exception:
                tokens = []

            token_ids = [self.char2ind.get(t, self.pad_idx) for t in tokens]
            if len(token_ids) > self.padded_length - 1:
                token_ids = token_ids[:self.padded_length - 1]
            
            final_sequence = [self.sos_idx] + token_ids
            padding_needed = self.padded_length - len(final_sequence)
            final_sequence.extend([self.pad_idx] * padding_needed)
            token_ids_batch.append(final_sequence)
        return torch.tensor(token_ids_batch, dtype=torch.long).to(target_device)

    # decode_token_ids_to_smiles remains the same
    def decode_token_ids_to_smiles(self, batch_token_ids: list[list[int]]) -> list[str]:
        # ... (as before) ...
        smiles_out = []
        for token_ids_single_content in batch_token_ids:
            selfie_chars = []
            for token_id_val in token_ids_single_content:
                char_token = self.ind2char.get(token_id_val)
                if char_token is None: selfie_chars.append(f"[UNK_{token_id_val}]"); continue
                selfie_chars.append(char_token)
            selfie_str = "".join(selfie_chars)
            decoded_smiles = f"RAW_SELFIES({selfie_str})"
            if selfie_str:
                try:
                    smiles_candidate = sf.decoder(selfie_str)
                    if smiles_candidate: decoded_smiles = smiles_candidate
                except Exception: pass
            smiles_out.append(decoded_smiles)
        return smiles_out

# --- Main Debug Function ---
def compare_memory_and_decode():
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    
    wrapper_model_path = "/data/hdd1/users/akouk/moldecular_biology_dsit/MolTransformer_repo/MolTransformer/model/models/best_models/SS_model/Best_SS_GPU.pt"
    # ... (path check) ...

    test_smiles = "N#Cc1ccc2c(c1)OCCOCCOCCOc1ccc(C#N)cc1OCCOCCOCCO2"
    print(f"Input SMILES: {test_smiles}")

    # --- Using MolGenTransformerWrapper ---
    print("\n--- Testing with MolGenTransformerWrapper ---")
    wrapper = MolGenTransformerWrapper(pretrained_model_path=wrapper_model_path, device=device_str)
    tokenizer = SmilesTokenizerForWrapper(wrapper_instance=wrapper)
    
    # Get token_ids from our tokenizer (this is input to wrapper.encode's model_actual.encoder)
    token_ids_for_wrapper_encoder = tokenizer.encode([test_smiles], device=device)
    print(f"Wrapper: token_ids_for_encoder (shape {token_ids_for_wrapper_encoder.shape}):")
    # print(token_ids_for_wrapper_encoder[0].tolist()) # Print all if needed, or a slice
    print(f"  First 30 tokens: {token_ids_for_wrapper_encoder[0, :30].tolist()}")
    print(f"  Last 30 tokens: {token_ids_for_wrapper_encoder[0, -30:].tolist()}")


    z_flat_wrapper, _, _ = wrapper.encode(token_ids_for_wrapper_encoder)
    memory_for_decoder_wrapper = z_flat_wrapper.reshape(
        1, wrapper._internal_max_seq_len, wrapper._internal_embedding_size
    ).permute(1, 0, 2).contiguous()
    print(f"Wrapper's memory_for_decoder_wrapper - Shape: {memory_for_decoder_wrapper.shape}, Mean: {memory_for_decoder_wrapper.mean().item():.4f}, Std: {memory_for_decoder_wrapper.std().item():.4f}")
    # ... (wrapper decode and print SMILES) ...
    decoded_ids_wrapper = wrapper.decode_to_token_ids_auto_regressive(z_flat_wrapper, debug_sample_idx=0) # Keep debug prints for wrapper's decoder
    reconstructed_smiles_wrapper = tokenizer.decode_token_ids_to_smiles(decoded_ids_wrapper)[0]
    print(f"Wrapper reconstructed SMILES: {reconstructed_smiles_wrapper}")


    # --- Using original GenerateMethods ---
    print("\n--- Testing with original GenerateMethods ---")
    if not MOLTRANSFORMER_LIB_AVAILABLE:
        print("Skipping GenerateMethods comparison due to missing library.")
        return
    try:
        from MolTransformer_repo.MolTransformer.generative import GenerateMethods
        # We need to intercept the token_ids that GenerateMethods creates internally
        # This requires modifying GenerateMethods or its components, or re-implementing its tokenization path here.
        # Let's re-implement GenerateMethods' tokenization path for direct comparison:

        gm_instance_for_tokenization = GenerateMethods() # Create a GM instance
        gm_index = gm_instance_for_tokenization.Index # Access its LoadIndex instance
        gm_settings_max_len = 400

        # 1. Preprocess SMILES like GenerateMethods
        mol_gm = Chem.MolFromSmiles(test_smiles)
        hmol_gm = Chem.AddHs(mol_gm)
        # Chem.Kekulize(hmol_gm) # Optional: Test if this was implicitly done or needed
        k_smile_gm = Chem.MolToSmiles(hmol_gm, kekuleSmiles=True)
        selfies_str_gm = sf.encoder(k_smile_gm)

        # 2. Tokenize SELFIES like GenerateMethods
        tokens_gm = list(sf.split_selfies(selfies_str_gm))
        token_ids_gm_content = [gm_index.char2ind.get(char, gm_index.char2ind['[nop]']) for char in tokens_gm]
        
        seq_len_gm = min(len(token_ids_gm_content), gm_settings_max_len) # gm_settings_max_len is 400
        
        # 3. Pad like GenerateMethods.selfies_2_latent_space
        # inputs_padd = torch.zeros((1, settings.max_sequence_length + 1), dtype=torch.long, device=self.device)
        # inputs_padd[0, 0] = self.Index.char2ind['G']
        # inputs_padd[0, 1:seq_len + 1] = torch.tensor(inputs[:seq_len], device=self.device)
        padded_len_gm = gm_settings_max_len + 1 # Should be 401
        token_ids_for_gm_encoder = torch.full((1, padded_len_gm), fill_value=gm_index.pad_indx, dtype=torch.long, device=device)
        token_ids_for_gm_encoder[0, 0] = gm_index.sos_indx
        token_ids_for_gm_encoder[0, 1:seq_len_gm + 1] = torch.tensor(token_ids_gm_content[:seq_len_gm], device=device)
        
        print(f"GenerateMethods-style: token_ids_for_encoder (shape {token_ids_for_gm_encoder.shape}):")
        # print(token_ids_for_gm_encoder[0].tolist())
        print(f"  First 30 tokens: {token_ids_for_gm_encoder[0, :30].tolist()}")
        print(f"  Last 30 tokens: {token_ids_for_gm_encoder[0, -30:].tolist()}")

        # Compare the two token_id sequences directly
        if torch.equal(token_ids_for_wrapper_encoder.cpu(), token_ids_for_gm_encoder.cpu()):
            print("SUCCESS: Input token ID sequences to encoder are IDENTICAL.")
        else:
            print("WARNING: Input token ID sequences to encoder DIFFER.")
            # Find first mismatch
            for k_idx in range(token_ids_for_wrapper_encoder.shape[1]):
                if token_ids_for_wrapper_encoder[0, k_idx] != token_ids_for_gm_encoder[0, k_idx]:
                    print(f"First mismatch at index {k_idx}: Wrapper has {token_ids_for_wrapper_encoder[0, k_idx].item()}, GM-style has {token_ids_for_gm_encoder[0, k_idx].item()}")
                    wrapper_char = tokenizer.ind2char.get(token_ids_for_wrapper_encoder[0, k_idx].item(), "UNK_W")
                    gm_char = gm_index.ind2char.get(token_ids_for_gm_encoder[0, k_idx].item(), "UNK_GM")
                    print(f"Corresponding chars: Wrapper '{wrapper_char}', GM-style '{gm_char}'")
                    break
        
        # Now, proceed to get memory using GenerateMethods' actual model instance
        gm_actual_model = GenerateMethods() # Fresh instance to ensure clean state
        # The selfies_2_latent_space method in GM takes a list of SELFIES strings.
        # We already have selfies_str_gm
        memory_numpy_gm = gm_actual_model.selfies_2_latent_space([selfies_str_gm]) # Pass the SELFIES string
        memory_tensor_gm_bse = torch.from_numpy(memory_numpy_gm).to(device)
        memory_for_chemtransformer_decoder_gm = memory_tensor_gm_bse.permute(1,0,2).contiguous()
        
        print(f"GenerateMethods' memory_for_chemtransformer_decoder_gm - Shape: {memory_for_chemtransformer_decoder_gm.shape}, Mean: {memory_for_chemtransformer_decoder_gm.mean().item():.4f}, Std: {memory_for_chemtransformer_decoder_gm.std().item():.4f}")

        if torch.allclose(memory_for_decoder_wrapper.cpu(), memory_for_chemtransformer_decoder_gm.cpu(), atol=1e-5):
            print("SUCCESS: Memory tensors from Wrapper and GenerateMethods are very close!")
        else:
            print("WARNING: Memory tensors from Wrapper and GenerateMethods DIFFER (still).")
            diff = torch.abs(memory_for_decoder_wrapper.cpu() - memory_for_chemtransformer_decoder_gm.cpu())
            print(f"  Max difference: {diff.max().item():.6f}, Mean difference: {diff.mean().item():.6f}")

        decoded_dict_gm = gm_actual_model.latent_space_2_strings(memory_numpy_gm) 
        reconstructed_smiles_gm = decoded_dict_gm['SMILES'][0]
        print(f"GenerateMethods reconstructed SMILES: {reconstructed_smiles_gm}")

    except ImportError:
        print("Could not import original GenerateMethods for comparison.")
    except Exception as e:
        print(f"Error during GenerateMethods test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    compare_memory_and_decode()