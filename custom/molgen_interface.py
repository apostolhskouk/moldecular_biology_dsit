# ChemFlow-main/src/molgen_interface.py

import torch
import numpy as np
from typing import List, Dict, Optional
import warnings
import os
import selfies as sf # Make sure selfies is imported here for encode method

# Attempt to import MolGen components
try:
    # Adjust these import paths to be correct relative to where ChemFlow's src/ is
    # Assuming MolTransformer_repo is a sibling directory or MolTransformer is installed
    from MolTransformer_repo.MolTransformer.generative.generative_method import GenerateMethods
    from MolTransformer_repo.MolTransformer.model.model_architecture import ChemTransformer
    from MolTransformer_repo.MolTransformer.model.model_architecture import settings as molgen_settings
    from MolTransformer_repo.MolTransformer.model.utils.general_utils import LoadIndex # For Index.char2ind etc.
except ImportError as e:
    warnings.warn(f"Could not import MolTransformer components: {e}. Ensure MolTransformer_repo is accessible or installed.", ImportWarning)
    # Define dummy classes if not found to allow type hinting and basic script structure
    class GenerateMethods: pass
    class ChemTransformer: pass
    class molgen_settings:
        max_sequence_length = 401
        embedding_size = 30
    class LoadIndex:
        def __init__(self):
            self.pad_indx = 0
            self.sos_indx = 1 # Placeholder
            self.vocab_size = 121 # Placeholder
            self.char2ind = {'[nop]': 0, 'G': 1} # Minimal placeholder

class MolGenInterface:
    """
    Interface for MolGen-Transformer, using teacher_forcing for differentiable logits.
    """
    def __init__(self, gm_config: Optional[Dict] = None, **kwargs):
        if gm_config is None: gm_config = {}
        print("Initializing MolTransformer.GenerateMethods for Interface...")
        self.GM = GenerateMethods(**gm_config, **kwargs) # GenerateMethods loads the model
        print("GenerateMethods initialized.")

        self.model: ChemTransformer = self.GM.model # Access the underlying ChemTransformer model
        if self.GM.gpu_mode and hasattr(self.model, 'module'): # Handle DataParallel
            self.model_for_direct_calls = self.model.module
            print("Using model.module for direct calls (likely DataParallel).")
        else:
            self.model_for_direct_calls = self.model
            print("Using model directly for calls.")
        self.model_for_direct_calls.eval() # Ensure model is in eval mode

        self.device = self.GM.device

        # --- Define Dimensions using attributes from the loaded ChemTransformer model ---
        self.model_max_seq_len = self.model_for_direct_calls.max_sequence_length # e.g., 401+1 from settings
        self.model_embedding_size = self.model_for_direct_calls.embedding_size   # e.g., 30
        self.model_vocab_size = self.model_for_direct_calls.vocab_size         # e.g., 121
        self.model_pad_idx = self.model_for_direct_calls.pad_idx
        self.model_sos_idx = self.model_for_direct_calls.sos_idx

        # --- Dimensions for ChemFlow ---
        # Latent 'z' will be encoder memory: (B, model_max_seq_len, model_embedding_size)
        self.latent_seq_len = self.model_max_seq_len
        self.latent_embed_dim = self.model_embedding_size

        # Output logits sequence length will match model's max_sequence_length
        self.output_seq_len = self.model_max_seq_len
        self.output_vocab_size = self.model_vocab_size

        print("MolGenInterface initialized successfully.")
        print(f"Interface using device: {self.device}")
        print(f"ChemFlow latent_seq_len (from model.max_sequence_length): {self.latent_seq_len}")
        print(f"ChemFlow latent_embed_dim (from model.embedding_size): {self.latent_embed_dim}")
        print(f"ChemFlow output_seq_len: {self.output_seq_len}")
        print(f"ChemFlow output_vocab_size (from model.vocab_size): {self.output_vocab_size}")


    def encode(self, smiles_list: List[str]) -> torch.Tensor:
        """
        Encodes SMILES to latent sequence (encoder memory).

        Returns:
            torch.Tensor: Latent sequence (encoder memory),
                          shape (batch_size, self.latent_seq_len, self.latent_embed_dim).
                          Requires gradients.
        """
        if not smiles_list:
             return torch.empty((0, self.latent_seq_len, self.latent_embed_dim),
                               device=self.device, requires_grad=True)

        # 1. Convert SMILES to token indices (B, model_max_seq_len)
        #    This needs to exactly match how ChemTransformer.encoder expects its input_idx
        token_idx_list = []
        for smile in smiles_list:
            # Use GM.Index which is loaded by GenerateMethods
            selfies_str = self.GM.smile_2_selfies(smile) # GM.smile_2_selfies handles single string
            inputs_char = sf.split_selfies(selfies_str)
            inputs_idx_num = [self.GM.Index.char2ind.get(char, self.model_pad_idx) for char in inputs_char]

            # Pad/truncate to model_max_seq_len, add SOS token 'G'
            # ChemTransformer.encoder adds positional encoding to embedding(input_idx.permute(1,0,2))
            # where input_idx is (B, S). So we need to provide (B,S)
            padded_indices = torch.full((self.model_max_seq_len,), self.model_pad_idx, dtype=torch.long, device=self.device)
            padded_indices[0] = self.model_sos_idx # Start token 'G'

            seq_len_to_copy = min(len(inputs_idx_num), self.model_max_seq_len - 1)
            if seq_len_to_copy > 0 :
                padded_indices[1 : 1 + seq_len_to_copy] = torch.tensor(inputs_idx_num[:seq_len_to_copy], device=self.device)
            token_idx_list.append(padded_indices)

        if not token_idx_list:
            return torch.empty((0, self.latent_seq_len, self.latent_embed_dim),
                               device=self.device, requires_grad=True)

        input_tensor_for_encoder = torch.stack(token_idx_list) # Shape (B, model_max_seq_len)

        # 2. Encode using the ChemTransformer model's encoder
        with torch.no_grad(): # Initial encoding for ChemFlow usually doesn't need grad tracking
             # model_for_direct_calls.encoder expects (B, S) and returns (S, B, E)
             memory_sbe = self.model_for_direct_calls.encoder(input_tensor_for_encoder)

        # 3. Permute to (B, S, E) as the standard 'z' for ChemFlow
        latent_tensor_bse = memory_sbe.permute(1, 0, 2).detach().clone()
        latent_tensor_bse.requires_grad_(True)

        if latent_tensor_bse.shape[1] != self.latent_seq_len or latent_tensor_bse.shape[2] != self.latent_embed_dim:
             raise ValueError(f"Encoded latent tensor shape mismatch. Got {latent_tensor_bse.shape}, expected (B, {self.latent_seq_len}, {self.latent_embed_dim})")

        return latent_tensor_bse

    def decode_to_logits(self, z_latent_bse: torch.Tensor) -> torch.Tensor:
        """
        Decodes latent sequence (encoder memory) z to pre-softmax logits
        using the teacher_forcing method with a fixed decoder input.

        Args:
            z_latent_bse (torch.Tensor): Latent sequence (encoder memory),
                                       shape (batch_size, self.latent_seq_len, self.latent_embed_dim).

        Returns:
            torch.Tensor: Output logits tensor, flattened shape (B, self.output_seq_len * self.output_vocab_size).
        """
        if z_latent_bse.shape[0] == 0:
             return torch.empty((0, self.output_seq_len * self.output_vocab_size), device=self.device)

        z_latent_bse = z_latent_bse.to(self.device)
        batch_size = z_latent_bse.shape[0]

        # 1. Permute z_latent_bse to (SeqLen, Batch, EmbDim) for teacher_forcing's 'memory' argument
        #    Encoder output is (S, B, E), our 'z' from encode() is (B, S, E). So permute back.
        if z_latent_bse.shape[1] != self.latent_seq_len or z_latent_bse.shape[2] != self.latent_embed_dim:
            raise ValueError(f"Input z_latent_bse has unexpected shape: {z_latent_bse.shape}. "
                             f"Expected (B, {self.latent_seq_len}, {self.latent_embed_dim}).")
        memory_sbe = z_latent_bse.permute(1, 0, 2).contiguous()

        # 2. Create fixed_decoder_input_idx (B, model_max_seq_len)
        #    Filled with SOS at pos 0, then PAD.
        fixed_decoder_input_idx = torch.full(
            (batch_size, self.model_max_seq_len),
            fill_value=self.model_pad_idx,
            dtype=torch.long,
            device=self.device
        )
        fixed_decoder_input_idx[:, 0] = self.model_sos_idx

        # 3. Call teacher_forcing to get decoder hidden states
        #    teacher_forcing(memory, input_idx) -> (SeqLen_out, Batch, EmbDim_out)
        #    where SeqLen_out = model_max_seq_len, EmbDim_out = model_embedding_size
        decoder_hidden_states_sbe = self.model_for_direct_calls.teacher_forcing(memory_sbe, fixed_decoder_input_idx)

        # 4. Permute hidden states to (Batch, SeqLen, EmbDim)
        decoder_hidden_states_bse = decoder_hidden_states_sbe.permute(1, 0, 2)

        # 5. Pass through the final linear layer (outputs2vocab)
        #    outputs2vocab expects (N, embedding_size)
        logits_flat_vocab_dim = self.model_for_direct_calls.outputs2vocab(
            decoder_hidden_states_bse.reshape(-1, self.model_embedding_size)
        ) # Output: (B*SeqLen, VocabSize)

        # 6. Reshape to (Batch, SeqLen, VocabSize) - these are pre-softmax logits
        pre_softmax_logits_bsv = logits_flat_vocab_dim.reshape(
            batch_size,
            self.model_max_seq_len,
            self.model_vocab_size
        )

        # Optional: Apply log_softmax if downstream components expect log-probabilities
        # (as per ChemTransformer.TF_2_logp)
        logp_bsv = torch.nn.functional.log_softmax(pre_softmax_logits_bsv, dim=-1)

        # 7. Flatten for ChemFlow's typical predictor input (B, SeqLen * VocabSize)
        final_output_flat = logp_bsv.reshape(batch_size, -1)

        return final_output_flat

    def decode_to_smiles(self, z_latent_bse: torch.Tensor) -> List[str]:
        """
        Decodes latent sequence (encoder memory) z to SMILES using the model's
        full autoregressive decoder.

        Args:
            z_latent_bse (torch.Tensor): Latent sequence,
                                       shape (batch_size, self.latent_seq_len, self.latent_embed_dim).
        Returns:
            List[str]: List of decoded SMILES strings.
        """
        if z_latent_bse.shape[0] == 0: return []
        z_latent_bse = z_latent_bse.to(self.device)

        # Permute z_latent_bse to (SeqLen, Batch, EmbDim) for model's decoder method
        if not (len(z_latent_bse.shape) == 3 and z_latent_bse.shape[1] == self.latent_seq_len and z_latent_bse.shape[2] == self.latent_embed_dim):
             raise ValueError(f"Input z_latent_bse has unexpected shape: {z_latent_bse.shape} for autoregressive decoding.")
        memory_sbe = z_latent_bse.permute(1, 0, 2).contiguous()

        with torch.no_grad():
            list_of_index_lists = self.model_for_direct_calls.decoder(memory_sbe)

        selfies_list = self.model_for_direct_calls.index_2_selfies(list_of_index_lists)
        smiles_list = self.model_for_direct_calls.selfies_2_smile(selfies_list)
        return smiles_list