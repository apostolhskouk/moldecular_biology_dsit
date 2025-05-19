# File: use_pretrained_seqflow.py
import os
import argparse
import random
from pathlib import Path
import contextlib
import hashlib

import numpy as np
import torch
import selfies as sf
from tqdm import tqdm
from rdkit import Chem # Needed for optional canonicalization during preprocessing
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'NFBO'))
# Import necessary components from the repository
# Adjust paths if you run this script from a different location relative to NFBO-main
from objective.guacamol.utils.mol_utils.data import SELFIESDataset
from generative_model.model.seqflow.lstm_flow import AFPrior
from torch.cuda.amp import autocast # For mixed precision

# --- Helper Functions ---

def set_seed(seed):
    """Sets random seed for reproducibility (less critical for inference but good practice)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Deterministic operations are not strictly necessary for inference
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def load_pretrained_model_and_dataobj(model_path, device, model_kwargs):
    """Loads the pretrained AFPrior model and the corresponding SELFIESDataset object."""
    # 1. Setup Data Object (for vocab and tokenization)
    # We assume the pretrained model used the 'guacamol' vocab type
    dataobj = SELFIESDataset(data_type='guacamol')
    vocab_size = dataobj.vocab_size
    print(f"Using vocabulary size: {vocab_size}")

    # Update model_kwargs with the determined vocab_size
    model_kwargs['vocab_size'] = vocab_size

    # 2. Instantiate Model
    model = AFPrior(**model_kwargs).to(device)

    # 3. Load State Dict
    try:
        print(f"Loading pretrained model state dict from: {model_path}")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading pretrained model from {model_path}: {e}")
        print("Ensure the model path is correct and the checkpoint matches the architecture.")
        return None, None

    # 4. Set model to evaluation mode and pass dataobj
    model.eval()
    model.set_dataobj(dataobj) # Important for model's internal decoding/canonicalization

    return model, dataobj

def get_cache_path(input_file_path, max_len, canonicalize, cache_dir):
    """Creates a unique cache file path based on input parameters."""
    input_path = Path(input_file_path)
    # Use a hash of the full path to avoid issues with same filenames in different dirs
    path_hash = hashlib.md5(str(input_path.resolve()).encode()).hexdigest()
    cache_filename = f"{input_path.stem}_maxlen{max_len}_canon{canonicalize}_{path_hash}.pt"
    return Path(cache_dir) / cache_filename

def smiles_to_tensor(smiles_list, dataobj, max_len, device, canonicalize=True, cache_path=None):
    """
    Converts a list of SMILES to a padded tensor of token IDs.
    Uses caching if cache_path is provided.
    """
    if cache_path and cache_path.exists():
        try:
            print(f"Loading cached processed data from: {cache_path}")
            cached_data = torch.load(cache_path)
            # Ensure tensor is on the correct device
            cached_data['tensor'] = cached_data['tensor'].to(device)
            print("Loaded from cache successfully.")
            return cached_data['tensor'], cached_data['smiles']
        except Exception as e:
            print(f"Warning: Could not load cache file {cache_path}. Re-processing. Error: {e}")

    # --- Processing ---
    selfies_list = []
    valid_smiles_indices = []
    print("Converting SMILES to SELFIES...")
    for i, smi in enumerate(tqdm(smiles_list, desc="SMILES->SELFIES")):
        if not smi: continue # Skip empty strings
        try:
            # Optional: Canonicalize SMILES first
            canon_smi = smi
            if canonicalize:
                 mol = Chem.MolFromSmiles(smi)
                 if mol:
                     canon_smi = Chem.MolToSmiles(mol)
                 else: # If canonicalization fails, skip
                     continue
            selfies_str = sf.encoder(canon_smi)
            if selfies_str:
                selfies_list.append(selfies_str)
                valid_smiles_indices.append(i) # Store original index
        except Exception:
            # Ignore conversion errors silently during bulk processing
            # You might want to add logging here if needed
            pass

    print(f"Tokenizing {len(selfies_list)} SELFIES...")
    if not selfies_list:
         print("No valid selfies generated.")
         return None, []
    tokenized_selfies = dataobj.tokenize_selfies(selfies_list)

    encoded_data = []
    final_valid_indices = [] # Stores original indices corresponding to rows in encoded_data
    skipped_unknown = 0
    skipped_len = 0
    print("Encoding and Padding Tokens...")
    for i, tokens in enumerate(tqdm(tokenized_selfies, desc="Tokenizing")):
        valid_tokens = True
        encoded = []
        for token in tokens:
            if token not in dataobj.vocab2idx:
                valid_tokens = False
                skipped_unknown += 1
                break # Skip molecule if it contains unknown tokens
            encoded.append(dataobj.vocab2idx[token])

        if valid_tokens:
            if len(encoded) <= max_len:
                padding_length = max_len - len(encoded)
                stop_token_idx = dataobj.vocab2idx.get('<stop>', 1)
                padded_encoded = encoded + [stop_token_idx] * padding_length
                encoded_data.append(padded_encoded)
                final_valid_indices.append(valid_smiles_indices[i]) # Map back to original index
            else:
                skipped_len += 1
        # else: token was already invalid

    if skipped_unknown > 0:
        print(f"Skipped {skipped_unknown} molecules containing unknown SELFIES tokens.")
    if skipped_len > 0:
        print(f"Skipped {skipped_len} molecules exceeding max_len ({max_len}).")

    if not encoded_data:
        print("No valid molecules could be encoded after filtering.")
        return None, []

    # Get the actual SMILES strings corresponding to the encoded data
    original_smiles_valid = [smiles_list[i] for i in final_valid_indices]
    encoded_tensor = torch.tensor(encoded_data, dtype=torch.long).to(device)

    # Save to cache if path provided
    if cache_path:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Saving processed data to cache: {cache_path}")
            # Move tensor to CPU before saving to avoid GPU memory in cache file
            torch.save({'tensor': encoded_tensor.cpu(), 'smiles': original_smiles_valid}, cache_path)
            print("Cache saved successfully.")
            # Move tensor back to original device for subsequent use
            encoded_tensor = encoded_tensor.to(device)
        except Exception as e:
            print(f"Warning: Could not save cache file {cache_path}. Error: {e}")

    return encoded_tensor, original_smiles_valid


# --- Main Inference Script ---

def main(args):
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    use_amp = device.type == 'cuda'
    print(f"Using Automatic Mixed Precision (AMP): {use_amp}")

    # --- Architecture Parameters (Hardcoded from seqflow.yaml for the specific checkpoint) ---
    # !!! Ensure these match the parameters used to train seqflow_1.27M_3.pt !!!
    model_kwargs = {
        # 'vocab_size': Will be set after loading dataobj
        'hidden_size': 500,
        'zsize': 40,
        'dropout_p': 0.2,
        'dropout_locations': ['prior_rnn'],
        'prior_type': 'AF',
        'num_flow_layers': 5,
        'rnn_layers': 2,
        'max_T': 288, # Max T for positional encoding in model
        'transform_function': "nlsq",
        'hiddenflow_params': {
            'nohiddenflow': False,
            'hiddenflow_layers': 2,
            'hiddenflow_units': 100,
            'hiddenflow_flow_layers': 10,
            'hiddenflow_scf_layers': True
        },
        'noise': 0.1,       # These don't affect inference if model is in eval mode
        'sim_coef': 1.0,    # These don't affect inference if model is in eval mode
        'cliping': True,    # This doesn't affect inference
        'canonical': args.canonicalize_output # Affects decode_z output processing
    }

    # --- Load Model and Data Object ---
    model, dataobj = load_pretrained_model_and_dataobj(args.model_path, device, model_kwargs)
    if model is None or dataobj is None:
        return

    # --- Perform Inference ---
    model.eval() # Ensure model is in eval mode
    with torch.no_grad(): # Ensure no gradients are calculated
        if args.mode == 'encode':
            print("--- Encoding Molecules ---")
            # Determine cache path
            cache_file_path = None
            if os.path.isfile(args.input):
                print(f"Reading SMILES from file: {args.input}")
                with open(args.input, 'r') as f:
                    input_smiles = [line.strip() for line in f if line.strip()]
                if args.cache_dir:
                    cache_file_path = get_cache_path(args.input, args.max_len_encode, args.canonicalize_input, args.cache_dir)
            else:
                print(f"Using input SMILES string: {args.input}")
                input_smiles = [args.input]
                # Caching is typically not useful for single strings

            # Preprocess SMILES (potentially using cache)
            input_tensor, valid_smiles = smiles_to_tensor(
                input_smiles, dataobj, args.max_len_encode, device,
                canonicalize=args.canonicalize_input, cache_path=cache_file_path
            )

            if input_tensor is None or input_tensor.numel() == 0:
                print("No valid molecules could be encoded.")
                return

            # Process in batches
            all_latent_vectors_flat = []
            num_mols = len(input_tensor)
            print(f"Encoding {num_mols} valid molecules in batches of {args.batch_size}...")

            for i in tqdm(range(0, num_mols, args.batch_size), desc="Encoding Batches"):
                batch_tensor = input_tensor[i:min(i + args.batch_size, num_mols)]
                if batch_tensor.numel() == 0: continue

                # The model's encode_z expects input shape (B, T)
                with autocast(enabled=use_amp): # Wrap model call
                    latent_vectors, logp, _ = model.encode_z(batch_tensor, sampling=args.sample_z)

                # Latent vectors might be FP16 from AMP, convert back before numpy
                latent_vectors_flat = latent_vectors.float().cpu().numpy().reshape(latent_vectors.shape[0], -1)
                all_latent_vectors_flat.append(latent_vectors_flat)

            if not all_latent_vectors_flat:
                print("No molecules were successfully encoded.")
                return

            final_latent_vectors = np.concatenate(all_latent_vectors_flat, axis=0)
            print(f"Encoded {len(valid_smiles)} valid molecules.")
            print(f"Final encoded latent vectors shape: {final_latent_vectors.shape}")


            if args.output_file:
                 output_path = Path(args.output_file)
                 output_path.parent.mkdir(parents=True, exist_ok=True)
                 np.save(output_path, final_latent_vectors)
                 print(f"Saved {len(final_latent_vectors)} latent vectors to {output_path}")
                 # Save corresponding valid SMILES
                 smiles_output_path = output_path.with_suffix('.smi')
                 try:
                     with open(smiles_output_path, 'w') as f_smi:
                         for smi in valid_smiles:
                             f_smi.write(f"{smi}\n")
                     print(f"Saved corresponding valid SMILES to {smiles_output_path}")
                 except Exception as e:
                     print(f"Warning: Could not save SMILES file {smiles_output_path}. Error: {e}")

        elif args.mode == 'decode':
            print("--- Decoding Latent Vectors ---")
            if os.path.isfile(args.input):
                 print(f"Loading latent vectors from file: {args.input}")
                 try:
                     latent_vectors_flat = np.load(args.input)
                     print(f"Loaded latent vectors with shape: {latent_vectors_flat.shape}")
                 except Exception as e:
                     print(f"Error loading latent vectors from {args.input}: {e}")
                     print("Please provide a .npy file.")
                     return
            else:
                 print("Error: Input for decode mode must be a file path to a .npy file.")
                 return

            # --- Shape Validation (more robust) ---
            expected_z_dim = model_kwargs['zsize']
            expected_t_dim = model_kwargs['max_T'] # Model expects max_T length internally for positional encoding
            expected_flat_dim = expected_t_dim * expected_z_dim

            if latent_vectors_flat.ndim != 2:
                print(f"Error: Loaded latent vectors have {latent_vectors_flat.ndim} dimensions, expected 2 (batch, features).")
                return

            if latent_vectors_flat.shape[1] != expected_flat_dim:
                 print(f"Error: Latent vector feature dimension ({latent_vectors_flat.shape[1]}) "
                       f"does not match expected flat dimension ({expected_flat_dim} = max_T * zsize = {expected_t_dim} * {expected_z_dim}).")
                 print("Ensure the loaded vectors correspond to the model architecture (max_T, zsize).")
                 return
            # --- End Shape Validation ---


            # Process in batches
            num_vectors = len(latent_vectors_flat)
            all_generated_smiles = []
            print(f"Decoding {num_vectors} latent vectors in batches of {args.batch_size}...")

            for i in tqdm(range(0, num_vectors, args.batch_size), desc="Decoding Batches"):
                batch_z_flat = latent_vectors_flat[i:min(i + args.batch_size, num_vectors)]
                if batch_z_flat.shape[0] == 0: continue

                input_z = torch.tensor(batch_z_flat, dtype=torch.float32).to(device)

                # Decode - model.decode_z handles reshaping z if needed internally
                with autocast(enabled=use_amp): # Wrap model call
                     generated_tokens, valid_mask, _, probs = model.decode_z(
                         z=input_z,
                         sampling=args.sample_decode,
                         temp=args.decode_temp,
                         length=args.decode_len # Max generation length
                     )

                if len(generated_tokens) == 0:
                    print(f"Batch {i // args.batch_size} generated no tokens.")
                    # Add placeholders if you need to maintain order corresponding to input Z
                    all_generated_smiles.extend(["GENERATION_FAILED"] * len(batch_z_flat))
                    continue

                # Convert token IDs back to SELFIES strings
                generated_selfies = dataobj.decode(generated_tokens)

                # Convert SELFIES to SMILES
                for selfie_str in generated_selfies:
                    try:
                        smi = sf.decoder(selfie_str)
                        # No RDKit check here for speed, model.canonicalize handles it if enabled
                        all_generated_smiles.append(smi)
                    except Exception as e:
                        all_generated_smiles.append(f"DECODING_ERROR: {e}")

            print("\n--- Generated Molecules (Examples) ---")
            # Print first few examples
            for i, smi in enumerate(all_generated_smiles[:10]):
                print(f"Decoded {i}: {smi}")
            if len(all_generated_smiles) > 10:
                print(f"... (showing first 10 of {len(all_generated_smiles)})")


            if args.output_file:
                 output_path = Path(args.output_file)
                 output_path.parent.mkdir(parents=True, exist_ok=True)
                 try:
                     with open(output_path, 'w') as f:
                         for smi in all_generated_smiles:
                             f.write(f"{smi}\n")
                     print(f"\nSaved {len(all_generated_smiles)} generated SMILES to {output_path}")
                 except Exception as e:
                     print(f"Warning: Could not save SMILES file {output_path}. Error: {e}")

        else:
            # This case should not be reachable due to argparse choices
            print(f"Error: Invalid mode '{args.mode}'. Choose 'encode' or 'decode'.")


# --- Argument Parser ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use Pretrained SeqFlow Model for Encoding/Decoding")

    # Required Args
    parser.add_argument("mode", type=str, choices=['encode', 'decode'], help="Operation mode: 'encode' (SMILES->latent) or 'decode' (latent->SMILES)")
    parser.add_argument("input", type=str, help="Input data. For 'encode': SMILES string or path to .smi file. For 'decode': path to .npy file containing latent vectors (shape B, D).")
    parser.add_argument("--model_path", type=str, default="./model_weight/seqflow_1.27M_3.pt", help="Path to the pretrained SeqFlow model (.pt file)")

    # Optional Output Args
    parser.add_argument("-o", "--output_file", type=str, default=None, help="Output file path. For 'encode': saves .npy latent vectors (and .smi). For 'decode': saves .smi generated SMILES.")

    # Inference Control Args
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for inference processing to manage memory.")
    parser.add_argument("--max_len_encode", type=int, default=128, help="Maximum sequence length for padding/truncation during SMILES encoding.")
    parser.add_argument("--decode_len", type=int, default=100, help="Maximum length of molecules to generate during decoding.")
    parser.add_argument("--decode_temp", type=float, default=1.0, help="Temperature for sampling during decoding (if --sample_decode is True). Higher values -> more random.")
    parser.add_argument("--sample_z", action='store_true', help="Sample z from distribution during encoding instead of using the mean.")
    parser.add_argument("--sample_decode", action='store_true', help="Use multinomial sampling during decoding instead of argmax.")
    parser.add_argument("--canonicalize_input", action='store_true', help="Canonicalize input SMILES before SELFIES conversion (during encoding).")
    parser.add_argument("--no-canonicalize_input", dest='canonicalize_input', action='store_false')
    parser.set_defaults(canonicalize_input=True)
    parser.add_argument("--canonicalize_output", action='store_true', help="Attempt to canonicalize decoded SMILES via RDKit roundtrip within the model's decode_z.")
    parser.add_argument("--no-canonicalize_output", dest='canonicalize_output', action='store_false')
    parser.set_defaults(canonicalize_output=True)


    # System Args
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index to use (-1 for CPU)")
    parser.add_argument("--cache_dir", type=str, default="./.cache_seqflow_preprocess", help="Directory to store/load preprocessed SMILES data.")


    args = parser.parse_args()

    # Basic input validation
    if args.mode == 'decode' and not (args.input.endswith('.npy') and os.path.isfile(args.input)):
         parser.error("Input for 'decode' mode must be an existing .npy file.")
    if args.mode == 'encode' and not isinstance(args.input, str):
         parser.error("Input for 'encode' mode must be a SMILES string or a file path.")


    main(args)