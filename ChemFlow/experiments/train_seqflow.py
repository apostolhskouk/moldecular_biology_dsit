import os
import argparse
import random
import math
from pathlib import Path
from rdkit import Chem
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm
import selfies as sf
import wandb  # Optional: for logging
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'NFBO'))

# Import necessary components from the provided repository
from objective.guacamol.utils.mol_utils.data import SELFIESDataset
from generative_model.model.seqflow.lstm_flow import AFPrior

# --- Helper Functions ---

def set_seed(seed):
    """Sets random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_and_preprocess_data(data_path, dataobj, max_len, canonicalize=True):
    """Loads SMILES, converts to SELFIES, tokenizes, encodes, pads, and caches the result."""
    data_path = Path(data_path)
    # Build a unique cache filename based on input + params
    cache_name = (
        f"{data_path.stem}_max{max_len}_can{int(canonicalize)}_v{dataobj.vocab_size}.cache.pt"
    )
    cache_path = data_path.parent / cache_name

    # If cache exists, load and return
    if cache_path.exists():
        print(f"Loading tokenized data from cache: {cache_path}")
        cache = torch.load(cache_path, map_location="cpu")
        return cache["encoded_tensor"], cache["original_smiles"]

    # Otherwise preprocess from scratch
    print(f"No cache found at {cache_path}. Preprocessing from scratch.")
    try:
        with open(data_path, 'r') as f:
            smiles_list = [line.strip() for line in f if line.strip()]
        print(f"Read {len(smiles_list)} SMILES strings.")
    except Exception as e:
        print(f"Error reading data file: {e}")
        return None

    # 1) SMILES -> SELFIES
    print("Converting SMILES to SELFIES...")
    selfies_list = []
    original_indices = []
    for i, smi in enumerate(tqdm(smiles_list, desc="SMILES to SELFIES")):
        try:
            s = sf.encoder(smi)
            if canonicalize:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    canon_smi = Chem.MolToSmiles(mol)
                    s = sf.encoder(canon_smi) if canon_smi else s
            selfies_list.append(s)
            original_indices.append(i)
        except Exception:
            continue
    print(f"Converted {len(selfies_list)} SMILES to valid SELFIES.")

    # 2) Tokenize
    print("Tokenizing SELFIES...")
    tokenized_selfies = dataobj.tokenize_selfies(selfies_list)

    # 3) Encode and pad
    print("Encoding tokens and padding...")
    encoded_data = []
    valid_indices = []
    skipped_unknown = 0
    stop_idx = dataobj.vocab2idx.get('<stop>', 1)

    for i, tokens in enumerate(tqdm(tokenized_selfies, desc="Encoding Tokens")):
        valid = True
        encoded = []
        for token in tokens:
            if token not in dataobj.vocab2idx:
                skipped_unknown += 1
                valid = False
                break
            encoded.append(dataobj.vocab2idx[token])
        if not valid:
            continue
        if len(encoded) > max_len:
            continue
        # pad
        padding = [stop_idx] * (max_len - len(encoded))
        encoded_data.append(encoded + padding)
        valid_indices.append(original_indices[i])

    if skipped_unknown:
        print(f"Skipped {skipped_unknown} molecules containing unknown tokens.")
    print(f"Encoded {len(encoded_data)} valid molecules.")
    if not encoded_data:
        print("No valid data after processing. Exiting.")
        return None

    tensor = torch.tensor(encoded_data, dtype=torch.long)
    valid_smiles = [smiles_list[i] for i in valid_indices]

    # Save cache
    torch.save({
        "encoded_tensor": tensor,
        "original_smiles": valid_smiles
    }, cache_path)
    print(f"Saved cache to {cache_path}")

    return tensor, valid_smiles


# --- Main Training Script ---

def main(args):
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Setup Data
    dataobj = SELFIESDataset(data_type='guacamol')
    vocab_size = dataobj.vocab_size
    print(f"Vocabulary size: {vocab_size}")
    max_len = args.max_len

    processed_data, original_smiles = load_and_preprocess_data(
        args.data_path,
        dataobj,
        max_len,
        args.canonicalize
    )
    if processed_data is None:
        return

    dataset = TensorDataset(processed_data)
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # 2. Setup Model
    model_kwargs = {
        'vocab_size': vocab_size,
        'hidden_size': args.hidden_size,
        'zsize': args.zsize,
        'dropout_p': args.dropout_p,
        'dropout_locations': ['prior_rnn'],
        'prior_type': 'AF',
        'num_flow_layers': args.num_flow_layers,
        'rnn_layers': args.rnn_layers,
        'max_T': args.model_max_T,
        'transform_function': args.transform_function,
        'hiddenflow_params': {
            'nohiddenflow': False,
            'hiddenflow_layers': 2,
            'hiddenflow_units': 100,
            'hiddenflow_flow_layers': 10,
            'hiddenflow_scf_layers': True
        },
        'noise': args.noise,
        'sim_coef': args.sim_coef,
        'cliping': args.cliping,
        'canonical': args.canonicalize
    }

    model = AFPrior(**model_kwargs).to(device)
    model.set_dataobj(dataobj)

    # Optional: load pretrained
    if args.pretrained_path:
        try:
            state_dict = torch.load(args.pretrained_path, map_location=device)
            model.load_state_dict(state_dict)
            print(f"Loaded pretrained model from {args.pretrained_path}")
        except Exception as e:
            print(f"Warning: Could not load pretrained model: {e}")

    # 3. Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # 4. Logging
    if args.use_wandb:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=args)
        wandb.watch(model, log='all', log_freq=100)

    # 5. Training Loop
    print("Starting training...")
    model.train()
    global_step = 0
    for epoch in range(args.epochs):
        epoch_loss = epoch_logp = epoch_recon = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        for batch in pbar:
            inputs = batch[0].to(device)
            optimizer.zero_grad()
            z, loss, out = model(inputs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_norm)
            optimizer.step()

            batch_loss = loss.item()
            batch_logp = out.get('log_p_loss', torch.tensor(0.0)).item()
            batch_recon = out.get('recon_loss', torch.tensor(0.0)).item()
            epoch_loss += batch_loss
            epoch_logp += batch_logp
            epoch_recon += batch_recon

            pbar.set_postfix({
                'loss': f"{batch_loss:.4f}",
                'logp': f"{batch_logp:.4f}",
                'recon': f"{batch_recon:.4f}"})

            if args.use_wandb:
                wandb.log({
                    "train/batch_loss": batch_loss,
                    "train/batch_logp": batch_logp,
                    "train/batch_recon": batch_recon,
                    "global_step": global_step
                })
            global_step += 1

        # Epoch summary
        n_batches = len(train_loader)
        print(f"Epoch {epoch+1} Summary: Avg Loss: {epoch_loss/n_batches:.4f}, "
              f"Avg LogP: {epoch_logp/n_batches:.4f}, "
              f"Avg Recon: {epoch_recon/n_batches:.4f}")
        if args.use_wandb:
            wandb.log({
                "train/epoch_loss": epoch_loss/n_batches,
                "train/epoch_logp": epoch_logp/n_batches,
                "train/epoch_recon": epoch_recon/n_batches,
                "epoch": epoch+1
            })

        # Generate check, save, etc. (unchanged)
        if (epoch+1) % args.generate_check_freq == 0:
            model.eval()
            with torch.no_grad():
                print("\n--- Generating Samples ---")
                gen, _, _, _ = model.decode_z(B=5, length=max_len, sampling=True)
                gf = dataobj.decode(gen)
                for i, sf_str in enumerate(gf):
                    try:
                        print(f" Sample {i+1}: {sf.decoder(sf_str)} ({sf_str})")
                    except:
                        print(f" Sample {i+1}: Error decoding {sf_str}")
                print("\n--- Reconstruction ---")
                if len(processed_data) > 0:
                    idxs = random.sample(range(len(processed_data)), k=min(5, len(processed_data)))
                    batch = processed_data[idxs].to(device)
                    z_recon, _, _ = model.encode_z(batch, sampling=False)
                    recon, _, _ = model.decode_z(z=z_recon, sampling=False)
                    rf = dataobj.decode(recon)
                    for j, orig_idx in enumerate(idxs):
                        try:
                            print(f" Orig {orig_idx}: {original_smiles[orig_idx]}")
                            print(f" Recon {orig_idx}: {sf.decoder(rf[j])} ({rf[j]})")
                        except:
                            continue
            model.train()

        if (epoch+1) % args.save_freq == 0 or (epoch+1) == args.epochs:
            out_dir = Path(args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            ckpt = out_dir / f"seqflow_epoch_{epoch+1}.pt"
            torch.save(model.state_dict(), ckpt)
            print(f"Saved checkpoint to {ckpt}")

    print("Training finished.")
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SeqFlow VAE Model with caching")
    parser.add_argument("--data_path", type=str, required=True, help="Path to SMILES file")
    parser.add_argument("--output_dir", type=str, default="./trained_seqflow", help="Where to save models")
    parser.add_argument("--max_len", type=int, default=128, help="Max sequence length")
    parser.add_argument("--canonicalize", action='store_true', help="Canonicalize during preprocessing")
    parser.add_argument("--no-canonicalize", dest='canonicalize', action='store_false')
    parser.set_defaults(canonicalize=True)
    # (other args same as before...)
    parser.add_argument("--hidden_size", type=int, default=500)
    parser.add_argument("--zsize", type=int, default=40)
    parser.add_argument("--dropout_p", type=float, default=0.2)
    parser.add_argument("--num_flow_layers", type=int, default=5)
    parser.add_argument("--rnn_layers", type=int, default=2)
    parser.add_argument("--model_max_T", type=int, default=288)
    parser.add_argument("--transform_function", type=str, default="nlsq", choices=["nlsq","affine"])
    parser.add_argument("--noise", type=float, default=0.1)
    parser.add_argument("--sim_coef", type=float, default=1.0)
    parser.add_argument("--cliping", action='store_true')
    parser.add_argument("--no-cliping", dest='cliping', action='store_false')
    parser.set_defaults(cliping=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--save_freq", type=int, default=10)
    parser.add_argument("--generate_check_freq", type=int, default=5)
    parser.add_argument("--pretrained_path", type=str, default=None)
    parser.add_argument("--use_wandb", action='store_true')
    parser.add_argument("--wandb_project", type=str, default="seqflow-training")
    parser.add_argument("--wandb_entity", type=str, default=None)

    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
