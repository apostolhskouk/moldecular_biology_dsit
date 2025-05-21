# experiments/train_rl_policy.py
import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.cli import LightningArgumentParser # Can be used if desired
from torch.utils.data import DataLoader, TensorDataset 

import numpy as np
import pandas as pd
from pathlib import Path
from collections import deque

from cd2root import cd2root
cd2root()

from src.vae import load_vae, VAE
from src.predictor import Predictor
from src.pinn.generator import PropGenerator # To use the property predictor
from src.utils.scores import PROP_FN, MINIMIZE_PROPS
from src.rl_policy import PolicyNetwork

class RLEnvironment:
    def __init__(self, vae_dm, vae_model, prop_predictor_model, property_name, minimize_property, device):
        self.dm = vae_dm
        self.vae = vae_model
        self.prop_predictor = prop_predictor_model # This is the PropGenerator
        self.property_name = property_name
        self.minimize_property = minimize_property
        self.device = device
        
        self.prop_fn = PROP_FN[property_name]



    @torch.no_grad()
    def calculate_property_for_z(self, z_vector: torch.Tensor):
        # Decode z to SMILES and then calculate property
        # VAE returns log_softmax, no .exp() needed if dm.decode handles it
        reconstructed_log_probs = self.vae.decode(z_vector)
        smiles_list = self.dm.decode(reconstructed_log_probs) # Expects list of SMILES
        
        # Calculate property for each SMILES string
        # Handle cases where some SMILES might be invalid for property calculation
        properties = []
        valid_smiles_indices = []
        valid_smiles_for_prop_calc = []

        for i, s in enumerate(smiles_list):
            try:
                # Ensure SMILES is valid before property calculation if prop_fn is sensitive
                mol = Chem.MolFromSmiles(s)
                if mol:
                    valid_smiles_indices.append(i)
                    valid_smiles_for_prop_calc.append(s)
                else:
                    properties.append(np.nan) # Or a very bad score
            except: # Catch any error from MolFromSmiles or property calculation
                properties.append(np.nan)
        
        if valid_smiles_for_prop_calc:
            calculated_props = self.prop_fn(valid_smiles_for_prop_calc)
            # Fill back into the properties list
            prop_iter = iter(calculated_props)
            full_properties = [np.nan] * len(smiles_list)
            for i, original_idx in enumerate(valid_smiles_indices):
                full_properties[original_idx] = calculated_props[i]
            return torch.tensor(full_properties, dtype=torch.float32, device=self.device)
        else:
            return torch.full((len(smiles_list),), np.nan, dtype=torch.float32, device=self.device)


    @torch.no_grad()
    def step(self, current_z: torch.Tensor, action_u_z: torch.Tensor):
        next_z = current_z + action_u_z
        
        # Calculate property of current_z and next_z using the actual property function
        # This is the "true" reward, not from the predictor necessarily
        prop_current_z_actual = self.calculate_property_for_z(current_z)
        prop_next_z_actual = self.calculate_property_for_z(next_z)

        reward = prop_next_z_actual - prop_current_z_actual
        
        # Handle NaN rewards (e.g., from invalid SMILES)
        # Replace NaNs with a very negative reward or zero, depending on strategy
        nan_mask = torch.isnan(reward)
        reward[nan_mask] = -10.0 # Penalize heavily for invalid transitions

        if self.minimize_property:
            reward = -reward # If minimizing, a decrease in property is a positive reward
            
        return next_z.detach(), reward.detach()


class REINFORCELightning(L.LightningModule):
    def __init__(self, latent_dim: int, action_dim: int, property_name: str, minimize_property: bool,
                 vae_dm, vae_model, prop_predictor_model, # For environment
                 hidden_dim_policy: int = 512, num_layers_policy: int = 3,
                 learning_rate: float = 1e-4, gamma: float = 0.99,
                 trajectory_length: int = 5, batch_size_rl: int = 64, # This is episodes per opt step
                 action_scale: float = 0.1,
                 dummy_dataloader_batch_size: int = 1): # New parameter for dummy dataloader
        super().__init__()
        # Add dummy_dataloader_batch_size to hparams if you want to configure it
        self.save_hyperparameters(ignore=["vae_dm", "vae_model", "prop_predictor_model"]) 
        
        self.policy_network = PolicyNetwork(latent_dim, action_dim, hidden_dim_policy, num_layers_policy)
        self.env = RLEnvironment(vae_dm, vae_model, prop_predictor_model, property_name, minimize_property, self.device)
        
        self.lr = learning_rate
        self.gamma = gamma
        self.trajectory_length = trajectory_length
        self.batch_size_rl = batch_size_rl # Number of episodes per optimizer step
        self.action_scale = action_scale
        self.dummy_dataloader_batch_size = dummy_dataloader_batch_size


        self.automatic_optimization = False
    def train_dataloader(self):
        # Returns a dummy dataloader. The batch content isn't used in training_step.
        # The length of this dataset determines how many times training_step is called per epoch.
        # We want training_step to be called effectively once per "RL batch collection and update".
        # So, a dataset of size 1 is fine if one call to training_step does one full RL update.
        dummy_data = TensorDataset(torch.zeros(self.batch_size_rl, 1)) # Content doesn't matter
        return DataLoader(dummy_data, batch_size=self.dummy_dataloader_batch_size) # Small batch size for dummy

    def val_dataloader(self):
        # Optional: if you want to run a validation epoch.
        # For RL, validation often means evaluating the policy without updates.
        # If ModelCheckpoint monitors a val_metric, this is needed.
        # Your current ModelCheckpoint monitors avg_episode_reward (from training_step).
        # If you add a proper validation_step later, this would feed it.
        dummy_val_data = TensorDataset(torch.zeros(self.batch_size_rl, 1))
        return DataLoader(dummy_val_data, batch_size=self.dummy_dataloader_batch_size)
    
    # Optional: Add a dummy validation_step if you want to log something or have ModelCheckpoint use it
    # def validation_step(self, batch_unused, batch_idx):
    #     # Implement evaluation logic here if needed, e.g., run N episodes and log average reward
    #     # For now, just pass if ModelCheckpoint doesn't rely on a val_loss/metric from here
    #     pass
    def training_step(self, batch_unused, batch_idx):
        optimizer = self.optimizers()
        optimizer.zero_grad()

        batch_log_probs = []
        batch_rewards = []
        batch_discounted_rewards = []
        
        avg_episode_reward = 0

        for _ in range(self.batch_size_rl): # Collect a batch of trajectories
            current_z = torch.randn(1, self.hparams.latent_dim, device=self.device) # Start from random z
            episode_rewards = []
            episode_log_probs = []
            
            current_prop = self.env.calculate_property_for_z(current_z.detach())

            for t in range(self.trajectory_length):
                action_u_z_raw, log_prob = self.policy_network.sample_action(current_z)
                action_u_z_scaled = torch.tanh(action_u_z_raw) * self.action_scale # Bound and scale action
                
                next_z, reward = self.env.step(current_z.detach(), action_u_z_scaled.detach())
                
                episode_rewards.append(reward.squeeze())
                episode_log_probs.append(log_prob)
                
                current_z = next_z
                if torch.isnan(reward.squeeze()): # Stop if episode becomes invalid
                    break
            
            avg_episode_reward += sum(r.item() for r in episode_rewards if not torch.isnan(r))

            # Calculate discounted rewards for this episode
            discounted_rewards = []
            R = 0
            for r in reversed(episode_rewards):
                if torch.isnan(r): R = -10.0 # Reset if invalid step encountered
                R = r + self.gamma * R
                discounted_rewards.insert(0, R)
            
            discounted_rewards_tensor = torch.tensor(discounted_rewards, device=self.device)
            
            # Normalize discounted rewards (optional but often helpful)
            if len(discounted_rewards_tensor) > 1:
                 discounted_rewards_tensor = (discounted_rewards_tensor - discounted_rewards_tensor.mean()) / (discounted_rewards_tensor.std() + 1e-9)

            batch_log_probs.extend(episode_log_probs)
            batch_discounted_rewards.append(discounted_rewards_tensor)

        if not batch_log_probs: # Skip if all episodes were invalid from start
            self.log("train_loss", torch.tensor(0.0), prog_bar=True)
            self.log("avg_episode_reward", 0.0, prog_bar=True)
            return None

        log_probs_tensor = torch.stack(batch_log_probs)
        
        # Pad discounted_rewards if trajectories have different lengths
        max_len = max(len(dr) for dr in batch_discounted_rewards)
        padded_discounted_rewards = torch.zeros(len(batch_discounted_rewards), max_len, device=self.device)
        for i, dr in enumerate(batch_discounted_rewards):
            padded_discounted_rewards[i, :len(dr)] = dr
        
        # Ensure log_probs_tensor and padded_discounted_rewards can be multiplied
        # This part needs careful alignment if trajectories have variable lengths.
        # For simplicity here, we assume REINFORCE updates per step using its discounted reward.
        # A common way is to flatten and align.
        
        policy_loss_terms = []
        current_log_prob_idx = 0
        for i in range(len(batch_discounted_rewards)): # For each episode in the batch
            episode_len = len(batch_discounted_rewards[i])
            for t_step in range(episode_len):
                log_p = batch_log_probs[current_log_prob_idx + t_step]
                discounted_r = batch_discounted_rewards[i][t_step]
                policy_loss_terms.append(-log_p * discounted_r)
            current_log_prob_idx += episode_len
            
        if not policy_loss_terms:
            self.log("train_loss", torch.tensor(0.0), prog_bar=True)
            self.log("avg_episode_reward", avg_episode_reward / self.batch_size_rl, prog_bar=True)
            return None

        policy_loss = torch.stack(policy_loss_terms).mean()
        
        self.manual_backward(policy_loss)
        optimizer.step()

        self.log("train_loss", policy_loss, prog_bar=True, on_epoch=True)
        self.log("avg_episode_reward", avg_episode_reward / self.batch_size_rl, prog_bar=True, on_epoch=True)
        return policy_loss


    def configure_optimizers(self):
        optimizer = optim.AdamW(self.policy_network.parameters(), lr=self.lr)
        return optimizer

    # We don't have a traditional validation loop here, but one could be added
    # to evaluate the policy by running it for N episodes without updates.


def main_train_rl_policy():
    # --- Configuration ---
    prop_name = "plogp"
    data_name = "zmc"
    minimize_prop = prop_name in MINIMIZE_PROPS
    dummy_dataloader_batch_size = 1
    latent_dim_vae = 1024 # From VAE
    action_dim_policy = latent_dim_vae # Policy outputs a change in z
    
    epochs = 10
    learning_rate_rl = 3e-5
    gamma_rl = 0.99
    trajectory_len_rl = 10 # Number of steps per episode
    batch_size_rl_episodes = 32 # Number of episodes per training batch
    action_scale_rl = 0.05 # Max magnitude of change per step (after tanh)
    hidden_dim_policy_rl = 256
    num_layers_policy_rl = 2
    seed = 42
    # --- End Configuration ---

    L.seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dm_vae, vae = load_vae(
        file_path=f"data/processed/{data_name}.smi",
        model_path=f"checkpoints/vae/{data_name}/checkpoint.pt", device=device
    )
    for p in vae.parameters(): p.requires_grad = False
    vae.eval()

    # The RL environment uses the *actual* property function, not the predictor for rewards.
    # However, a predictor might be used if the actual prop_fn is too slow for frequent calls.
    # For now, we assume prop_fn is fast enough.
    # If you wanted to use the predictor for reward shaping:
    # predictor_nn = Predictor(dm_vae.max_len * dm_vae.vocab_size).to(device)
    # predictor_nn.load_state_dict(torch.load(f"checkpoints/prop_predictor/{prop_name}/checkpoint.pt", map_location=device))
    # for p in predictor_nn.parameters(): p.requires_grad = False
    # predictor_nn.eval()
    # guidance_generator_for_reward = PropGenerator(vae, predictor_nn).to(device)
    guidance_generator_for_reward = None # Using actual prop_fn

    rl_model = REINFORCELightning(
        latent_dim=latent_dim_vae,
        action_dim=action_dim_policy,
        property_name=prop_name,
        minimize_property=minimize_prop,
        vae_dm=dm_vae,
        vae_model=vae,
        prop_predictor_model=guidance_generator_for_reward, # Pass None if using actual prop_fn
        hidden_dim_policy=hidden_dim_policy_rl,
        num_layers_policy=num_layers_policy_rl,
        learning_rate=learning_rate_rl,
        gamma=gamma_rl,
        trajectory_length=trajectory_len_rl,
        batch_size_rl=batch_size_rl_episodes,
        action_scale=action_scale_rl,
        dummy_dataloader_batch_size=dummy_dataloader_batch_size 
    ).to(device)

    output_dir_rl = Path(f"checkpoints/rl_policy/{data_name}/{prop_name}")
    output_dir_rl.mkdir(parents=True, exist_ok=True)

    trainer = L.Trainer(
        max_epochs=epochs,
        logger=[WandbLogger(project=f"soc_rl_policy_{data_name}", name=f"{prop_name}", entity="lakhs")], # Replace 'lakhs'
        callbacks=[
            LearningRateMonitor(logging_interval="epoch"),
            ModelCheckpoint(monitor="avg_episode_reward", mode="max", save_top_k=1, dirpath=output_dir_rl, save_last=True),
        ],
        accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1,
        # gradient_clip_val=1.0 # Optional: gradient clipping
    )

    print(f"Starting RL training for {prop_name}...")
    trainer.fit(rl_model) # No datamodule needed as data is generated on-the-fly

    # Save the final policy network
    best_model_path_ckpt = trainer.checkpoint_callback.best_model_path
    final_policy_save_path = output_dir_rl / "policy_network_final.pt"
    if best_model_path_ckpt:
        # For PL Callbacks, the model saved is the LightningModule. We need to extract the policy.
        loaded_lightning_model = REINFORCELightning.load_from_checkpoint(best_model_path_ckpt)
        torch.save(loaded_lightning_model.policy_network.state_dict(), final_policy_save_path)
        print(f"Saved best policy network to {final_policy_save_path}")
    else:
        torch.save(rl_model.policy_network.state_dict(), final_policy_save_path)
        print(f"Saved final policy network to {final_policy_save_path}")


if __name__ == "__main__":
    from rdkit import Chem # Add Chem import here for the environment
    main_train_rl_policy()
    
    
    