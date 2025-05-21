# src/rl_policy.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class PolicyNetwork(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int = 512, num_layers: int = 3, log_std_min: float = -20.0, log_std_max: float = 2.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim # Should be same as latent_dim if u_z is the action
        self.hidden_dim = hidden_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        layers = []
        current_dim = latent_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.Mish())
            current_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*layers)
        
        self.mean_head = nn.Linear(current_dim, action_dim)
        self.log_std_head = nn.Linear(current_dim, action_dim)

    def forward(self, state_z: torch.Tensor):
        x = self.shared_layers(state_z)
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        
        return mean, std

    def sample_action(self, state_z: torch.Tensor):
        mean, std = self.forward(state_z)
        distribution = Normal(mean, std)
        action_u_z = distribution.rsample() # rsample for reparameterization trick if needed for other algos
        log_prob = distribution.log_prob(action_u_z).sum(dim=-1) # Sum log_probs across action dimensions
        return action_u_z, log_prob