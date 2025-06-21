import torch
from torch import nn, Tensor
from ChemFlow.src.pinn.pde.pde import SinusoidalPositionEmbeddings

class NeuralODEFunc(nn.Module):
    def __init__(self, latent_dim: int, time_embedding_dim: int = 128, hidden_dim: int = 512, num_layers: int = 3):
        super().__init__()
        self.latent_dim = latent_dim
        self.time_embedding_dim = time_embedding_dim
        self.hidden_dim = hidden_dim

        self.time_embedder = SinusoidalPositionEmbeddings(time_embedding_dim)
        
        layers = []
        current_dim = latent_dim + time_embedding_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.Mish())
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, latent_dim))
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, t: Tensor, z: Tensor) -> Tensor:
        if t.ndim == 0:
            t_expanded = t.unsqueeze(0).repeat(z.size(0))
        elif t.ndim == 1 and len(t) == 1 and z.size(0) > 1:
            t_expanded = t.repeat(z.size(0))
        else:
            t_expanded = t

        t_emb = self.time_embedder(t_expanded)
        zt = torch.cat([z, t_emb], dim=-1)
        return self.mlp(zt)