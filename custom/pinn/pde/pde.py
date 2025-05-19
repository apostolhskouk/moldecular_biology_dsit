from abc import ABC
import torch
from torch import nn, Tensor
from typing import Optional
import math
from ChemFlow.src.pinn.pde import SinusoidalPositionEmbeddings
import warnings

# ChemFlow-main/src/pinn/pde/pde.py
# (Keep SinusoidalPositionEmbeddings and PDE classes as they are)

# --- Start Replacement ---
class MLP(nn.Module):
    """
    Modified MLP to handle sequence inputs (e.g., from MolGen-Transformer)
    by pooling the sequence before processing.
    """
    def __init__(
        self,
        n_in_feature: int, # Renamed from n_in to clarify it's feature dim
        n_in_sequence: Optional[int] = None, # Optional: sequence length (not strictly needed for pooling)
        n_out: int = 1,
        h: int = 512,
        time_embed_dim: int = 512 # Dimension for time embedding, can be different
    ):
        super(MLP, self).__init__()

        self.is_sequence = n_in_sequence is not None # Check if sequence length is provided

        # --- Embedding for the main input (x) ---
        # If it's a sequence, pool it first, then apply Linear.
        # If not a sequence, just apply Linear (original behavior).
        if self.is_sequence:
            # Pool sequence dim (dim=1 assuming input B, seq_len, features)
            # Then map pooled features (n_in_feature) to hidden dim (h)
            self.x_embedding = nn.Sequential(
                # nn.AdaptiveAvgPool1d(1), # Alternative pooling
                # nn.Flatten(), # Flatten after pooling
                # nn.Linear(n_in_feature, h), # Map pooled features to h
                # --- Using mean pooling ---
                 nn.Linear(n_in_feature, h), # Map features to h *before* pooling? Maybe better.
                 nn.ReLU() # Activation after linear map
            )
            print(f"MLP configured for SEQUENCE input (pooling features {n_in_feature} -> {h}).")
        else:
            # Original behavior for non-sequence input (B, n_in_feature)
            self.x_embedding = nn.Sequential(
                nn.Linear(n_in_feature, h),
                nn.ReLU(),
            )
            print(f"MLP configured for VECTOR input (features {n_in_feature} -> {h}).")


        # --- Embedding for time (t) ---
        # Use a separate dimension for time embedding if specified
        self.t_embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(time_embed_dim), # Use time_embed_dim
            nn.Linear(time_embed_dim, h),                # Map time embedding to hidden dim (h)
            nn.ReLU(),
            nn.Linear(h, h),                             # Optional extra layer for time
        )

        # --- Final MLP layers ---
        dims = [h, h, n_out] # Combine x and t embeddings (size h each, element-wise add => size h)
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)  # (b, h) -> (b, n_out)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Parametrized potential energy function u(x, t).

        Args:
            x (Tensor): Input tensor. Shape (b, n_in_feature) or (b, n_in_sequence, n_in_feature).
            t (Tensor): Time tensor. Shape (b,).

        Returns:
            Tensor: Potential energy u(x, t). Shape (b, n_out).
        """
        if self.is_sequence:
            # Expecting x shape: (batch_size, n_in_sequence, n_in_feature)
            if not (len(x.shape) == 3 and x.shape[2] == self.hparams.n_in_feature):
                 warnings.warn(f"MLP forward expected sequence shape (B, SeqLen, FeatDim={self.hparams.n_in_feature}) but got {x.shape}. Attempting mean pooling over dim 1.")
            # Apply embedding to features first, then pool sequence dimension (dim=1)
            # x_feat_embedded = self.x_embedding(x) # Apply Linear(feat->h)+ReLU to each element: (B, Seq, h)
            # x_pooled = x_feat_embedded.mean(dim=1) # Pool sequence: (B, h)
            # --- Alternative: Pool first, then embed ---
            x_pooled = x.mean(dim=1) # Pool sequence: (B, n_in_feature)
            x_embedded = self.x_embedding(x_pooled) # Apply embedding: (B, h)
        else:
            # Expecting x shape: (batch_size, n_in_feature)
            if not (len(x.shape) == 2 and x.shape[1] == self.hparams.n_in_feature):
                 warnings.warn(f"MLP forward expected vector shape (B, FeatDim={self.hparams.n_in_feature}) but got {x.shape}. Check input.")
            x_embedded = self.x_embedding(x) # (B, h)

        # Embed time
        t_embedded = self.t_embedding(t) # (B, h)

        # Combine and pass through final layers
        combined = x_embedded + t_embedded
        return self.mlp(combined)
# --- End Replacement ---

# (Keep SinusoidalPositionEmbeddings and PDE classes as they are)