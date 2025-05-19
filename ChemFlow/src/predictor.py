from torch import Tensor, nn
from typing import List, Optional

class Block(nn.Module):
    """
    residual block
    """

    def __init__(self, features: int): # Input and output features are the same
        super().__init__()
        self.block = nn.Sequential(
            # LayerNorm might be more stable than BatchNorm if needed later
            # nn.LayerNorm(features),
            nn.Mish(),
            nn.Linear(features, features),
            # nn.LayerNorm(features),
            nn.Mish(),
            nn.Linear(features, features),
        )
        self.skip = nn.Identity() # Skip connection always works

    def forward(self, x: Tensor):
        return self.block(x) + self.skip(x)

class Predictor(nn.Module):
    """ Simplified Predictor using SimpleBlocks. """
    def __init__(
        self,
        input_dim: int, # Renamed latent_size to input_dim
        hidden_sizes: Optional[List[int]] = None # Corrected type hint
        ):
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [1024, 1024] # Default hidden sizes

        layers = []
        # Initial Linear layer to project input_dim to the first hidden size
        layers.append(nn.Linear(input_dim, hidden_sizes[0]))
        layers.append(nn.Mish()) # Activation after initial projection

        # Add simplified residual blocks
        for h_size in hidden_sizes:
            layers.append(Block(h_size))

        # Final layer to output a single value
        layers.append(nn.Linear(hidden_sizes[-1], 1))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor):
        return self.mlp(x)