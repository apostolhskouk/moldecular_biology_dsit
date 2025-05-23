from abc import ABC
from torch import nn, Tensor

from ChemFlow.src.vae import VAE
from ChemFlow.src.predictor import Predictor


class Generator(nn.Module, ABC):
    latent_size: int
    reverse_size: int

    def forward(self, z: Tensor) -> Tensor: ...



class VAEGenerator(Generator):
    def __init__(self, model: VAE):
        super().__init__()

        self.model = model
        self.latent_size = model.latent_dim
        self.reverse_size = model.max_len * model.vocab_size

    def forward(self, z: Tensor) -> Tensor:
        return self.model.decode(z)


class PropGenerator(Generator):
    def __init__(self, vae: VAE, model: Predictor):
        super().__init__()

        self.vae = vae
        self.model = model
        self.latent_size = vae.latent_dim
        self.reverse_size = 1

    def forward(self, z: Tensor) -> Tensor:
        return self.model(self.vae.decode(z).exp())
