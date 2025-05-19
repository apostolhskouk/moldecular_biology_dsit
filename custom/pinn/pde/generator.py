from custom.molgen_interface import MolGenInterface
from torch import Tensor
from ChemFlow.src.pinn.generator import Generator, Predictor
import torch
class MolGenVAEGenerator(Generator):
    """
    Generator using the MolGenInterface to decode latent sequences to logits.
    Used for unsupervised guidance (Jacobian calculation).
    """
    def __init__(self, molgen_interface: MolGenInterface):
        super().__init__()

        self.molgen_interface = molgen_interface
        # For sequence input, 'latent_size' is ambiguous. Store both dims.
        self.latent_seq_len = molgen_interface.latent_seq_len
        self.latent_embed_dim = molgen_interface.latent_embed_dim
        # reverse_size is the flattened dimension of the decoder output (logits)
        self.reverse_size = molgen_interface.output_seq_len * molgen_interface.vocab_size

    def forward(self, z: Tensor) -> Tensor:
        """
        Decodes the latent sequence z to output logits.

        Args:
            z (Tensor): Latent sequence tensor, shape (B, latent_seq_len, latent_embed_dim).

        Returns:
            Tensor: Output logits tensor, flattened shape (B, output_seq_len * vocab_size).
        """
        # decode_to_logits already handles device placement and returns flattened logits
        return self.molgen_interface.decode_to_logits(z)


class MolGenPropGenerator(Generator):
    """
    Generator using MolGenInterface and a Predictor model.
    Decodes latent sequence z to logits, then feeds to the predictor.
    Used for supervised guidance.
    """
    def __init__(self, molgen_interface: MolGenInterface, predictor: Predictor):
        super().__init__()

        self.molgen_interface = molgen_interface
        self.predictor = predictor
        # Store latent dimensions
        self.latent_seq_len = molgen_interface.latent_seq_len
        self.latent_embed_dim = molgen_interface.latent_embed_dim
        # reverse_size for property prediction is typically 1 (scalar output)
        self.reverse_size = 1

    def forward(self, z: Tensor) -> Tensor:
        """
        Decodes latent sequence z to logits, applies softmax, then predicts property.

        Args:
            z (Tensor): Latent sequence tensor, shape (B, latent_seq_len, latent_embed_dim).

        Returns:
            Tensor: Predicted property tensor, shape (B, 1).
        """
        # 1. Decode z to logits (flattened: B, output_seq_len * vocab_size)
        # This is actually log_softmax output from the interface's approximation
        logp_flat = self.molgen_interface.decode_to_logits(z)
        self.predictor.to(logp_flat.device)
        predicted_prop = self.predictor(logp_flat) # Feed log-probabilities directly
        return predicted_prop