from torch import nn
import pickle

import torch


class CNN(nn.Module):
    """Base Model for CNN class."""

    def __init__(self, device: torch.device) -> None:
        """Set values for all CNNs."""
        self.device = device

    def from_bytestream(self, bytestream: bytes) -> None:
        """Read weigths from from pickled state dict."""
        self.load_state_dict(pickle.loads(bytestream))

    def to_bytestream(self) -> bytes:
        """Dump weigths to bytestream."""
        return pickle.dumps(self.state_dict())
