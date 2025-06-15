from torch import nn
import pickle

import torch


class CNN(nn.Module):
    """Base Model for CNN class.

    Derived Models must use device as an argument in their __init__.
    """

    def __init__(self, device: torch.device) -> None:
        """!!!Do not Use directly!!! Use CNN.create() instead."""
        super().__init__()
        print("cnn", device)
        self.device = device

    @classmethod
    def create(cls, device: torch.device, *args, **kwargs) -> "CNN":
        """Create CNN on device."""
        instance = cls(*args, **kwargs, device=device)
        instance.to_device()
        return instance

    def to_device(self) -> None:
        """Send model to corresponding device."""
        self.to(self.device)

    def from_bytestream(self, bytestream: bytes) -> None:
        """Read weigths from from pickled state dict."""
        self.load_state_dict(pickle.loads(bytestream))

    def to_bytestream(self) -> bytes:
        """Dump weigths to bytestream."""
        return pickle.dumps(self.state_dict())
