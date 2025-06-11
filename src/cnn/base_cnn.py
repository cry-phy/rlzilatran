from torch import nn
import pickle


class CNN(nn.Module):
    """Base Model for CNN class."""

    def from_bytestream(self, bytestream: bytes) -> None:
        """Read weigths from from pickled state dict."""
        self.load_state_dict(pickle.loads(bytestream))

    def to_bytestream(self) -> bytes:
        """Dump weigths to bytestream."""
        return pickle.dumps(self.state_dict())
