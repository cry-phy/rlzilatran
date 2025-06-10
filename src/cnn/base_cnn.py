from torch import nn
import pickle


class CNN(nn.Module):
    """Base Model for CNN class."""

    def from_bytestream(self, bytestream: bytearray) -> None:
        """Read weigths from from pickled state dict."""
        self.load_state_dict(pickle.loads(bytestream))

    def to_bytestream(self) -> bytearray:
        """Dump weigths to bytestream."""
        return bytearray(pickle.dumps(self.state_dict()))
