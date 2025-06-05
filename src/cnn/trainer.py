from torch import nn, optim
from torch.utils.data import DataLoader
from src.cnn.base_cnn import CNN
from src.config import Settings


class Trainer:
    """Encapsulates the training process."""

    def __init__(self, optimizer, loss_fn) -> None:
        """Initialize training necessities."""
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    @classmethod
    def from_config(cls, model: CNN, config: Settings) -> "Trainer":
        """TODO: Read from config."""
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        loss_fn = nn.MSELoss()
        instance = cls(optimizer, loss_fn)
        return instance

    def train(self, loader: DataLoader) -> None:
        """Run training on loaded data."""
        for episode_data in loader:
            pass
