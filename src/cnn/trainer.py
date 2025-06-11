from collections.abc import Callable
from torch import nn, optim
from torch.utils.data import DataLoader
from cnn.base_cnn import CNN
from config import Settings
import logging

logger: logging.Logger = logging.getLogger(__name__)


class Trainer:
    """Encapsulates the training process."""

    def __init__(
        self,
        model: CNN,
        optimizer: optim.Optimizer,
        loss_fn: Callable,
    ) -> None:
        """Initialize training necessities."""
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = model.device

    @classmethod
    def from_config(cls, model: CNN, config: Settings) -> "Trainer":
        """TODO: Read from config."""
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        loss_fn = nn.MSELoss()
        instance = cls(
            model,
            optimizer,
            loss_fn,
        )
        return instance

    def train(self, loader: DataLoader) -> None:
        """Run training on loaded data."""
        self.model.train()

        for batch in loader:
            inputs, targets = batch
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()

            logger.info(f"Loss: {loss.item():.4f}")
