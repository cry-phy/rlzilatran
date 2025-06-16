from torch import nn
import torch

from cnn.base_cnn import CNN


class SmallCNN(CNN):
    """Small CNN fof testing Datatloader and procedures."""

    def __init__(
        self,
        num_filters: int,
        state_shape: tuple[int, ...],
        move_shape: tuple[int, ...],
        device: torch.device,
    ) -> None:
        """Generate CNN."""
        super().__init__(device)

        self.num_filters = num_filters
        self.state_shape = state_shape
        self.input_channels = state_shape[0]
        self.move_shape = move_shape

        self.trunk = nn.Sequential(
            nn.Conv2d(self.input_channels, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(num_filters, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(1 * 4 * 8, 1),
            nn.Tanh(),
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_filters, move_shape[2], kernel_size=1),
            nn.BatchNorm2d(move_shape[2]),
            nn.ReLU(inplace=True),
        )

    def forward(self, state):
        """Inference on the CNN."""
        intermediate = self.trunk(state)
        value = self.value_head(intermediate)
        moves = self.policy_head(intermediate)
        value = value.squeeze(1)
        moves = moves.permute(0, 2, 3, 1)
        return moves, value
