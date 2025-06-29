import torch
from torch import nn
from cnn.base_cnn import CNN


class TrunkBlock(nn.Module):
    """Feature extraction block."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Init Module."""
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass on state tensor."""
        return self.block(x)


class ValueHead(nn.Module):
    """Predict scalar value."""

    def __init__(self, in_channels: int, spatial_shape: tuple[int, int]) -> None:
        """Init Module."""
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(1 * spatial_shape[0] * spatial_shape[1], 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass on state tensor."""
        return self.block(x).squeeze(1)


class PolicyHead(nn.Module):
    """Predict move logits."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Init Module."""
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass on state tensor."""
        return self.block(x).permute(0, 2, 3, 1)


class SmallCNN(CNN):
    """Small CNN for training/debugging."""

    def __init__(
        self,
        num_filters: int,
        state_shape: tuple[int, ...],
        move_shape: tuple[int, ...],
        device: torch.device,
    ) -> None:
        """Initialize SmallCNN."""
        super().__init__(device)
        self.num_filters = num_filters
        self.state_shape = state_shape
        self.move_shape = move_shape
        self.input_channels = state_shape[0]
        spatial_shape = state_shape[1:]

        self.trunk = TrunkBlock(self.input_channels, num_filters)
        self.value_head = ValueHead(num_filters, spatial_shape)
        self.policy_head = PolicyHead(num_filters, move_shape[2])

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass on state tensor."""
        x = self.trunk(state)
        value = self.value_head(x)
        moves = self.policy_head(x)
        return moves, value
