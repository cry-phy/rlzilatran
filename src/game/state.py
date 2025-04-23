from game.utils import board_generator
import torch


class State:
    """Wrapper around game state pytorch tensor.

    Contains:
        - Game State
        - Available moves
        - Reward
    """

    def __init__(self, tensor: torch.Tensor) -> None:
        """Generate State instance.

        Should not be used directly! Use factory Methods instead.
        """
        self._tensor: torch.Tensor = tensor
        self._moves: torch.Tensor

    @classmethod
    def initial_state(cls) -> "State":
        """Generate Initial game State."""
        return cls(board_generator.initial_board())

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "State":
        """Generate State from state Tensor."""
        return cls(tensor)

    def set_state(self, tensor: torch.Tensor) -> None:
        """Set state Tensor."""
        self._tensor = tensor

    def get_moves(self) -> torch.Tensor:
        """Return viable moves from state."""
        ...

    def get_move_mask(self) -> torch.Tensor:
        """Mask out invalid moves."""
        ...

    def get_state(self) -> torch.Tensor:
        """Return state Tensor for training."""
        return self._tensor.clone()

    def make_move(self, index: int) -> None:
        """Make move corresponding to index."""
        ...

    def check_half_empty(self) -> bool:
        """Termination state this half of the board is empty."""
        return not bool(self._tensor[:, :, :4].sum())

    def copy(self) -> "State":
        """Generate copy."""
        return self.from_tensor(self._tensor)

    def reset(self) -> None:
        """Set to initial state."""
        self._tensor = board_generator.initial_board()

    def save(self, path: str) -> None:
        """Save game state to torch tensor file."""
        torch.save(self._tensor, path)

    def load(self, path: str) -> None:
        """Load game state from torch tensor file."""
        self._tensor = torch.load(path)
