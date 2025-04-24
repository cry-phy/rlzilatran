from game.utils import board_generator
import torch


class State:
    """Wrapper around game state pytorch tensor.

    Contains:
        - Pieces State
        - Game End Logic
        - Reward
    """

    def __init__(self, tensor: torch.Tensor) -> None:
        """Generate State instance.

        Should not be used directly! Use factory Methods instead.
        """
        self._tensor: torch.Tensor = tensor
        self._moves: torch.Tensor
        self._captured_value: float = 0
        self._board_size = board_generator.board_size()

    @classmethod
    def initial_state(cls) -> "State":
        """Generate Initial game State."""
        return cls(board_generator.initial_board())

    @classmethod
    def empty(cls) -> "State":
        """Generate empty state."""
        return cls(board_generator.empty_board())

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "State":
        """Generate State from state Tensor."""
        return cls(tensor)

    def set_state(self, tensor: torch.Tensor) -> None:
        """Set state Tensor."""
        self._tensor = tensor

    def get_state(self) -> torch.Tensor:
        """Return state Tensor for training."""
        return self._tensor.clone()

    def check_move(self, x1, y1, x2, y2) -> bool:
        """Check whether move is valid."""
        return not (
            self._board_size[0] >= x1 >= 0
            and self._board_size[1] >= y1 >= 0
            and self._board_size[0] >= x2 >= 0
            and self._board_size[1] >= y2 >= 0
            and bool(self._tensor[:, x2, y2].sum(axis=0))
            and y2 < int(board_generator.board_size()[1] / 2)
        )

    def make_move(self, x1, y1, x2, y2) -> None:
        """Make move corresponding to index."""
        capture = torch.where(
            self._tensor[:, x2, y2] != 0, board_generator.layer_values(), 0
        )
        self._captured_value += capture.sum()
        self._tensor[:, x2, y2] = self._tensor[:, x1, y1]
        self._tensor[:, x1, y1] = 0

    def captured_value(self) -> float:
        """Return current capture valuation."""
        return self._captured_value

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
