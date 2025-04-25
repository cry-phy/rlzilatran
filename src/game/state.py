from src.game.utils import board_generator
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
        self._prev_user_move: torch.Tensor = tensor
        self._captured_value: float = 0
        self._since_capture: int = 0
        self._board_size: tuple[int, int] = board_generator.board_size()

    @property
    def tensor_reference(self) -> torch.Tensor:
        """Provide access to base."""
        return self._tensor

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

    @classmethod
    def from_file(cls, path: str) -> "State":
        """Load directly from a file."""
        instance = cls(board_generator.empty_board())
        instance.load(path)
        return instance

    def set_state(self, tensor: torch.Tensor) -> None:
        """Set state Tensor."""
        self._tensor = tensor

    def get_state(self) -> torch.Tensor:
        """Return state Tensor for training."""
        return self._tensor.clone()

    def square_filled(self, x: int, y: int) -> bool:
        """Return if th square has pieces on it."""
        return bool(self._tensor[:, x, y].sum(axis=0))

    def repeat_move(self) -> bool:
        """Check whether move was made right before."""
        return torch.allclose(self._tensor, self._prev_user_move)

    def check_move(self, x1: int, y1: int, x2: int, y2: int) -> bool:
        """Check whether move is valid."""
        return not (
            self._board_size[0] >= x1 >= 0
            and self._board_size[1] >= y1 >= 0
            and self._board_size[0] >= x2 >= 0
            and self._board_size[1] >= y2 >= 0
            and self.square_filled(x2, y2)
            and board_generator.player_half(y2)
        )

    def make_move(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """Make move corresponding to index."""
        capture = torch.where(
            self._tensor[:, x2, y2] != 0, board_generator.layer_values(), 0
        )
        capture_value = capture.sum()
        if capture_value != 0:
            self._since_capture += 1
            self._captured_value += capture_value
        # Benchmarked this: Faster than setting with add/remove_piece and where.
        self._tensor[:, x2, y2] = self._tensor[:, x1, y1]
        self._tensor[:, x1, y1] = 0

    def add_piece(self, x: int, y: int, piece: str) -> None:
        """Add Piece to certain layer."""
        layer = board_generator.piece_layer_dict()[piece]
        if not board_generator.player_half(y):
            layer = -(layer + 1)
        print(x, y, layer)
        self._tensor[layer, x, y] = 1

    def remove_piece(self, x: int, y: int, piece: str) -> None:
        """Remove certain piece."""
        layer = board_generator.piece_layer_dict()[piece]
        if not board_generator.player_half(y):
            layer = -(layer + 1)
        print(x, y, layer)
        self._tensor[layer, x, y] = 0

    def clear_square(self, x: int, y: int) -> None:
        """Clear square entirely."""
        self._tensor[:, x, y] = 0

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

    def __eq__(self, other: "State") -> bool:
        """Compare states to each other."""
        return (
            torch.allclose(self._tensor, other.tensor_reference())
            and self._captured_value == other.captured_value()
        )
