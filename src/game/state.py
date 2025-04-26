from game.utils import Board
import torch
import json
import logging

logger: logging.Logger = logging.getLogger(__name__)


class State:
    """Wrapper around game state pytorch tensor.

    Contains:
        - Pieces State
        - Game End Logic
        - Reward
    """

    def __init__(
        self,
        tensor: torch.Tensor,
        player: bool = True,
        captured_value: float = 0,
        since_capture: int = 0,
        since_capture_call: int = 0,
    ) -> None:
        """Generate State instance.

        Should not be used directly! Use factory Methods instead.
        """
        self._tensor: torch.IntTensor = tensor
        self._player: bool = player
        self._captured_value: float = captured_value
        self._since_capture: int = since_capture
        self._since_capture_call: int = since_capture_call
        self._board_size: tuple[int, int] = Board.board_size

    @property
    def tensor_reference(self) -> torch.Tensor:
        """Provide access to base."""
        return self._tensor

    @property
    def player(self) -> bool:
        """Current player."""
        return self._player

    @property
    def captured_value(self) -> float:
        """Value of captured pieces by player False."""
        return self._captured_value

    @property
    def since_capture(self) -> int:
        """Moves since last capture."""
        return self._since_capture

    @property
    def since_capture_call(self) -> int:
        """Moves since last capture."""
        return self._since_capture_call

    @property
    def state(self) -> torch.Tensor:
        """Return state Tensor for training."""
        return self._tensor.clone()

    @player.setter
    def player(self, player: bool) -> None:
        """Set current player."""
        self._player = player

    @captured_value.setter
    def captured_value(self, captured_value: float) -> None:
        """Value of captured pieces by player False."""
        self._captured_value = captured_value

    @since_capture.setter
    def since_capture(self, since_capture: int) -> None:
        """Set moves since last capture."""
        self._since_capture = since_capture

    @since_capture_call.setter
    def since_capture_call(self, since_capture_call: int) -> None:
        """Set moves since last capture."""
        self._since_capture_call = since_capture_call

    @state.setter
    def state(self, tensor: torch.Tensor) -> None:
        """Set state Tensor."""
        self._tensor = tensor

    @classmethod
    def initial_state(cls) -> "State":
        """Generate Initial game State."""
        return cls(Board.initial_board())

    @classmethod
    def empty(cls) -> "State":
        """Generate empty state."""
        return cls(Board.empty_board())

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "State":
        """Generate State from state Tensor."""
        return cls(tensor)

    @classmethod
    def from_state(cls, state: "State") -> "State":
        """Create copy from state for copy operations."""
        return cls(state.state, state.player, state.captured_value, state.since_capture)

    # --- Dunder Methods ---
    def __eq__(self, other: "State") -> bool:
        """Compare states to each other."""
        return (
            torch.allclose(self._tensor, other.tensor_reference)
            and self._player == other.player
            and self._captured_value == other.captured_value
            and self._since_capture == other.since_capture
            and self._since_capture_call == other.since_capture_call
        )

    # --- File IO ---
    def state_dict(self) -> dict:
        """Return a plain dict of only the data fields."""
        return {
            "tensor": self._tensor,
            "player": self._player,
            "captured_value": self._captured_value,
            "since_capture": self._since_capture,
            "since_capture_call": self._since_capture_call,
            "board_size": self._board_size,
        }

    def save(self, path: str) -> None:
        """Save only the data to a JSON file at `filepath`."""
        data = self.state_dict()
        data["tensor"] = data["tensor"].tolist()
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_file(cls, path: str) -> "State":
        """Load data from JSON and rebuild a GameState instance."""
        with open(path) as f:
            data = json.load(f)
        tensor = torch.tensor(
            data["tensor"],
            dtype=torch.int32,
        )
        instance = cls(
            tensor=tensor,
            player=data["player"],
            captured_value=data["captured_value"],
            since_capture=data["since_capture"],
            since_capture_call=data["since_capture_call"],
        )
        return instance

    # --- Simple operations ---
    def copy(self) -> "State":
        """Generate copy of self."""
        return self.from_state(self)

    def reset(self) -> None:
        """Set to initial state."""
        self._tensor = Board.initial_board()

    def square_filled(self, x: int, y: int) -> bool:
        """Return if th square has pieces on it."""
        return bool(self._tensor[:, x, y].sum(axis=0))

    def get_layer(self, x: int, y: int) -> torch.Tensor:
        """Get layers with pieces on square."""
        return torch.where(self._tensor[:, x, y])[0]

    def get_player_mask(self, player: bool) -> torch.BoolTensor:
        """Get mask of player."""
        return (
            self._tensor[
                int(player) * Board.num_pieces : (int(player) + 1) * Board.num_pieces,
                :,
                :,
            ]
            .sum(dim=0)
            .to(bool)
        )

    # --- Game Rules ---
    def check_piece_move(self, x1: int, y1: int, x2: int, y2: int) -> bool:
        """Check whether attempted move is in valid moves for piece."""
        target = torch.IntTensor([x2, y2])
        piece = Board.piece_from_layer(self.get_layer(x1, y1))
        moves = Board.moves_from_piece(piece)(x1, y1, self)
        return torch.any(torch.all(moves == target, dim=1))

    def check_move(self, x1: int, y1: int, x2: int, y2: int) -> bool:
        """Check whether move is valid."""
        return (
            Board.valid_position(x1, y1)
            and Board.valid_position(x2, y2)
            and Board.player_half(y1, self._player)
            and self.check_piece_move(x1, y1, x2, y2)
        )

    def check_half_empty(self) -> bool:
        """Termination state this half of the board is empty."""
        return not torch.any(
            self._tensor[
                :,
                :,
                int(self._player) * Board.num_pieces : (1 + int(self._player))
                * Board.num_pieces,
            ].to(bool)
        )

    # --- Board Manipulations ---
    def make_move(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """Make move corresponding to index."""
        capture_layer: torch.Tensor = torch.where(self._tensor[:, x2, y2])[0]
        if capture_layer.nelement() != 0:
            capture_piece = Board.piece_from_layer(capture_layer)
            capture_value = Board.value_from_piece(capture_piece)
            self._since_capture += 1
            self._captured_value += capture_value
            logger.debug("Capturing")
        # Benchmarked this: Faster than setting with add/remove_piece and where. TODO: Retest after additional logic.
        i1, i2 = Board.half_indices(y1)
        j1, j2 = Board.half_indices(y2)
        logger.debug(
            "Moving %s to %s",
            str(self._tensor[i1:i2, x1, y1]),
            str(self._tensor[j1:j2, x2, y2]),
        )
        self._tensor[j1:j2, x2, y2] = self._tensor[i1:i2, x1, y1]
        self._tensor[i1:i2, x1, y1] = 0
        self._player = not self._player

    def add_piece(self, x: int, y: int, piece: str) -> None:
        """Add Piece to certain layer."""
        layer: torch.Tensor = Board.layer_from_piece(piece)
        i1, _ = Board.half_indices(y)
        self._tensor[i1 + layer, x, y] = 1

    def remove_piece(self, x: int, y: int, piece: str) -> None:
        """Remove certain piece."""
        layer = Board.layer_from_piece(piece)
        self._tensor[layer, x, y] = 0
        self._tensor[layer + Board.num_pieces, x, y] = 0

    def clear_square(self, x: int, y: int) -> None:
        """Clear square entirely."""
        self._tensor[:, x, y] = 0

    # -- Retrieve Channels --
    def player_channel(self) -> torch.Tensor:
        """Generate entire layer of player values."""
        return torch.full(Board.board_size, int(self._player))

    def captured_value_channel(self) -> torch.Tensor:
        """Generate entire layer of player values."""
        return torch.full(Board.board_size, int(self._captured_value))

    def since_capture_channel(self) -> torch.Tensor:
        """Generate entire layer of player values."""
        return torch.full(Board.board_size, int(self._since_capture))

    def since_capture_call_channel(self) -> torch.Tensor:
        """Generate entire layer of player values."""
        return torch.full(Board.board_size, int(self._since_capture_call))

    def get_additional_channels(self) -> list[torch.Tensor]:
        """Get list of additional channels."""
        return [
            self.player_channel(),
            self.captured_value_channel(),
            self.since_capture_channel(),
            self.since_capture_call_channel(),
        ]
