import torch

BOARD_DIMENSIONS = (7, 4, 8)
PRIVATE_LAYER = 2
PRIVATE_POSITIONS = [
    (2, 1),
    (1, 2),
    (2, 2),
    (-3, -2),
    (-2, -3),
    (-3, -3),
]
SERGEANT_LAYER = 1
SERGEANT_POSITIONS = [
    (0, 2),
    (1, 1),
    (2, 0),
    (-1, -3),
    (-2, -2),
    (-3, -1),
]
OFFICER_LAYER = 0
OFFICER_POSITIONS = [
    (0, 0),
    (0, 1),
    (1, 0),
    (-1, -1),
    (-1, -2),
    (-2, -1),
]


class BoardGenerator(object):
    """Generator for board tensors.

    TODO: Extend to read config Files.
    """

    _instance = None
    _board: torch.Tensor = None

    def __new__(cls) -> "BoardGenerator":
        """Set up Singleton instance of BoardGenerator."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._board = cls._instance.empty_board()
            cls._instance.set_initial_game()
        return cls._instance

    def empty_board(self) -> torch.Tensor:
        """Get empty board."""
        return torch.empty(BOARD_DIMENSIONS)

    def set_privates(self) -> None:
        """Get Private Board."""
        self._board[PRIVATE_LAYER][PRIVATE_POSITIONS] = 1

    def set_sergeants(self) -> None:
        """Get Private Board."""
        self._board[SERGEANT_LAYER][SERGEANT_POSITIONS] = 1

    def set_officers(self) -> None:
        """Get Private Board."""
        self._board[OFFICER_LAYER][OFFICER_POSITIONS] = 1

    def set_initial_game(self) -> None:
        """Set up Initial board."""
        self.set_privates()
        self.set_sergeants()
        self.set_officers()

    def get_board(self) -> torch.Tensor:
        """Return new instance of board."""
        return self._board.clone()
