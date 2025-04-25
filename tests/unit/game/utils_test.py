from game.utils import Board
import torch


def test_empty_board_size():
    """Test whether board is correct sizes."""
    size = Board.board_size
    layers = Board.num_layers
    board = Board.empty_board()
    assert (layers, *size) == board.size()


def test_init_board_size():
    """Test whether oard is correct sizes."""
    size = Board.board_size
    layers = Board.num_layers
    board = Board.initial_board()
    assert (layers, *size) == board.size()


def test_init_board():
    """Test whether initial board is set correctly."""
    print(torch.load("tests/fixtures/boards/init_board.pt"))
    print(Board.initial_board())
    assert (
        torch.allclose(
            torch.load("tests/fixtures/boards/init_board.pt"),
            Board.initial_board(),
        )
        is True
    )
