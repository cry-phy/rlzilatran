from game.utils import board_generator
import torch


def test_empty_board_size():
    """Test whether oard is correct sizes."""
    size = board_generator.board_size()
    layers = board_generator.num_layers()
    board = board_generator.empty_board()
    assert (layers, *size) == board.size()


def test_init_board_size():
    """Test whether oard is correct sizes."""
    size = board_generator.board_size()
    layers = board_generator.num_layers()
    board = board_generator.initial_board()
    assert (layers, *size) == board.size()


def test_init_board():
    """Test whether initial board is set correctly."""
    print(torch.load("tests/fixtures/init_board.pt"))
    print(board_generator.initial_board())
    assert (
        torch.allclose(
            torch.load("tests/fixtures/init_board.pt"),
            board_generator.initial_board(),
        )
        is True
    )
