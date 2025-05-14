from game.utils import Board
from testlib import FIXTURES
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
    assert (
        torch.allclose(
            torch.load(FIXTURES + "boards/init_board.pt").to(torch.int32),
            Board.initial_board(),
        )
        is True
    )


def test_indexing():
    """Test indexing from move to move_tensor."""
    x, y = torch.meshgrid(torch.arange(4), torch.arange(8))
    indices = Board.index_from_move(x, y)
    x2, y2, flag = Board.move_from_index(indices)
    assert torch.allclose(x, x2)
    assert torch.allclose(y, y2)
