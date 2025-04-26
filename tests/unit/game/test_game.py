from game.state import State
from testlib import TEST_CASES, IDS
import torch
import pytest


@pytest.mark.parametrize(
    "name,start_board,end_board,x1,y1,x2,y2,validity", TEST_CASES, ids=IDS
)
def test_move_validity(
    name: str,
    start_board: State,
    end_board: State,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    validity: bool,
):
    """Run batch tests for move in states."""
    start_board.make_move(x1, y1, x2, y2)
    assert torch.allclose(start_board.state, end_board.state) is validity
