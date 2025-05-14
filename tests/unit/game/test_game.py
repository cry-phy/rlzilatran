from game.state import State
from game.game import Game
from game.utils.translator import MoveFlag, Translator
from testlib import FIXTURES, TEST_CASES, IDS
import torch
import pytest


def test_move_mask():
    """Test in move mask is generated correctly."""
    reference = torch.load(FIXTURES + "game_fixtures/init_move_targets.pt")
    game = Game.new_game()
    mask = game.move_mask()
    generated = Translator.move_from_index(torch.where(mask)[2])
    assert torch.allclose(generated[0], reference[0])
    assert torch.allclose(generated[1], reference[1])


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
    start_board.make_move(x1, y1, x2, y2, MoveFlag.STANDARD)
    assert torch.allclose(start_board.state, end_board.state) is validity
