from timeit import timeit
import torch
import sys
sys.path.append("../src")
from src.game import Game

game = Game.new_game()

valid_moves = game.move_mask().nonzero()
flat_index = valid_moves[0][-1].item()  # Just an example valid index
shape = game.move_mask().shape

# Correct: wrap index as tensor
x1, y1, move_index = torch.unravel_index(torch.tensor(flat_index), shape)
x2, y2, flag = game.move_from_index(move_index)

# Time game.copy()
copy_time = timeit('game.copy()', globals=globals(), number=1000)
print(f"Game.copy() avg time: {copy_time / 1000:.6f} sec")

# Time game.make_move()
def make_move_test():
    g = game.copy()  # simulate what happens in your _new_node()
    g.make_move(x1, y1, x2, y2, flag)

move_time = timeit(make_move_test, number=1000)
print(f"Game.make_move() avg time (including copy): {move_time / 1000:.6f} sec")
