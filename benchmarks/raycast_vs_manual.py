import torch
from typing import List, Tuple
from timeit import timeit


def generate_officer_moves_simple(
    x: int,
    y: int,
    board_size: Tuple[int, int],
    state_tensor: torch.Tensor,
    officer_directions: List[Tuple[int, int]],
) -> torch.IntTensor:
    device = state_tensor.device
    dtype = torch.int64
    W, H = board_size

    moves = []
    own_mask = state_tensor[:3, :, :].sum(dim=0).to(torch.bool)
    other_mask = state_tensor[3:, :, :].sum(dim=0).to(torch.bool)

    for direction in officer_directions:
        dx, dy = direction
        for step in range(1, max(W, H)):
            nx = x + dx * step
            ny = y + dy * step
            if not (0 <= nx < W and 0 <= ny < H):
                break
            pos = (nx, ny)
            if own_mask[pos]:
                break
            moves.append(pos)
            if other_mask[pos]:
                break

    return torch.tensor(moves, dtype=dtype, device=device)


def generate_officer_moves(
    x: int,
    y: int,
    board_size: Tuple[int, int],
    state_tensor: torch.Tensor,
    officer_directions: List[Tuple[int, int]],
) -> torch.IntTensor:
    """Optimized officer (queen-like) moves, stopping at blockers and captures."""
    device = state_tensor.device
    dtype = torch.int64
    W, H = board_size
    origin = torch.tensor([x, y], device=device, dtype=dtype)

    own_mask = state_tensor[:3, :, :].sum(dim=0).to(torch.bool)
    other_mask = state_tensor[3:, :, :].sum(dim=0).to(torch.bool)

    all_moves = []
    for dx, dy in officer_directions:
        max_step_x = (
            (W - 1 - x) // dx if dx > 0 else (x // -dx if dx < 0 else float("inf"))
        )
        max_step_y = (
            (H - 1 - y) // dy if dy > 0 else (y // -dy if dy < 0 else float("inf"))
        )
        max_steps = int(min(max_step_x, max_step_y))
        if max_steps <= 0:
            continue

        steps = torch.arange(1, max_steps + 1, device=device, dtype=dtype)
        dirs = torch.tensor([dx, dy], device=device, dtype=dtype)
        positions = origin + steps.unsqueeze(1) * dirs.unsqueeze(0)

        xs = positions[:, 0].clamp(0, W - 1)
        ys = positions[:, 1].clamp(0, H - 1)

        own_hits = own_mask[xs, ys]
        other_hits = other_mask[xs, ys]

        blocker_indices = torch.nonzero(own_hits, as_tuple=False)
        block_step = (
            blocker_indices[0].item() + 1 if blocker_indices.numel() else max_steps + 1
        )
        capture_indices = torch.nonzero(other_hits, as_tuple=False)
        cap_step = (
            capture_indices[0].item() + 1 if capture_indices.numel() else max_steps + 1
        )

        stop_step = min(block_step, cap_step)
        valid_steps = steps <= stop_step
        valid_positions = positions[valid_steps]
        all_moves.append(valid_positions)

    if not all_moves:
        return torch.empty((0, 2), device=device, dtype=dtype)

    return torch.cat(all_moves, dim=0)


# Benchmarking setup
if __name__ == "__main__":
    device = "cpu"
    iterations = 10000
    x, y = 2, 2
    board_size = (4, 8)
    state_tensor = torch.randint(0, 2, (6, *board_size), dtype=torch.int, device=device)
    directions = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]

    time_optimized = timeit(
        stmt="generate_officer_moves(x, y, board_size, state_tensor, directions)",
        number=iterations,
        globals=globals(),
    )
    print(
        f"Optimized: {iterations} iterations in {time_optimized:.4f}s ({time_optimized / iterations * 1e6:.2f} µs per call)"
    )

    time_simple = timeit(
        stmt="generate_officer_moves_simple(x, y, board_size, state_tensor, directions)",
        number=iterations,
        globals=globals(),
    )
    print(
        f"Simple: {iterations} iterations in {time_simple:.4f}s ({time_simple / iterations * 1e6:.2f} µs per call)"
    )
