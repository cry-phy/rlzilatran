import torch
from timeit import timeit
from collections import deque
import sys

sys.path.insert(0, "src")

from game.state import State

state = State.initial_state()


class RingFrameStack:
    def __init__(self, K, device=None, dtype=torch.float32):
        """
        Args:
            K (int): number of frames to stack.
            C, H, W (int): channel, height, width of each frame.
        """
        self.K = K
        self.buffer = torch.zeros(
            (K, *state.tensor_reference.shape), device=device, dtype=dtype
        )
        self.ptr = 0  # next write index
        self.num_filled = 0  # how many unique frames filled

    def reset(self, obs: torch.Tensor) -> torch.Tensor:
        """Fill buffer entirely with initial observation."""
        for i in range(self.K):
            self.buffer[i].copy_(obs)  # clone into all slots
        self.ptr = 0
        self.num_filled = self.K
        return self._get_stack()

    def step(self, obs: torch.Tensor) -> torch.Tensor:
        """Overwrite the oldest frame and advance pointer."""
        self.buffer[self.ptr].copy_(obs)  # write new frame
        self.ptr = (self.ptr + 1) % self.K  # wrap-around pointer
        self.num_filled = min(self.K, self.num_filled + 1)
        return self._get_stack()

    def _get_stack(self) -> torch.Tensor:
        """Return frames in temporal order as (C×K×H×W)."""
        if self.num_filled < self.K:
            # if not yet full, pad earliest entries with the first frame
            padded = torch.cat(
                [
                    self.buffer[0:1].repeat(self.K - self.num_filled, 1, 1, 1),
                    self.buffer[0 : self.num_filled],
                ],
                dim=0,
            )
        else:
            # gather in correct order
            idxs = [(self.ptr + i) % self.K for i in range(self.K)]
            padded = self.buffer[idxs]
        # reorder to (C, K, H, W)
        return padded.permute(1, 0, 2, 3).contiguous()  # (C, K, H, W)


class FrameStack:
    def __init__(self, K, device=None, dtype=torch.float32):
        """
        Args:
            K (int): number of frames to stack.
            device (torch.device or str): where to place the stacked tensor.
            dtype: torch dtype for the tensor.
        """
        self.K = K
        self.frames = deque(
            maxlen=K
        )  # auto-drops oldest frame :contentReference[oaicite:3]{index=3}
        self.device = torch.device(device) if device else None
        self.dtype = dtype

    def reset(self, obs: torch.Tensor) -> torch.Tensor:
        """Call this with the first observation after env.reset()."""
        for _ in range(self.K):
            self.frames.append(
                obs.clone()
            )  # fill buffer with initial frame :contentReference[oaicite:4]{index=4}
        return self._get_stack()

    def step(self, obs: torch.Tensor) -> torch.Tensor:
        """Call this with each new observation from env.step()."""
        self.frames.append(obs.clone())  # old frames auto-evicted
        return self._get_stack()

    def _get_stack(self) -> torch.Tensor:
        """Return a (C×K×H×W) tensor on the specified device."""
        stack = torch.stack(
            list(self.frames), dim=1
        )  # shape: (C, K, H, W) :contentReference[oaicite:5]{index=5}
        if self.device:
            stack = stack.to(
                self.device, dtype=self.dtype
            )  # move to GPU if desired :contentReference[oaicite:6]{index=6}
        return stack


if __name__ == "__main__":
    num_push = 1000
    iterations = 1000
    device = "cpu"
    time = timeit(
        setup=f"stack = FrameStack(8, device='{device}')",
        stmt=f"[stack.step(state.tensor_reference) for _ in range({num_push})]",
        number=iterations,
        globals=globals(),
    )
    print(
        f"Performed {iterations} iterations of {num_push} pushes on {device} in {time:.2f}s"
    )
    print(
        f"Thats on average {num_push} push operations in {time / iterations * 1e3:.2f}e-3s"
    )
    print(
        f"Thats on average {time / iterations / num_push * 1e6:.2f}e-6s per iteration"
    )

    device = "cuda"
    time = timeit(
        setup=f"stack = FrameStack(8, device='{device}')",
        stmt=f"[stack.step(state.tensor_reference) for _ in range({num_push})]",
        number=iterations,
        globals=globals(),
    )
    print(
        f"Performed {iterations} iterations of {num_push} pushes on {device} in {time:.2f}s"
    )
    print(
        f"Thats on average {num_push} push operations in {time / iterations * 1e3:.2f}e-3s"
    )
    print(
        f"Thats on average {time / iterations / num_push * 1e6:.2f}e-6s per iteration"
    )

    device = "cpu"
    time = timeit(
        setup=f"stack = RingFrameStack(8, device='{device}')",
        stmt=f"[stack.step(state.tensor_reference) for _ in range({num_push})]",
        number=iterations,
        globals=globals(),
    )
    print(
        f"Performed {iterations} iterations of {num_push} pushes on {device} in {time:.2f}s"
    )
    print(
        f"Thats on average {num_push} push operations in {time / iterations * 1e3:.2f}e-3s"
    )
    print(
        f"Thats on average {time / iterations / num_push * 1e6:.2f}e-6s per iteration"
    )

    num_push = 1000
    iterations = 1000
    device = "cuda"
    time = timeit(
        setup=f"stack = RingFrameStack(8, device='{device}')",
        stmt=f"[stack.step(state.tensor_reference) for _ in range({num_push})]",
        number=iterations,
        globals=globals(),
    )
    print(
        f"Performed {iterations} iterations of {num_push} pushes on {device} in {time:.2f}s"
    )
    print(
        f"Thats on average {num_push} push operations in {time / iterations * 1e3:.2f}e-3s"
    )
    print(
        f"Thats on average {time / iterations / num_push * 1e6:.2f}e-6s per iteration"
    )
