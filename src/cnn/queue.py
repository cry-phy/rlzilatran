from dataclasses import dataclass
from typing import Generic, Iterable, TypeVar
from mpi4py.MPI import Request
from collections import deque
import torch

Item = TypeVar("Item", bound="BaseItem")


@dataclass
class BaseItem:
    """Basic Item of the Data queue."""

    request: Request
    data: torch.Tensor


@dataclass
class LeafItem(BaseItem):
    """Leaf Item."""

    source: int
    token: int


@dataclass
class EpisodeItem(BaseItem):
    """Leaf Item."""

    value: torch.Tensor
    moves: torch.Tensor


class Queue(Generic[Item]):
    """Wrapper around Deque for only retrieving fully recived messages."""

    def __init__(self, batch_size: int) -> None:
        """Generate an empty Queue object."""
        self._batch_size: int = batch_size
        self._data: deque[Item] = deque()

    def gen_readiness(self) -> torch.Tensor:
        """Generate the readiness of the requests."""
        return torch.tensor([item.request.Get_status() for item in self._data])

    def add(self, items: Iterable[BaseItem]) -> None:
        """Add to deque."""
        for item in items:
            self._data.appendleft(item)

    def __len__(self) -> int:
        """Pass underlying deq length."""
        return self._data.__len__()

    def num_ready(self) -> int:
        """Determine number of currently resolved requests, generates the corresponding mask."""
        return self.gen_readiness.sum().item()

    def retrieve_ready(self) -> list[Item]:
        """Retrieve and remove ready items."""
        ready_mask = self.gen_readiness()
        if not torch.any(ready_mask):
            return []
        ready_indices = torch.arange(len(self))[ready_mask]
        batch = [self._data[i] for i in ready_indices]
        for i in ready_indices:
            del self._data[i]
        return batch
