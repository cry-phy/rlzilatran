from typing import Iterator
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
from cnn.queue import EpisodeItem, LeafItem, Queue


# non_blocking = True
class LeafDataset(IterableDataset):
    """Dataset for leaf loader."""

    def __init__(self, queue: Queue) -> None:
        """Generate Dataloader from existing Queue."""
        self.queue: Queue[LeafItem] = queue
        self.data: list[torch.Tensor] = []
        self.tokens: list[int] = []
        self.source: list[int] = []

    def __iter__(self) -> Iterator:
        """Generate Iterator for a frozen super batch."""
        super_batch_items = self.queue.retrieve_ready()
        if super_batch_items:
            self.data = [item.data for item in super_batch_items]
            self.tokens = [item.token for item in super_batch_items]
            self.source = [item.source for item in super_batch_items]
            yield from self.data

    def retrieve_identifiers(self) -> tuple[list[int], list[int]]:
        """Return list of tokens and sources respectively."""
        return self.tokens, self.source


class EpisodeDataset(IterableDataset):
    """Dataset for Episode loader."""

    def __init__(self, queue) -> None:
        """Generate dataset from Queue."""
        self.queue: Queue[EpisodeItem] = queue

    def __iter__(self) -> Iterator:
        """Generate Iterator for frozen super batch."""
        super_batch_items = self.queue.retrieve_ready()
        if super_batch_items:
            data = [item.data for item in super_batch_items]
            values = [item.values for item in super_batch_items]
            moves = [item.moves for item in super_batch_items]
            yield from zip(data, moves, values, strict=True)


def collate_leaf(super_batch: list[torch.Tensor]) -> torch.Tensor:
    """Convert and concatenate leaf batches into super batch."""
    return torch.cat(super_batch, dim=0)


def collate_episode(super_batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Collate data, moves and values."""
    data = torch.cat([batch[0] for batch in super_batch], dim=0)
    moves = torch.cat([batch[1] for batch in super_batch], dim=0)
    values = torch.cat([batch[2] for batch in super_batch], dim=0)
    return data, moves, values


def get_leaf_loader(queue: Queue, super_size: int) -> tuple[Dataset, DataLoader]:
    """Unified way to generate dataloader for leaves."""
    dataset = LeafDataset(queue)
    loader = DataLoader(
        dataset,
        batch_size=super_size,
        num_workers=0,
        pin_memory=True,
    )
    return dataset, loader


def get_episode_loader(queue, super_size: int) -> tuple[Dataset, DataLoader]:
    """Unified way to generate dataloader for episodes."""
    dataset = EpisodeDataset(queue)
    loader = DataLoader(
        dataset,
        batch_size=super_size,
        num_workers=0,
        pin_memory=True,
    )
    return dataset, loader
