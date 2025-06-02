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
            data = [item.data for item in super_batch_items]
            tokens = [item.token for item in super_batch_items]
            source = [item.source for item in super_batch_items]
            yield from zip(data, tokens, source, strict=True)

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
            values = [item.value for item in super_batch_items]
            moves = [item.moves for item in super_batch_items]
            yield from zip(data, moves, values, strict=True)


def collate_leaf(
    super_batch: list[tuple[torch.Tensor, int, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert and concatenate leaf batches into super batch."""
    data_list = [batch[0] for batch in super_batch]
    tokens_list = [torch.full((batch[0].shape[0],), batch[1]) for batch in super_batch]
    source_list = [torch.full((batch[0].shape[0],), batch[2]) for batch in super_batch]
    data = torch.cat(data_list, dim=0)
    tokens = torch.cat(tokens_list, dim=0)
    source = torch.cat(source_list, dim=0)
    return data, tokens, source


def collate_episode(
    super_batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
