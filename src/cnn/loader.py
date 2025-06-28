from typing import Iterator
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
from cnn.queue import EpisodeItem, LeafItem, Queue, ReplayBuffer
import logging

logger: logging.Logger = logging.getLogger(__name__)


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
        super_batch_items = self.queue.poll()
        yield from super_batch_items

    def retrieve_identifiers(self) -> tuple[list[int], list[int]]:
        """Return list of tokens and sources respectively."""
        return self.tokens, self.source


class EpisodeDataset(IterableDataset):
    """Dataset for Episode loader."""

    def __init__(self, queue: Queue, buffer_size: int) -> None:
        """Generate dataset from Queue."""
        self.queue: Queue[EpisodeItem] = queue
        self.replay_buffer: ReplayBuffer[EpisodeItem] = ReplayBuffer(buffer_size)

    def __iter__(self) -> Iterator:
        """Generate Iterator for frozen super batch."""
        super_batch_items = self.queue.poll()
        self.replay_buffer.add(super_batch_items)
        logger.info("Using %i recorded batches.", len(self.replay_buffer))
        yield from self.replay_buffer

    def replay_buffer_len(self) -> int:
        """Return Number of entries in replay buffer."""
        return len(self.replay_buffer)


def collate_leaf(
    super_batch: list[LeafItem],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert and concatenate leaf batches into super batch."""
    data_list = []
    tokens_list = []
    source_list = []
    for item in super_batch:
        data_list.append(item.data)
        tokens_list.append(torch.full((item.data.shape[0],), int(item.token)))
        source_list.append(torch.full((item.data.shape[0],), int(item.source)))
    data = torch.concatenate(data_list, dim=0)
    tokens = torch.concatenate(tokens_list, dim=0)
    source = torch.concatenate(source_list, dim=0)
    return data, tokens, source


def collate_episode(
    super_batch: list[EpisodeItem],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate data, moves and values."""
    data_list = []
    moves_list = []
    values_list = []
    for item in super_batch:
        data_list.append(item.data)
        moves_list.append(item.moves)
        values_list.append(item.value)
    data = torch.concatenate(data_list, dim=0)
    moves = torch.concatenate(moves_list, dim=0)
    values = torch.concatenate(values_list, dim=0)
    return data, moves, values


def get_leaf_loader(queue: Queue, super_size: int) -> tuple[Dataset, DataLoader]:
    """Unified way to generate dataloader for leaves."""
    dataset = LeafDataset(queue)
    loader = DataLoader(
        dataset,
        batch_size=super_size,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_leaf,
    )
    return dataset, loader


def get_episode_loader(
    queue: Queue,
    super_size: int,
    replay_buffer_size: int,
) -> tuple[Dataset, DataLoader]:
    """Unified way to generate dataloader for episodes."""
    dataset = EpisodeDataset(queue, replay_buffer_size)
    loader = DataLoader(
        dataset,
        batch_size=super_size,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_episode,
    )
    return dataset, loader
