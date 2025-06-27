import torch
from torch.utils.data import DataLoader
from typing import Iterator
import logging

from cnn.base_cnn import CNN
from game.utils.constants import Constants

logger = logging.getLogger(__name__)


class Inferrer:
    """Handles running inference on a CNN model and chunking its outputs."""

    def __init__(self, cnn_model: CNN, device: torch.device):
        """Initialize the Inferrer."""
        self.cnn = cnn_model
        self.device = device
        self.cnn.eval()

    def infer_and_chunk(
        self, dataloader: DataLoader
    ) -> Iterator[tuple[int, int, torch.Tensor, torch.Tensor]]:
        """Run inference over a DataLoader, processes outputs, and yields chunked results."""
        self.cnn.eval()

        with torch.no_grad():
            for data, tokens, sources in dataloader:
                logger.debug("Running CNN inference on batch.")
                data = data.to(self.device, non_blocking=True)
                moves_super, values_super = self.cnn(data)
                moves_chunks = torch.split(
                    moves_super.to(device="cpu"), Constants.BATCH_SIZE
                )
                values_chunks = torch.split(
                    values_super.to(device="cpu"), Constants.BATCH_SIZE
                )
                sources_chunked = [
                    sources[i] for i in range(0, len(sources), Constants.BATCH_SIZE)
                ]
                tokens_chunked = [
                    tokens[i] for i in range(0, len(tokens), Constants.BATCH_SIZE)
                ]
                yield from zip(
                    sources_chunked,
                    tokens_chunked,
                    moves_chunks,
                    values_chunks,
                    strict=True,
                )
