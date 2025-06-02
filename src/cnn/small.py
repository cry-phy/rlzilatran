from torch import nn


class SmallCNN(nn.Module):
    """Small CNN fof testing Datatloader and procedures."""

    def __init__(self, *args, **kwargs) -> None:
        super(SmallCNN, self).__init__()
