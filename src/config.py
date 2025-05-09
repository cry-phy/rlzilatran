from dataclasses import dataclass, is_dataclass, fields
import json
from typing import Any

from game.utils import Board, __BoardMaster


def _recursive_init(cls, data: dict) -> Any:
    """Recursively instanciate configs from json format.

    Given a dataclass "cls" and a dict "data", builds an instance of "cls",
    recursively instantiating any sub-dataclass fields.
    """
    init_kwargs = {}
    for f in fields(cls):
        val = data.get(f.name)
        if is_dataclass(f.type) and isinstance(val, dict):
            init_kwargs[f.name] = _recursive_init(f.type, val)
        else:
            init_kwargs[f.name] = val
    return cls(**init_kwargs)


class SingletonMeta(type):
    """Metaclass making all config classes singletons."""

    _instances: dict[type, object] = {}

    def __call__(cls, *args, **kwargs):
        """Keep track of instances of configs."""
        if cls not in SingletonMeta._instances:
            SingletonMeta._instances[cls] = super().__call__(*args, **kwargs)
        return SingletonMeta._instances[cls]


@dataclass(frozen=True)
class MPIConfig(metaclass=SingletonMeta):
    """Configuration of MPI and ranks."""

    TRAINING_WRANK: int = 0
    INFERENCE_NRANK: int = 0
    OBSERVER_ORANK: int = 0


@dataclass(frozen=True)
class GameConfig:
    """Configuration regarding game and rules."""

    board: __BoardMaster = Board


@dataclass(frozen=True)
class Config(metaclass=SingletonMeta):
    """Reads json config file, top level config."""

    mpi: MPIConfig
    game: GameConfig

    @classmethod
    def load(cls, path: str) -> "Config":
        """Load JSON from `path` and construct nested config singletons automatically."""
        with open(path) as f:
            raw = json.load(f)
        return _recursive_init(cls, raw)
