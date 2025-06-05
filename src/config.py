from typing import TypedDict, ClassVar
from dynaconf import Dynaconf, Validator

from src.agents.base import agent_class_dict


class MPIConfig(TypedDict):
    """MPI Config."""

    TRAINING_WRANK: int
    INFERENCE_NRANK: int
    OBSERVER_ORANK: int
    MPI_SHUTDOWN_WAIT: float


class OBSConfig(TypedDict):
    """Observer, Agents config."""

    GUI: bool
    agent1: str
    agent2: str


class CNNConfig(TypedDict):
    """CNN batch sizes, and others."""

    leaf_batch_num: int
    episode_batch_num: int


class Settings(Dynaconf):
    """Specify data types for Dynaconf settings."""

    mpi: MPIConfig
    obs: OBSConfig
    cnn: CNNConfig

    class Config:
        """Configuration for Dynaconf."""

        envvar_prefix = "DYNACONF"
        settings_files: ClassVar = ["config.toml"]
        namespaces: ClassVar = ["mpi", "obs", "cnn"]

        VALIDATORS: ClassVar = [
            Validator("mpi.TRAINING_WRANK", must_exist=True, gt=-1),
            Validator("mpi.INFERENCE_NRANK", must_exist=True, gt=-1),
            Validator("mpi.OBSERVER_ORANK", must_exist=True, gt=-1),
            Validator("mpi.MPI_SHUTDOWN_WAIT", must_exist=True, gt=0),
            Validator("obs.agent1", must_exist=True),
            Validator("obs.agent2", must_exist=True),
            Validator("obs.GUI", default=True),
        ]


_config: Settings | None = None


def get_config() -> Settings:
    """Generate and return the configuration."""
    global _config
    if _config is None:
        _config = Settings()
        _config.agent_class_dict = agent_class_dict()
        _config.validators.validate(_config)
    return _config
