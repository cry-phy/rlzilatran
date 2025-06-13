import os
from dynaconf import Dynaconf
from pydantic import BaseModel, Field, PrivateAttr
from agents import agent_class_dict


class MPIConfig(BaseModel):
    """MPI Config."""

    TRAINING_WRANK: int = Field(0, ge=0)
    INFERENCE_NRANK: int = Field(0, ge=0)
    OBSERVER_ORANK: int = Field(0, ge=0)
    MPI_SHUTDOWN_WAIT: float = Field(10.0, gt=0)
    AGENTS_PER_OBSERVER: int = Field(4, gt=0)
    PROCESSES_PER_NODE: int = Field(..., gt=0)
    GPU_ENV_VARIABLE: str = Field("config.mpi.GPU_ENV_VARIABLE")

    NUM_GPUS: int = PrivateAttr(default=0)


class OBSConfig(BaseModel):
    """Observer and Agents."""

    GUI: bool = False
    agent1: str
    agent2: str


class CNNConfig(BaseModel):
    """CNN config using Pydantic."""

    leaf_batch_num: int = 16
    episode_batch_num: int = 16
    replay_buffer_size: int = 500000


class Settings(Dynaconf):
    """Dynaconf settings class."""

    mpi: MPIConfig
    obs: OBSConfig
    cnn: CNNConfig


# Global variable for the configuration instance
_config: Settings | None = None


def get_config() -> Settings:
    """Generate and return the configuration."""
    global _config
    if _config is None:
        _config = Settings(
            envvar_prefix="DYNACONF",
            settings_files=["config.toml"],
            environments=False,
        )
        _config.agent_class_dict = agent_class_dict()
        num_gpus = int(os.environ[_config.mpi.GPU_ENV_VARIABLE])
        if num_gpus is not None:
            _config.mpi.NUM_GPUS = num_gpus
        else:
            raise ValueError(
                "Cannot load number of GPUS from %s", _config.mpi.GPU_ENV_VARIABLE
            )
    return _config
