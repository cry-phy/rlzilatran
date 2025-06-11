from dynaconf import Dynaconf
from pydantic import BaseModel, Field
from agents import agent_class_dict


class MPIConfig(BaseModel):
    """MPI Config using Pydantic for validation and typing."""

    TRAINING_WRANK: int = Field(0, ge=0)
    INFERENCE_NRANK: int = Field(0, ge=0)
    OBSERVER_ORANK: int = Field(0, ge=0)
    MPI_SHUTDOWN_WAIT: float = Field(10.0, gt=0)
    AGENTS_PER_OBSERVER: int = Field(4, gt=0)
    PROCESSES_PER_NODE: int = Field(..., gt=0)


class OBSConfig(BaseModel):
    """Observer and Agents config using Pydantic."""

    GUI: bool = False
    agent1: str
    agent2: str


class CNNConfig(BaseModel):
    """CNN config using Pydantic."""

    leaf_batch_num: int = 16
    episode_batch_num: int = 16
    replay_buffer_size: int = 500000


class Settings(Dynaconf):
    """Dynaconf settings class using Pydantic models for data validation."""

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
    return _config
