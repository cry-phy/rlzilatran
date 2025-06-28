from dynaconf import Dynaconf
from pydantic import BaseModel
from agents import agent_class_dict


class OBSConfig(BaseModel):
    """Observer and Agents."""

    GUI: bool = False
    agent1: str
    agent2: str


class CNNConfig(BaseModel):
    """CNN config using Pydantic."""

    inference_timing: float = 1
    super_batch_size: int = 8
    leaf_batch_num: int = 16
    episode_batch_num: int = 16
    replay_buffer_size: int = 500000


class Settings(Dynaconf):
    """Dynaconf settings class."""

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
