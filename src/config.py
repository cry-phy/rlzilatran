from dynaconf import Dynaconf

from agents import agent_class_dict


CONFIG_PATH = "config.json"

settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=["config.toml"],
)

_config = None


def get_config():
    """Ensure config is only generated once."""
    global _config
    if _config is None:
        _config = Dynaconf(
            envvar_prefix="DYNACONF",
            settings_files=["config.toml"],
        )
    _config.agent_class_dict = agent_class_dict()
    return _config
