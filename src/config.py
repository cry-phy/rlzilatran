from dynaconf import Dynaconf

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
    return _config
