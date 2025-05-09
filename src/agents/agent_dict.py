import importlib
import pkgutil
import agents
from agents.base import BaseAgent


def __import_all_modules(package):
    """Import all submodules in certain module."""
    for _, module_name, _ in pkgutil.iter_modules(
        package.__path__, package.__name__ + "."
    ):
        importlib.import_module(module_name)


def __get_all_subclasses(base_class):
    """Glob all subclasses of a base class in the namespace."""
    subclasses = set()

    def recurse(cls):
        for subclass in cls.__subclasses__():
            subclasses.add(subclass)
            recurse(subclass)

    recurse(base_class)
    return subclasses


def agent_class_dict() -> dict[str, type[BaseAgent]]:
    """Build full dictionary of agent classes."""
    __import_all_modules(agents)
    subclasses = __get_all_subclasses(BaseAgent)
    return {cls.__name__: cls for cls in subclasses}
