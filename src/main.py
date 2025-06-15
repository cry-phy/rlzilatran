import logging
import sys
import click
from mpi4py import MPI

from config import Settings, get_config
from mpi import MPIHandler
from processes import get_process


def setup_logging(rank: int, level: int):
    """Set root logger.

    Configure the root logger to write to stdout via mpi buffer.
    The mpi buffer makes this truly usable since the buffer is block buffered
    instead of line buffered allowing fast writes.
    """
    fmt = "%(asctime)s [rank:%(rank)d] %(levelname)s %(name)s: %(message)s"
    root = logging.getLogger()
    root.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter(fmt)
    handler.setFormatter(formatter)
    root.handlers.clear()
    root.addHandler(handler)
    old_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.rank = rank
        return record

    logging.setLogRecordFactory(record_factory)


def prepare() -> tuple[logging.Logger, Settings]:
    """Prepare for execution."""
    wcomm = MPI.COMM_WORLD
    wrank = wcomm.Get_rank()
    setup_logging(wrank, logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Logging initialized.")
    config = get_config()
    return logger, config


def run(logger: logging.Logger, config: Settings, handler: MPIHandler) -> None:
    """Centralized execution."""
    process = get_process(config, handler)
    logger.info("Processes initialized. Running ...")
    process.run()


@click.group()
def cli() -> None:
    """Entry point."""
    pass


@cli.command()
@click.option("-ng", "--num_gpus", default=1, type=int)
@click.option("-apo", "--agents_per_observer", default=4, type=int)
def local(num_gpus, agents_per_observer):
    """Generate MPIHandler for local run."""
    logger, config = prepare()
    handler = MPIHandler.local(num_gpus, agents_per_observer)
    run(logger, config, handler)


@cli.command()
@click.option("--process", default=None, type=str)
@click.option("-apo", "--agents_per_observer", default=1, type=int)
def slurm(process, agents_per_observer):
    """Generate MPI Handler by process type."""
    logger, config = prepare()
    handler = MPIHandler.slurm(process, agents_per_observer)
    run(logger, config, handler)


@cli.command()
@click.option("-ng", "--num_gpus", default=1, type=int)
@click.option("-apo", "--agents_per_observer", default=1, type=int)
def slurm_test(num_gpus, agents_per_observer):
    """Generate MPI Handler emulating slurm usage on local machine."""
    logger, config = prepare()
    handler = MPIHandler.slurm_test(num_gpus, agents_per_observer)
    run(logger, config, handler)


if __name__ == "__main__":
    cli()
