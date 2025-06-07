import logging
import sys
from mpi4py import MPI

from config import get_config
from mpi import MPIHandler
from processes import get_process


def setup_logging(rank: int, level: int):
    """Set root logger.

    Configure the root logger to write to stdout via mpi buffer.
    The mpi buffer makes this truly usable since the buffer is block buffered
    instead of line buffered allowing fast writes.
    """
    print(f"Setting up logging ({level}) on Rank: {rank}")
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


def main():
    """Run central loop."""
    wcomm = MPI.COMM_WORLD
    wrank = wcomm.Get_rank()
    setup_logging(wrank, logging.DEBUG)
    logger = logging.getLogger(__name__)
    logger.info("Logging initialized.")

    config = get_config()
    handler = MPIHandler(config)
    logger.info("MPI initialized.")

    process = get_process(config, handler)
    logger.info("Processes initialized. Running ...")

    process.run()


if __name__ == "__main__":
    main()
