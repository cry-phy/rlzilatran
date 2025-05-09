import logging
import sys

from mpi.handler import MPIHandler
from mpi.processes import get_process


def setup_logging(rank: int = 0):
    """Set root logger.

    Configure the root logger to write to stdout,
    embedding a rank placeholder for future MPI use.
    """
    fmt = "%(asctime)s [rank:%(rank)d] %(levelname)s %(name)s: %(message)s"
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
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
    logger = logging.getLogger(__name__)
    logger.info("Starting application")

    MPIHandler()

    process = get_process()
    process.run()


if __name__ == "__main__":
    main()
