from agents.human import HumanAgent
import logging
import sys

from observer.ui import UIObserver


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
    rank = 0
    setup_logging(rank)

    logger = logging.getLogger(__name__)
    logger.info("Starting application")

    observer = UIObserver.any_vs_any(HumanAgent, HumanAgent, (0, 1))
    observer.run()


if __name__ == "__main__":
    main()
