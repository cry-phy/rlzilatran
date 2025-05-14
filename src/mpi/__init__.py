"""Module encapsulating MPI operations.

This contains the MPIHandler Singleton in charge of generating the corresponding MPI Communicators
and especially encapsulate send and recieve operations.


"""

from .handler import MPIHandler, gracefull_error

__all__ = [
    "MPIHandler",
    "gracefull_error",
]
