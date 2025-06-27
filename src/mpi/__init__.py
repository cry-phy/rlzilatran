"""Module encapsulating MPI operations.

Use only these modules to interact with MPI:
    - MPIHandler: Resource handler for parallelism
    - SMPI: Safe way to interact with the mpi4py.MPI submodule

The usage of MPI should purely be through this module to maintain modularity,
avoid double initialization (e.g., with srun), and facilitate backend changes.

This module contains the MPIHandler Singleton responsible for generating the
corresponding MPI Communicators and encapsulating send and receive operations.
"""
print("from the other side")
# To globally prepare processes before usage of MPI, we need this!
# This must remain the first import in this file!
import mpi4py

mpi4py.rc.initialize = False
mpi4py.rc.finalize = False

from .set_state import safe_process_init  # noqa: E402

safe_process_init()

# Proceed as usual
# These imports are intentionally placed after MPI initialization.
from . import SMPI  # noqa: E402
from .handler import MPIHandler, gracefull_error  # noqa: E402

__all__ = [
    "SMPI",
    "MPIHandler",
    "gracefull_error",
]
