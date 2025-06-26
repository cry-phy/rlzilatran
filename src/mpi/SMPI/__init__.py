"""Module encapsulating !ALL! MPI operations from mpi4py.MPI.

if your linter does not find a specific export from MPI here:
    - Import the corresponding passed operation explicitly!
    - Add the import to the explicit exports!

This module acts as a direct re-export of mpi4py.MPI, providing its full
functionality under the 'mpi.SMPI' namespace.
"""

import mpi4py.MPI as _original_mpi_module

# --- BEGIN EXPLICIT IMPORTS FOR STATIC ANALYSIS / LINTERS ---
# Add any specific MPI operations (types, constants, functions) here
# that your linter/IDE complains about. This makes them directly visible
# to static analysis tools.

from mpi4py.MPI import (
    Request,  # noqa: F401, W0611
    Status,  # noqa: F401, W0611
    Comm,  # noqa: F401, W0611
    COMM_WORLD,  # noqa: F401, W0611
    ANY_SOURCE,  # noqa: F401, W0611
    SUM,  # noqa: F401, W0611
    Datatype,  # noqa: F401, W0611
    INT,  # noqa: F401, W0611
    BYTE,  # noqa: F401, W0611
    COMM_NULL,  # noqa: F401, W0611
    COMM_TYPE_SHARED,  # noqa: F401, W0611
    UNDEFINED,  # noqa: F401, W0611
    Get_processor_name,  # noqa: F401, W0611
    Finalize,  # noqa: F401, W0611
)

# Create a list of the names that were explicitly imported.
_explicitly_imported_symbols = [
    "Request",
    "Status",
    "Comm",
    "COMM_WORLD",
    "ANY_SOURCE",
    "SUM",
    "Datatype",
    "INT",
    "BYTE",
    "COMM_NULL",
    "COMM_TYPE_SHARED",
    "UNDEFINED",
    "Get_processor_name",
    "Finalize",
]

# --- END EXPLICIT IMPORTS FOR STATIC ANALYSIS / LINTERS ---


_reexported_symbols = []
for name in dir(_original_mpi_module):
    if not name.startswith("_"):
        try:
            # This line already ensures the symbols are available at runtime
            globals()[name] = getattr(_original_mpi_module, name)
            _reexported_symbols.append(name)
        except AttributeError:
            pass

# Combine dynamically found symbols with explicitly imported ones.
# Using a set ensures uniqueness, and sorted makes the list consistent.
__all__ = sorted(set(_reexported_symbols + _explicitly_imported_symbols))
__doc__ = _original_mpi_module.__doc__
