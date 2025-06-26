import atexit
from mpi4py import MPI

_mpi_module_initialized = False


def safe_process_init(required=MPI.THREAD_FUNNELED):
    """Safely initialize MPI with controlled thread level."""
    global _mpi_module_initialized
    if not MPI.Is_initialized():
        provided = MPI.Init_thread(required)
        if provided < required:
            raise RuntimeError(
                f"MPI thread level insufficient: got {provided}, need {required}"
            )
        atexit.register(MPI.Finalize)
        print(f"[MPI] Initialized with thread support level {provided}")
    else:
        print("[MPI] MPI already initialized")
    _mpi_module_initialized = True
