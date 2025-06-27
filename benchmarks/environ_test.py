import os
import socket
import sys
import time
from mpi4py import MPI

# --- MPI Initialization ---
# This is the standard way to start MPI in a Python script
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # The rank of the current process (e.g., 0, 1, 2, 3)
size = comm.Get_size()  # The total number of processes

# --- Helper function for printing ---
# This ensures that output from different ranks doesn't get jumbled
def print_rank(message):
    """Prints a message prefixed with the current MPI rank and hostname."""
    hostname = socket.gethostname()
    print(f"[Rank {rank:02d}/{size:02d} on {hostname}]: {message}", flush=True)

# --- Main Test Logic ---
try:
    # 1. Print Basic MPI and System Info
    print_rank("Hello from the updated test script!")

    # 2. Check for SLURM and MPI Environment Variables
    print_rank("-" * 40)
    print_rank("SLURM & MPI Environment:")
    env_vars_to_check = [
        'SLURM_JOB_ID',
        'SLURM_NTASKS',
        'SLURM_TASK_PID', # More reliable than PROC_ID
        'PMI_RANK',       # Often set by the MPI library
        'SLURM_CPUS_PER_TASK',
        'SLURM_GPUS',
        'CUDA_VISIBLE_DEVICES'
    ]
    for var in env_vars_to_check:
        value = os.getenv(var, "== NOT SET ==")
        print_rank(f"  {var} = {value}")

    # 3. Check for a Custom Exported Variable
    print_rank("-" * 40)
    print_rank("Custom Variable Check:")
    custom_var = os.getenv('MY_TEST_VARIABLE', "== NOT SET ==")
    print_rank(f"  MY_TEST_VARIABLE = {custom_var}")
    if "== NOT SET ==" in custom_var:
        print_rank("  WARNING: Custom variable was not exported correctly!")

    # 4. Check PyTorch and GPU Availability
    print_rank("-" * 40)
    print_rank("PyTorch GPU Check:")
    try:
        import torch
        print_rank(f"  PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print_rank("  SUCCESS: PyTorch can access CUDA.")
            device_count = torch.cuda.device_count()
            print_rank(f"  PyTorch sees {device_count} GPU(s).")
            
            # Print info for each visible device
            for i in range(device_count):
                device_name = torch.cuda.get_device_name(i)
                print_rank(f"    - Device [{i}]: {device_name}")
        else:
            print_rank("  FAILURE: torch.cuda.is_available() is False.")
            if os.getenv('CUDA_VISIBLE_DEVICES') is None:
                print_rank("  REASON: The 'CUDA_VISIBLE_DEVICES' environment variable is not set!")

    except ImportError:
        print_rank("  FAILURE: Could not import PyTorch.")
    except Exception as e:
        print_rank(f"  An error occurred during PyTorch check: {e}")

    # Synchronize before exiting to make sure all output is printed
    comm.Barrier()
    time.sleep(0.5)
    if rank == 0:
        print("\n[Rank 00/--]: Test script finished.")

except Exception as e:
    print_rank(f"An unhandled error occurred: {e}")

