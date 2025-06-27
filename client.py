# client.py
from mpi4py import MPI
import numpy as np
import time

PORT_NAME = "my_mpi_port"

def test_communicator(comm, rank):
    """Test large message latency and throughput on given communicator."""
    size = 1024 * 1024 * 50  # 50 MB
    data = np.ones(size, dtype=np.byte)
    recv_buf = np.empty_like(data)

    partner = (rank + 1) % comm.Get_size()

    # Warm-up
    comm.Barrier()

    # Measure latency
    start = time.time()
    for _ in range(10):
        if rank == 0:
            comm.Send([data, MPI.BYTE], dest=partner, tag=77)
            comm.Recv([recv_buf, MPI.BYTE], source=partner, tag=78)
        elif rank == partner:
            comm.Recv([recv_buf, MPI.BYTE], source=0, tag=77)
            comm.Send([data, MPI.BYTE], dest=0, tag=78)
    end = time.time()
    latency = (end - start) / 20

    # Measure throughput
    start = time.time()
    for _ in range(10):
        if rank == 0:
            comm.Send([data, MPI.BYTE], dest=partner, tag=99)
        elif rank == partner:
            comm.Recv([recv_buf, MPI.BYTE], source=0, tag=99)
    end = time.time()
    throughput = (size * 10) / (end - start) / 1024 / 1024  # MB/s

    if rank == 0:
        print(f"Latency (ping-pong) per message: {latency*1000:.2f} ms")
        print(f"Throughput: {throughput:.2f} MB/s")

def main():
    comm_world = MPI.COMM_WORLD
    rank = comm_world.Get_rank()

    # Get port name from somewhere, here passed as CLI arg or file
    # For simplicity, input port name
    if rank == 0:
        port = input("Enter server port: ")
    else:
        port = None
    port = comm_world.bcast(port, root=0)

    # Connect
    intercomm = comm_world.Connect(port, root=0)
    print(f"[{rank}] Connected intercommunicator size: {intercomm.Get_size()}")

    # Merge into intracomm
    intracomm = intercomm.Merge(high=True)
    print(f"[{rank}] Merged communicator size: {intracomm.Get_size()}")

    # Test baseline with MPI_COMM_WORLD
    if rank == 0:
        print("Baseline test with MPI_COMM_WORLD")
    test_communicator(comm_world, rank)

    # Test with merged communicator (dynamic)
    if rank == 0:
        print("Test with merged intercommunicator")
    test_communicator(intracomm, rank)

if __name__ == "__main__":
    main()

