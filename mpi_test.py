import sys
sys.path.append("src")
from mpi import SMPI

rank = SMPI.COMM_WORLD.Get_rank()
print("Hello from rank ", rank)
