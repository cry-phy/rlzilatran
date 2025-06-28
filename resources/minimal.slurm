#!/bin/bash

#SBATCH --job-name=het
#SBATCH --nodes=1
#SBATCH --output=het-test.log

#SBATCH --ntasks=1
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=1

#SBATCH hetjob

#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1

srun --mpi=pmi2 --export=ALL \
   echo "Hello from GPU task" \
  : echo "Hello from CPU task"
