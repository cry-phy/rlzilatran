#!/bin/bash

# Function to display help
show_help() {
    echo "Usage: $0 <jobname> [options]"
    echo ""
    echo "Options:"
    echo "  -h, --help                     Display this help message"
    echo "  -q=<queue>, --queue=<queue>    Specify the queue name (default: std)"
    echo "  -np=<number>, --number-processes=<number>"
    echo "                                 Number of processes (default: 1)"
    echo "  -nc=<cpus>, --cpus-per-node=<cpus>"
    echo "                                 Number of CPUs per node (required)"
    echo "  -nt=<threads>, --threads-per-process=<threads>"
    echo "                                 Number of threads per process (default: 8)"
    echo "  -t=<hours>, --execution-time=<hours>"
    echo "                                 Estimated calculation time in hours (default: 1)"
    echo "  -m=<memory>, --memory=<memory> Specify memory per CPU in MB (default: 1000)"
    echo ""
    echo "Example:"
    echo "  $0 rlzilatran_xxxx --queue=bigmem --number-processes=4 --cpus-per-node=16 --threads-per-process=8 --execution-time=5 --memory=4000"
}
for ARG in "$@"; do
    case $ARG in
        -h|--help)
            show_help
            exit 0
            ;;
    esac
done
echo "Preparing slurm script for RLZilatran:"

# Generate correct Conda environment
# Standard values for the parameters
STD_QUEUE_NAME="std"
STD_NUM_PROCESSES="1"
STD_THREADS_PER_PROCESS="1"
STD_EST_CALC_TIME="1"
STD_MEM_in_MB="1000"
STD_NUM_GPUS="2"
STD_CONDA_ENV="rlzilatran"

# Set Standard values
QUEUE_NAME="$STD_QUEUE_NAME"
NUM_PROCESSES="$STD_NUM_PROCESSES"
THREADS_PER_PROCESS="$STD_THREADS_PER_PROCESS"
EST_CALC_TIME="$STD_EST_CALC_TIME"
MEM_in_MB="$STD_MEM_in_MB"
NUM_GPUS="$STD_NUM_GPUS"
CONDA_ENV="$STD_CONDA_ENV"
PREPARE=false

# Check through flags
if [ -z "$1" ]; then echo "The parameter jobname (at position 1) must be specified. Call the function like
./SendToQueue <jobname>... keyed arguments" && exit 1; else JOBNAME=$1 && shift; fi
for ARG in "$@"; do
  case $ARG in
      -h=*|--help=*)
        show_help
        exit
        ;;
      -q=*|--queue=*)
        QUEUE_NAME="${1#*=}"
        shift
        ;;
      -np=*|--number-processes=*)
        NUM_PROCESSES="${1#*=}"
        shift
        ;;
      -nc=*|--cpus-per-node=*)
        CPUS_PER_NODE="${1#*=}"
        shift
        ;;
      -nt=*|--threads-per-process=*)
        THREADS_PER_PROCESS="${1#*=}"
        shift
        ;;
      -t=*|--execution-time=*)
        EST_CALC_TIME="${1#*=}"
        shift
        ;;
      -m=*|--memory=*)
        MEM_in_MB="${1#*=}"
        shift
        ;;
      -g=*|--gpus=*)
        NUM_GPUS="${1#*=}"
        shift
        ;;
      --conda=*)
        CONDA_ENV="${1#*=}"
        shift;;
      --prepare)
        PREPARE=true
        shift
        ;;
      *)
        echo "Unknown option: $ARG"
        exit 1
        ;;
  esac
done

# Check if variables have been set by user and notify fallback to standard value
if [ "$QUEUE_NAME" = "$STD_QUEUE_NAME" ]; then echo "The queue name defaults to $QUEUE_NAME."; fi
if [ "$NUM_PROCESSES" = "$STD_NUM_PROCESSES" ]; then echo "The number of processes defaults to $NUM_PROCESSES."; fi
if [ "$THREADS_PER_PROCESS" = "$STD_THREADS_PER_PROCESS" ]; then echo "The number of threads per process defaults to $THREADS_PER_PROCESS."; fi
if [ "$EST_CALC_TIME" = "$STD_EST_CALC_TIME" ]; then echo "The estimated calculation time in hours defaults to $EST_CALC_TIME."; fi
if [ "$MEM_in_MB" = "$STD_MEM_in_MB" ]; then echo "The memory per cpu defaults too $MEM_in_MB."; fi
if [ "$CONDA_ENV" = "$STD_CONDA_ENV" ]; then echo "Using default conda environment $CONDA_ENV."; fi


# Check to use CPUs per node
if [ -n "$CPUS_PER_NODE" ]; then
  echo "Using $CPUS_PER_NODE CPUs per node."
else
  if [ -z "$SLURM_CPUS_PER_NODE" ]; then
    echo "Variable SLURM_CPUS_PER_NODE not set in this environment! Avoid this message by using --cpus-per-node or -nc"
    echo "Please enter the value for the number of CPUs per Node: "
    read CPUS_PER_NODE

    # Check if the user entered a value
    if [ -z "$CPUS_PER_NODE" ]; then
      echo "Error: No value entered for CPUS_PER_NODE. Please try again."
      exit 1
    else
      echo "You entered: $CPUS_PER_NODE"
    fi
  fi
fi

# Calculate how many nodes are needed
TOTAL_CPUS=$((NUM_PROCESSES * THREADS_PER_PROCESS))
NUM_NODES=$(( (TOTAL_CPUS + CPUS_PER_NODE - 1) / CPUS_PER_NODE ))

# Provide user Feedback
echo "Reserving $NUM_NODES nodes for $NUM_PROCESSES tasks with $THREADS_PER_PROCESS threads per task"


# Prepare Working directory and provide Slurm script
mkdir -p $JOBNAME
echo "#!/bin/bash

#SBATCH --job-name=zilatran.$JOBNAME
#SBATCH --time=$EST_CALC_TIME:0:0
#SBATCH --nodes=$NUM_NODES
#SBATCH --ntasks=$NUM_PROCESSES
#SBATCH --cpus-per-task=$THREADS_PER_PROCESS
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-1}
#SBATCH --gpus=$NUM_GPUS
#SBATCH --mem-per-cpu=${MEM_in_MB}M
#SBATCH --output=$JOBNAME/$JOBNAME.log
#SBATCH --output=$JOBNAME/$JOBNAME.log

#SBATCH --hint=nomultithread
#SBATCH --partition=gpustd.q

# disables core dump file generation
ulimit -c 0

conda activate "${CONDA_ENV}"

echo "Using Python:"
which python
sinfo -o "%P %G %D %N"

srun --mpi=pmi2 --export=ALL python ../src/main.py $JOBNAME
conda deactivate
" > $JOBNAME/run.$JOBNAME.slurm

# Copy config from:
#  Job directory if it already exists
#  Current directory if it exists
#  Warn user when using backup config as backup (used for debugging, can be removed at any time)
if [ -f ./$JOBNAME/config.toml ]; then
  echo "Config is not copied since already exists"
else
  if [ -f ./config.toml ]; then
    echo "Using config from current directory"
    cp --update=none config.toml $JOBNAME
  else
    echo "ERROR: No config found!"
    exit
  fi
fi

if [ -f ./$JOBNAME/constants.json ]; then
  echo "Constants is not copied since already exists"
else
  if [ -f ./constants.json ]; then
    echo "Using config from current directory"
    cp --update=none constants.json $JOBNAME
  else
    echo "ERROR: NO constants.json found!"
    exit
  fi
fi
echo "Stating Job: $JOBNAME"

if [ "$PREPARE" = false ]; then
  cd $JOBNAME
  sbatch run.$JOBNAME.slurm
  cd ..
else
 echo "Not running submission script as run with --prepare flag."
 echo "Can be run manually with:"
 echo "   sbatch run.$JOBNAME.slurm"
fi
