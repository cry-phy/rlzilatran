from __future__ import annotations
import os

from game.utils.constants import Constants
from cnn import CNN

from enum import IntEnum, auto
from functools import wraps
from mpi4py import MPI
from time import time
import numpy as np
import traceback
import logging
import torch
import sys


logger = logging.getLogger(__name__)


def send_gracefull_exit():
    """Gracefully Shutdown all processes."""
    mpi = MPIHandler()
    requests = mpi.send_exit()
    mpi.recv_exit()
    start_time = time()
    while requests and time() - start_time <= mpi.MPI_SHUTDOWN_WAIT:
        to_remove: list[MPI.Request] = []
        for req in requests:
            if req.Get_status():
                to_remove.append(req)
        [requests.remove(req) for req in to_remove]
    mpi.wcomm.Abort(1)


def handle_gracefull_exit():
    """Handle exit call."""
    mpi = MPIHandler()
    mpi.recv_exit()
    logger.info("Different Rank observed exception. Finalizing.")
    sys.exit(1)


def gracefull_error(func):
    """Wrap enabling MPI shutdown on Exception.

    Handles errors inside the wrapped function gracefully.
    I. e. Sends termination to all Ranks to finalize MPI.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        mpi = MPIHandler()
        try:
            if mpi.probe_exit():
                handle_gracefull_exit()
            value = func(*args, **kwargs)
        except Exception:
            exc_type, exc_value, exc_tb = sys.exc_info()
            tb_list = traceback.format_exception(exc_type, exc_value, exc_tb)
            tb_str = "".join(tb_list)
            logger.error("In function %s: \n %s", func.__name__, tb_str)
            value = None
            send_gracefull_exit()
        return value

    return wrapper


class Tag(IntEnum):
    """Make tags easier to understand."""

    EXIT = auto()
    EPISODE = auto()
    NOTIFY_WEIGHTS = auto()
    LEAF_BATCH = auto()
    VALUE_BATCH = auto()


class TokenHandler:
    """Handler for integer message Tokens.

    Currently only creates unique tokens on rank.
    """

    def __init__(self, dtype: type[np.integer] = np.int32) -> None:
        """Set token starting value to 0."""
        self._current_id = 0
        self.dtype = dtype

    def new(self) -> np.ndarray:
        """Generate a new token."""
        self._current_id += 1
        return np.full(1, self._current_id, dtype=self.dtype)

    def empty_message(self, data: np.ndarray) -> np.ndarray:
        """Generate an empty message buffer."""
        message: np.ndarray = np.empty(
            1,
            dtype=[
                ("token", self.dtype),
                ("data", data.dtype, data.shape),
            ],
        )
        return message

    def tokenize_data(self, data: np.ndarray) -> np.ndarray:
        """Copy Data into structure array and add a token."""
        message = self.empty_message(data)
        message["token"] = self.new()
        message["data"] = data
        return message


class MPIHandler:
    """MPI Handler singleton."""

    _instance = None
    token = TokenHandler()
    _init = False

    valid_process = ("training", "inference", "worker")

    def __new__(cls) -> MPIHandler:
        """Singleton Instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    @gracefull_error
    def slurm_test(cls, num_gpus: int, agents_per_observer: int) -> MPIHandler:
        """Emulate Slurm MPI handler locally."""
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
        processes = (
            ["training"] + ["inference"] * num_gpus + ["worker"] * (size - num_gpus - 1)
        )
        return cls.slurm(processes[rank], agents_per_observer)

    @classmethod
    @gracefull_error
    def slurm(cls, process: str, agents_per_observer: int) -> MPIHandler:
        """Generate MPIHandler for Slurm hetjob execution."""
        if process not in cls.valid_process:
            raise ValueError(f"{process} is not a valid process {cls.valid_process}")

        instance = cls()
        instance.process = process

        if process == "worker" and torch.cuda.is_available():
            logger.warning(
                f"Rank {instance.wrank} was assigned a GPU, but the process role '{process}' does not require it."
            )
        if process in ("training", "inference") and not torch.cuda.is_available():
            raise RuntimeError(
                f"Rank {instance.wrank} was not assigned a GPU, but the process role '{process}' requires it."
            )

        if process in ("training", "inference"):
            instance.device = torch.device("cuda:0")
            logger.info(
                "Assigning %s to this process.",
                torch.cuda.get_device_name(instance.device),
            )

        world_duties = instance.wcomm.allgather((instance.wrank, process))
        training_ranks = [r for r, p in world_duties if p == "training"]

        if not training_ranks:
            raise RuntimeError("No training ranks found — cannot assign.")

        num_training = len(training_ranks)
        bp_rank = instance.wsize % num_training
        block_size = instance.wsize // num_training

        inference_ranks = [r for r, p in world_duties if p == "inference"]
        worker_ranks = [r for r, p in world_duties if p == "worker"]
        if not inference_ranks:
            raise RuntimeError("No inference ranks found — cannot assign workers.")
        average_workers = len(worker_ranks) // len(inference_ranks)

        if instance.wrank < (block_size + 1) * bp_rank:
            block_index = instance.wrank // (block_size + 1)
        else:
            block_index = (instance.wrank - bp_rank) // block_size

        instance.TRAINING_WRANK = training_ranks[block_index]

        non_training_comm = instance._split_training()
        if non_training_comm != MPI.COMM_NULL:
            shared_comms = non_training_comm.Split_type(MPI.COMM_TYPE_SHARED)
            rank = non_training_comm.Get_rank()
            shared_duties = shared_comms.allgather((rank, instance.process))
            shared_inference = [r for r, p in shared_duties if p == "inference"]

            abandoned = not shared_inference
            node_id = int(os.environ["SLURM_NODEID"])
            ranks_status = non_training_comm.allgather((rank, node_id, abandoned))
            abandoned = [r for r, _, a in ranks_status if a]
            non_abandoned = [(r, nid) for r, nid, a in ranks_status if not a]

            shared_workers = [r for r, p in shared_duties if p == "worker"]
            num_shared_workers = len(shared_workers)
            if not abandoned:
                needed_workers = max(average_workers - num_shared_workers, 0)
            else:
                needed_workers = None
            needed_list = non_training_comm.allgather((node_id, needed_workers))
            all_node_ids = [
                node_id
                for node_id, needed_workers in needed_list
                if needed_workers is not None
            ]
            node_ids = list(set(all_node_ids))
            node_to_needed = {}
            for node_id, needed in needed_list:
                # If node_id not recorded yet, save it
                if node_id not in node_to_needed and needed != 0:
                    node_to_needed[node_id] = needed

            assignments = {node_id: [] for node_id in node_ids}
            while abandoned:
                # highest need first
                node_id = max(
                    (nid for nid in node_to_needed if node_to_needed[nid] > 0),
                    key=lambda nid: node_to_needed[nid],
                    default=None,
                )
                abandoned_rank = abandoned.pop()
                assignments[node_id].append(abandoned_rank)
                node_to_needed[node_id] -= 1

            for rank, node_id in non_abandoned:
                assignments[node_id].append(rank)

            rank_to_node = {}
            for node_id, ranks in assignments.items():
                for r in ranks:
                    rank_to_node[r] = node_id

            ncolor = rank_to_node.get(rank, MPI.UNDEFINED)
            full_node_comm = non_training_comm.Split(ncolor, rank)
            full_node_rank = full_node_comm.Get_rank()
            full_node_size = full_node_comm.Get_size()

            node_duties = full_node_comm.allgather((full_node_rank, process))
            inference_ranks = [r for r, p in node_duties if p == "inference"]

            if instance.process == "inference":
                block_index = inference_ranks.index(full_node_rank)
            else:
                num_inference = len(inference_ranks)
                bp_rank = full_node_size % num_inference
                block_size = full_node_size // num_inference
                if full_node_rank < (block_size + 1) * bp_rank:
                    block_index = full_node_rank // (block_size + 1)
                else:
                    block_index = (full_node_rank - bp_rank) // block_size
            full_node_color = block_index
            instance.ncomm = full_node_comm.Split(full_node_color, full_node_color)
            instance.nrank = instance.ncomm.Get_rank()
            instance.nsize = instance.ncomm.Get_size()

            ncomm_duties = instance.ncomm.allgather((instance.nrank, process))
            inference_nranks = [r for r, p in ncomm_duties if p == "inference"]
            if not inference_nranks:
                raise RuntimeError("No inference ranks found — cannot assign workers.")
            if len(inference_nranks) > 1:
                logger.warning(
                    "CArefull there might be the wrong number of inference processes in ncomm: Is: %i should be 1",
                    len(inference_nranks),
                )
            instance.INFERENCE_NRANK = inference_nranks[0]

            color = MPI.UNDEFINED if instance.is_inference() else 0
            non_inference_comm = instance.ncomm.Split(color, instance.nrank)
            if non_inference_comm != MPI.COMM_NULL:
                non_inference_size = non_inference_comm.Get_size()
                non_inference_rank = non_inference_comm.Get_rank()
                number_observers = non_inference_size // agents_per_observer
                bp_rank = non_inference_size % number_observers
                observer_color = (
                    non_inference_rank // (agents_per_observer + 1)
                    if non_inference_rank <= bp_rank
                    else non_inference_rank // agents_per_observer
                )
                instance.ocomm = non_inference_comm.Split(
                    observer_color, non_inference_rank
                )
                instance.orank = instance.ocomm.Get_rank()
                instance.osize = instance.ocomm.Get_size()

        cnn_wranks = instance._get_cnn_wranks()
        cnncomm, cnnrank, cnnsize, cnn_training_rank = instance._split_cnn(
            instance.wcomm, cnn_wranks
        )
        instance.cnncomm = cnncomm
        instance.cnnrank = cnnrank
        instance.cnnsize = cnnsize

        instance.cnn_training_rank = cnn_training_rank

        return instance

    def _avail_num_gpus(self, num_gpus: int) -> int:
        """Validate and calculate the number of available gpus for this process."""
        if self.wrank == self.TRAINING_WRANK:
            self.process = "training"
            training_name = MPI.Get_processor_name()
            if num_gpus == 0:
                raise ValueError("No GPUS! Cannot continue.")
            if num_gpus >= 2:
                self.device = torch.device(f"cuda:{num_gpus - 1}")
            else:
                self.device = torch.device("cuda:0")
        else:
            training_name = "NON_TRAINING_NODE"

        training_name = self.wcomm.bcast(training_name, self.TRAINING_WRANK)

        name = MPI.Get_processor_name()
        if name == training_name:
            num_gpus = max(num_gpus - 1, 1)

        return num_gpus

    def _split_training(self) -> MPI.Comm:
        """Split away training rank."""
        color = MPI.UNDEFINED if self.wrank == self.TRAINING_WRANK else 0
        return self.wcomm.Split(color, self.wrank)

    def _split_node(
        self, avail_comm: MPI.Comm, num_gpus: int
    ) -> tuple[MPI.Comm, int | None, int | None, torch.device | None]:
        """Split by shared memory resources."""
        if avail_comm != MPI.COMM_NULL:
            tempcomm = avail_comm.Split_type(MPI.COMM_TYPE_SHARED)
            tempsize = tempcomm.Get_size()
            temprank = tempcomm.Get_rank()

            # Group by GPU
            num_per_gpu = tempsize // num_gpus
            bp_rank = tempsize % num_gpus
            color = (
                temprank // (num_per_gpu + 1)
                if temprank <= bp_rank
                else temprank // num_per_gpu
            )
            device = torch.device(f"cuda:{color}")
            comm = tempcomm.Split(color, temprank)
            rank = comm.Get_rank()
            size = comm.Get_size()
            if rank == self.INFERENCE_NRANK:
                self.process = "inference"
        else:
            comm = MPI.COMM_NULL
            rank = None
            size = None
            device = self.device
        return comm, rank, size, device

    def _split_observer(
        self, avail_comm: MPI.Comm, agents_per_observer: int
    ) -> tuple[MPI.Comm, int | None, int | None]:
        """Split communicator into observer comms."""
        if avail_comm != MPI.COMM_NULL:
            rank = avail_comm.Get_rank()
            color = MPI.UNDEFINED if rank == self.INFERENCE_NRANK else 0
            tempcomm = avail_comm.Split(color, rank)
            if tempcomm != MPI.COMM_NULL:
                processes_per_gpu = tempcomm.Get_size()
                gpu_rank = tempcomm.Get_rank()
                number_observers = processes_per_gpu // agents_per_observer
                bp_rank = processes_per_gpu % number_observers
                observer_color = (
                    gpu_rank // (agents_per_observer + 1)
                    if gpu_rank <= bp_rank
                    else gpu_rank // agents_per_observer
                )
                comm = tempcomm.Split(observer_color, gpu_rank)

                # If actually part of Observer
                rank = comm.Get_rank()
                size = comm.Get_size()
                return comm, rank, size

        # Default
        comm = MPI.COMM_NULL
        rank = None
        size = None
        return comm, rank, size

    def _get_cnn_wranks(self) -> torch.Tensor:
        """Get world comm ranks of inference ranks."""
        is_cnn = self.is_training() or self.is_inference()
        if is_cnn:
            logger.info(
                f"Rank {self.wrank} is using {torch.cuda.get_device_name(self.device)} for CNN."
            )
        are_cnn = self.wcomm.allgather(is_cnn)
        return torch.where(torch.tensor(are_cnn, dtype=torch.bool))[0]

    def _split_cnn(
        self, wcomm: MPI.Comm, cnn_wranks: torch.Tensor
    ) -> tuple[MPI.Comm, int | None, int | None, int | None]:
        """Weight sending comm setup."""
        cnn_color = MPI.UNDEFINED if self.wrank not in cnn_wranks else 0
        comm = wcomm.Split(cnn_color, self.wrank)
        if comm != MPI.COMM_NULL:
            rank = comm.Get_rank()
            size = comm.Get_size()

            # Broadcast training rank in the cnn comm to all
            cnn_training_rank = wcomm.bcast(rank, self.TRAINING_WRANK)
        else:
            comm = MPI.COMM_NULL
            rank = None
            size = None
            cnn_training_rank = None
        return comm, rank, size, cnn_training_rank

    @classmethod
    @gracefull_error
    def local(cls, num_gpus: int, agents_per_observer: int) -> MPIHandler:
        """!!! DEPRECATED !!!.

        Will be replaced by slurm_test soon.

        Generate MPIHandler for local execution.
        """
        logger.warning("local standard exeution be replaced by slurm test soon.")

        instance = cls()
        avail_gpus = instance._avail_num_gpus(num_gpus)
        non_train_comm = instance._split_training()
        ncomm, nrank, nsize, device = instance._split_node(non_train_comm, avail_gpus)
        instance.ncomm = ncomm
        instance.nrank = nrank
        instance.nsize = nsize
        instance.device = device
        ocomm, orank, osize = instance._split_observer(ncomm, agents_per_observer)
        instance.ocomm = ocomm
        instance.orank = orank
        instance.osize = osize
        cnn_wranks = instance._get_cnn_wranks()
        cnncomm, cnnrank, cnnsize, cnn_training_rank = instance._split_cnn(
            instance.wcomm, cnn_wranks
        )
        instance.cnncomm = cnncomm
        instance.cnnrank = cnnrank
        instance.cnnsize = cnnsize

        instance.cnn_training_rank = cnn_training_rank

        return instance

    def __init__(self) -> None:
        """Initialize MPI Comms.

        Global rank 0:      Training process
        Node rank 0:        Inference process
        Observer rank 0:    Observer process
        Observer comm is needed for BCast and AllReduce of the move suggestions.
        For reference please see resources/Processes.png.

        !!! Will remove Singleton Pattern soon !!!
        """
        # Guard against multiple initializations due to MPIHandler calls in class blocks.
        if self._init:
            return

        self.MPI_SHUTDOWN_WAIT = 30
        self.TRAINING_WRANK = 0
        self.INFERENCE_NRANK = 0
        self.OBSERVER_ORANK = 0

        # World comm setup
        self.wcomm: MPI.Comm = MPI.COMM_WORLD
        self.wrank: int = self.wcomm.Get_rank()
        self.wsize: int = self.wcomm.Get_size()

        # Initialize to empty comms
        self.ncomm: MPI.Comm = MPI.COMM_NULL
        self.nrank: int | None = None
        self.nsize: int | None = None
        self.ocomm: MPI.Comm = MPI.COMM_NULL
        self.orank: int | None = None
        self.osize: int | None = None
        self.cnncomm: MPI.Comm = MPI.COMM_NULL
        self.cnnrank: int | None = None
        self.cnnsize: int | None = None

        self.device: torch.device | None = None
        self.process: str = "UNNASSIGNED"

        self.cnn_training_rank: int | None = None
        self._init = True

    def get_device(self) -> torch.device:
        """Return the current device."""
        if self.device is None:
            raise ValueError("Cuda Device not set. Exiting.")
        return self.device

    def is_training(self) -> bool:
        """Check if this rank is an inference Process."""
        return self.process == "training"

    def is_inference(self) -> bool:
        """Check if this rank is an inference Process."""
        return self.process == "inference"

    def is_observer(self) -> bool:
        """Check if this rank is an inference Process."""
        return self.orank == self.OBSERVER_ORANK

    def is_agent(self) -> bool:
        """Check if this rank is an inference Process."""
        return (
            not self.is_training()
            and not self.is_inference()
            and not self.is_observer()
        )

    def force_exit(self):
        """Force world comm abort."""
        self.wcomm.Abort(1)

    def send_exit(self) -> list[MPI.Request]:
        """Send exit message to all ranks."""
        requests: list[MPI.Request] = []
        empty_msg = np.empty(0, dtype="b")
        for target in range(self.wsize):
            req: MPI.Request = self.wcomm.Isend(
                [empty_msg, MPI.BYTE], target, int(Tag.EXIT)
            )
            requests.append(req)
        return requests

    def probe_exit(self) -> bool:
        """Check if an exit message is sent."""
        return self.wcomm.Iprobe(MPI.ANY_SOURCE, int(Tag.EXIT))

    def recv_exit(self) -> None:
        """Recieve an exit message."""
        empty_msg = np.empty(0, dtype="b")
        self.wcomm.Recv([empty_msg, MPI.BYTE], MPI.ANY_SOURCE, int(Tag.EXIT))

    def finalize(self) -> None:
        """Gracefull finalize MPI Environment."""
        MPI.Finalize()

    # --- Send Episode Data from Observer to Training ---
    def send_episode(
        self, states: torch.Tensor, values: torch.Tensor, move_tensor: torch.Tensor
    ) -> None:
        """Send the episode data to the training Rank."""
        np_states = states.numpy()
        np_values = values.numpy()
        np_move_tensors = move_tensor.numpy()
        message: np.ndarray = np.empty(
            1,
            dtype=[
                ("states", "f4", Constants.state_tensor_batch_shape),
                ("values", "f4", Constants.value_batch_shape),
                ("move_tensor", "f4", Constants.move_tensor_batch_shape),
            ],
        )
        message["states"] = np_states
        message["values"] = np_values
        message["move_tensor"] = np_move_tensors
        self.wcomm.Isend([message, MPI.BYTE], self.TRAINING_WRANK, int(Tag.EPISODE))

    def probe_episode(self, source: int) -> bool:
        """Probe if episode was sent from rank."""
        return self.wcomm.Iprobe(source, int(Tag.EPISODE))

    def recv_episode(
        self, source: int
    ) -> tuple[MPI.Request, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Recieve the episode data fom the training Rank.

        !!! Carefull: Before copy operations await data. !!!
        Using a future apporach as it allows for gathering without inhibiting.
        """
        message: np.ndarray = np.empty(
            1,
            dtype=[
                ("states", "f4", Constants.state_tensor_batch_shape),
                ("values", "f4", Constants.value_batch_shape),
                ("move_tensor", "f4", Constants.move_tensor_batch_shape),
            ],
        )
        req: MPI.Request = self.wcomm.Irecv(
            [message, MPI.BYTE], source, int(Tag.EPISODE)
        )
        return (
            req,
            torch.from_numpy(message["states"].squeeze(0)),
            torch.from_numpy(message["values"].squeeze(0)),
            torch.from_numpy(message["move_tensor"].squeeze(0)),
        )

    # --- Distribute Weights from Training to Inference ---
    def send_weights_ready(self) -> None:
        """Send data to all processes that weights are ready."""
        empty_msg = np.empty(0, dtype="b")
        for target in range(self.cnnsize):
            self.cnncomm.Isend(
                [empty_msg, MPI.BYTE], int(target), int(Tag.NOTIFY_WEIGHTS)
            )

    def probe_weights_ready(self) -> bool:
        """Check if a notification of weights ready exists."""
        return self.wcomm.Iprobe(0, int(Tag.NOTIFY_WEIGHTS))

    def recv_weights_ready(self) -> None:
        """Send data to all processes that weights are ready."""
        empty_msg = np.empty(0, dtype="b")
        self.wcomm.Irecv([empty_msg, MPI.BYTE], 0, int(Tag.NOTIFY_WEIGHTS))

    def _bcast_size(self, size: int) -> int:
        """Blocking size distribution to allow for non blocking."""
        size = self.cnncomm.bcast(size, self.TRAINING_WRANK)
        return size

    def bcast_weights(self, model: CNN) -> None:
        """Send Weights to Inference Process.

        The weights are flattened and need to be reassembled by the CNN.
        In the future it might be helpfull to switch to torch.distribution to allow
        for NCCL direct GPU-GPU interaction.
        """
        logger.debug("Broadcasting weights.")
        if self.wrank == self.TRAINING_WRANK:
            bytestream: bytes = model.to_bytestream()
            buffer = memoryview(bytestream)
            _ = self._bcast_size(len(bytestream))
        else:
            size = self._bcast_size(0)
            buffer = bytearray(size)
        req: MPI.Request = self.cnncomm.Ibcast(
            [buffer, MPI.BYTE],
            self.cnn_training_rank,
        )
        if not self.wrank == self.TRAINING_WRANK:
            req.Wait()
            model.from_bytestream(bytes(buffer))

    # --- Synchronize Observer and Agents ---
    def bcast_decided_move(
        self, move: tuple[int, int, int, int, int] | None
    ) -> tuple[int, int, int, int, int]:
        """Broadcast new game state to all agents."""
        logger.debug("Broadcasting decided move.")
        if self.orank == self.OBSERVER_ORANK:
            if move is None:
                raise ValueError("Attempting to Bcast None from self.OBSERVER_ORANK.")
        else:
            if move is not None:
                raise ValueError("Attempting to send data on Agent Rank.")
        return self.ocomm.bcast(move, self.OBSERVER_ORANK)

    def reduce_move_tensors(self, move_tensor: torch.Tensor) -> torch.Tensor:
        """Gather all move suggestion tensors from agents."""
        logger.debug("Reducing move Tensors")
        move_numpy_sum = np.empty(Constants.move_tensor_shape, dtype=np.float32)
        move_numpy = move_tensor.numpy()
        self.ocomm.Reduce(move_numpy, move_numpy_sum, MPI.SUM)
        return torch.from_numpy(move_numpy_sum)

    def reduce_values(self, value: float) -> float | None:
        """Reduce values from all ocomm."""
        logger.debug("Reducing values.")
        red_value: float | None = self.ocomm.reduce(value, MPI.SUM, self.OBSERVER_ORANK)
        return red_value

    # --- Leaves from Agents to Inference ---
    def send_leaf_batch(self, leaf_batch: torch.Tensor) -> int:
        """Send Leaf batch to inference, returning a unique Token."""
        np_leaves = leaf_batch.numpy()
        full_message = self.token.tokenize_data(np_leaves)
        flat_message = np.concatenate(
            (
                full_message["token"].view(Constants.leaf_batch_dtype),
                full_message["data"].flatten().view(Constants.leaf_batch_dtype),
            )
        )
        self.ncomm.Isend(
            [flat_message, MPI.BYTE], self.INFERENCE_NRANK, int(Tag.LEAF_BATCH)
        )
        print("sent", int(full_message["token"]))
        self.ncomm.Barrier()
        return int(full_message["token"])

    def probe_leaf_batch(self, nsource: int) -> bool:
        """Check if leaf batch from nsource exists."""
        return self.ncomm.Iprobe(nsource, int(Tag.LEAF_BATCH))

    def recv_leaf_batch(
        self, nsource: int
    ) -> tuple[MPI.Request, int, int, torch.Tensor]:
        """Recieve Leaf batch and nsource from evaluation."""
        data = np.empty(Constants.leaf_batch_shape, dtype=Constants.leaf_batch_dtype)
        full_message = self.token.empty_message(data)
        flat_message = np.concatenate(
            (
                full_message["token"].view(Constants.leaf_batch_dtype),
                full_message["data"].flatten().view(Constants.leaf_batch_dtype),
            )
        )
        req: MPI.Request = self.ncomm.Irecv(flat_message, nsource, int(Tag.LEAF_BATCH))
        header_slots = full_message["token"].nbytes // flat_message.dtype.itemsize
        req.Wait()
        self.ncomm.Barrier()
        token = flat_message[:header_slots].view(self.token.dtype)[0]
        print("recv", token)
        recv_data = flat_message[header_slots:].reshape(data.shape)
        return req, nsource, token, torch.from_numpy(recv_data)

    # --- Values from Inference to Agents ---
    def send_value_batch(
        self,
        ntarget: int,
        token: int,
        values: torch.Tensor,
        move_tensors: torch.Tensor,
    ) -> None:
        """Send value + move batch to target nrank, flattened into one float32 buffer."""
        np_values = values.detach().cpu().numpy().astype("f4")
        np_move_tensors = move_tensors.detach().cpu().numpy().astype("f4")

        token_dtype = np.dtype(self.token.dtype)
        float_dtype = np.dtype(Constants.value_batch_dtype)
        token_slots = token_dtype.itemsize // float_dtype.itemsize
        val_slots = int(np_values.size)
        move_slots = int(np_move_tensors.size)
        total_slots = token_slots + val_slots + move_slots

        flat = np.empty(total_slots, dtype=Constants.value_batch_dtype)
        flat[:token_slots] = np.array([token], dtype=self.token.dtype).view("f4")
        flat[token_slots : token_slots + val_slots] = np_values.ravel()
        flat[token_slots + val_slots :] = np_move_tensors.ravel()
        self.ncomm.Isend([flat, MPI.BYTE], ntarget, int(Tag.VALUE_BATCH))

    def probe_value_batch(self) -> bool:
        """Check if a value batch exists."""
        return self.ncomm.Iprobe(self.INFERENCE_NRANK, int(Tag.VALUE_BATCH))

    def recv_value_batch(
        self,
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        """Receive flattened value+move batch from inference, unpack token + tensors."""
        token_dtype = np.dtype(self.token.dtype)
        float_dtype = np.dtype(Constants.value_batch_dtype)
        token_slots = token_dtype.itemsize // float_dtype.itemsize

        val_shape = Constants.value_batch_shape
        move_shape = Constants.move_tensor_batch_shape
        val_slots = int(np.prod(val_shape))
        move_slots = int(np.prod(move_shape))
        total_slots = token_slots + val_slots + move_slots

        flat = np.empty(total_slots, dtype=Constants.value_batch_dtype)
        self.ncomm.Recv([flat, MPI.BYTE], self.INFERENCE_NRANK, int(Tag.VALUE_BATCH))

        token = flat[:token_slots].view(self.token.dtype)[0]
        start = token_slots
        vals_flat = flat[start : start + val_slots]
        moves_flat = flat[start + val_slots :]

        values = torch.from_numpy(vals_flat.reshape(val_shape))
        move_tensors = torch.from_numpy(moves_flat.reshape(move_shape))

        return token, values, move_tensors
