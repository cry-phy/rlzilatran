from __future__ import annotations
from mpi import SMPI
import os
import pynvml

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cnn import CNN

from game.utils.constants import Constants
from cnn.queue import MessageItem, Queue

from enum import IntEnum, auto
from functools import wraps
from time import sleep, time
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
        to_remove: list[SMPI.Request] = []
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
    I. e. Sends termination to all Ranks to finalize SMPI.
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


FieldSpec = tuple[str, type[np.generic], tuple[int, ...]]


class TokenHandler:
    """Handler for integer message Tokens.

    Currently only creates unique tokens on rank.
    """

    def __init__(self, dtype: type[np.integer] = np.int32) -> None:
        """Set token starting value to 0."""
        self._current_id = 0
        self.token_dtype = np.dtype(dtype)

    def new(self) -> np.ndarray:
        """Generate a new token."""
        self._current_id += 1
        return np.full(1, self._current_id, dtype=self.token_dtype)

    def empty_message(
        self,
        fields: list[FieldSpec],
    ) -> tuple[np.ndarray, SMPI.Datatype]:
        """Create an empty structured message with a token and given fields."""
        # Full dtype with token prepended
        dtype_fields = [("token", self.token_dtype)] + fields
        struct_dtype = np.dtype(dtype_fields)

        # Compute MPI datatype
        block_lengths = []
        displacements = []
        mpi_types = []

        for name in struct_dtype.names:
            field_dtype, offset = struct_dtype.fields[name][:2]
            base_dtype = field_dtype.base if field_dtype.subdtype else field_dtype
            block_len = int(np.prod(field_dtype.shape)) if field_dtype.shape else 1
            block_lengths.append(block_len)
            displacements.append(int(offset))
            mpi_types.append(SMPI._typedict[base_dtype.char])

        mpi_dtype = SMPI.Datatype.Create_struct(block_lengths, displacements, mpi_types)
        mpi_dtype.Commit()

        message = np.empty(1, dtype=struct_dtype)
        return message, mpi_dtype

    def tokenize_data(
        self,
        names: list[str],
        arrays: list[np.ndarray],
    ) -> tuple[np.ndarray, SMPI.Datatype]:
        """Build structured message from field names and homogeneous arrays."""
        assert len(names) == len(arrays), "Name and array count mismatch"

        fields: list[FieldSpec] = []
        for name, arr in zip(names, arrays, strict=True):
            dtype = arr.dtype
            shape = arr.shape
            fields.append((name, dtype, shape))

        message, mpi_dtype = self.empty_message(fields)
        message["token"] = self.new()

        for name, array in zip(names, arrays, strict=True):
            message[name] = array

        return message, mpi_dtype


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
        comm = SMPI.COMM_WORLD
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

        logger.info(
            "Node %s is preparing to run a process %s",
            os.environ["SLURM_NODEID"],
            process,
        )

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
        if non_training_comm != SMPI.COMM_NULL:
            shared_comms = non_training_comm.Split_type(SMPI.COMM_TYPE_SHARED)
            rank = non_training_comm.Get_rank()
            shared_duties = shared_comms.allgather((rank, instance.process))
            shared_inference = [r for r, p in shared_duties if p == "inference"]

            abandoned = not shared_inference
            node_id = int(os.environ["SLURM_NODEID"])
            ranks_status = non_training_comm.allgather((rank, node_id, abandoned))
            abandoned_ranks = [r for r, _, a in ranks_status if a]
            non_abandoned_ranks = [(r, nid) for r, nid, a in ranks_status if not a]

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
                if node_id not in node_to_needed and needed is not None:
                    node_to_needed[node_id] = needed

            assignments = {node_id: [] for node_id in node_ids}
            while abandoned_ranks:
                # highest need first
                node_id = max(
                    (nid for nid in node_to_needed if node_to_needed[nid] > 0),
                    key=lambda nid: node_to_needed[nid],
                    default=None,
                )
                abandoned_rank = abandoned_ranks.pop()
                assignments[node_id].append(abandoned_rank)
                node_to_needed[node_id] -= 1

            for rank, node_id in non_abandoned_ranks:
                assignments[node_id].append(rank)

            rank_to_node = {}
            for node_id, ranks in assignments.items():
                for r in ranks:
                    rank_to_node[r] = node_id

            ncolor = rank_to_node.get(rank, SMPI.UNDEFINED)
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
                    "Carefull there might be the wrong number of inference processes in ncomm: Is: %i should be 1",
                    len(inference_nranks),
                )
            instance.INFERENCE_NRANK = inference_nranks[0]

            color = SMPI.UNDEFINED if instance.is_inference() else 0
            non_inference_comm = instance.ncomm.Split(color, instance.nrank)
            if non_inference_comm != SMPI.COMM_NULL:
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

    @classmethod
    @gracefull_error
    def slurm_shared_gpu(cls, agents_per_observer: int) -> MPIHandler:
        """Yet another way to start the mpi handler.

        Used when the GPUS are non uniquely assigned to all processes via a 1 to n mapping.
        """

        def cuda_index_from_uuid(target_uuid: str) -> int:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                uuid = pynvml.nvmlDeviceGetUUID(handle)
                if uuid == target_uuid:
                    return i
            raise RuntimeError("Cannot find Cuda device for this rank.")

        instance = cls()
        pynvml.nvmlInit()
        local_index = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
        handle = pynvml.nvmlDeviceGetHandleByIndex(local_index)
        gpu_uuid = pynvml.nvmlDeviceGetUUID(handle)
        all_uuids = sorted(set(instance.wcomm.allgather(gpu_uuid)))
        if len(all_uuids) > 1:
            training_uuid = all_uuids.pop()
        else:
            training_uuid = all_uuids[0]
        on_training_gpu = instance.wcomm.allgather(
            (instance.wrank, gpu_uuid == training_uuid)
        )
        abandoned_ranks = [r for r, is_on in on_training_gpu if is_on]
        instance.TRAINING_WRANK = abandoned_ranks.pop()
        abandoned_assignments = {
            r: i % len(all_uuids) for i, r in enumerate(abandoned_ranks)
        }
        inference_color_mappings = {uuid: i for i, uuid in enumerate(all_uuids)}
        if instance.wrank == instance.TRAINING_WRANK:
            color = SMPI.UNDEFINED
        elif gpu_uuid != training_uuid:
            color = inference_color_mappings[gpu_uuid]
        else:
            color = abandoned_assignments[instance.wrank]
        instance.ncomm = instance.wcomm.Split(color, instance.wrank)
        instance.INFERENCE_NRANK = 0
        if instance.ncomm != SMPI.COMM_NULL:
            instance.nrank = instance.ncomm.Get_rank()
            instance.nsize = instance.ncomm.Get_size()
            color = SMPI.UNDEFINED if instance.INFERENCE_NRANK else 0
            non_inference_comm = instance.ncomm.Split(color, instance.nrank)
            if non_inference_comm != SMPI.COMM_NULL:
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

            if instance.nrank == instance.INFERENCE_NRANK:
                instance.process = "inference"
                device_index = cuda_index_from_uuid(gpu_uuid)
                instance.device = torch.device(f"cuda:{device_index}")
            else:
                instance.process = "worker"

        if instance.wrank == instance.TRAINING_WRANK:
            instance.process = "training"
            device_index = cuda_index_from_uuid(training_uuid)
            instance.device = torch.device(f"cuda:{device_index}")

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
            training_name = SMPI.Get_processor_name()
            if num_gpus == 0:
                raise ValueError("No GPUS! Cannot continue.")
            if num_gpus >= 2:
                self.device = torch.device(f"cuda:{num_gpus - 1}")
            else:
                self.device = torch.device("cuda:0")
        else:
            training_name = "NON_TRAINING_NODE"

        training_name = self.wcomm.bcast(training_name, self.TRAINING_WRANK)

        name = SMPI.Get_processor_name()
        if name == training_name:
            num_gpus = max(num_gpus - 1, 1)

        return num_gpus

    def _split_training(self) -> SMPI.Comm:
        """Split away training rank."""
        color = SMPI.UNDEFINED if self.wrank == self.TRAINING_WRANK else 0
        return self.wcomm.Split(color, self.wrank)

    def _split_node(
        self, avail_comm: SMPI.Comm, num_gpus: int
    ) -> tuple[SMPI.Comm, int | None, int | None, torch.device | None]:
        """Split by shared memory resources."""
        if avail_comm != SMPI.COMM_NULL:
            tempcomm = avail_comm.Split_type(SMPI.COMM_TYPE_SHARED)
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
            comm = SMPI.COMM_NULL
            rank = None
            size = None
            device = self.device
        return comm, rank, size, device

    def _split_observer(
        self, avail_comm: SMPI.Comm, agents_per_observer: int
    ) -> tuple[SMPI.Comm, int | None, int | None]:
        """Split communicator into observer comms."""
        if avail_comm != SMPI.COMM_NULL:
            rank = avail_comm.Get_rank()
            color = SMPI.UNDEFINED if rank == self.INFERENCE_NRANK else 0
            tempcomm = avail_comm.Split(color, rank)
            if tempcomm != SMPI.COMM_NULL:
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
        comm = SMPI.COMM_NULL
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
        self, wcomm: SMPI.Comm, cnn_wranks: torch.Tensor
    ) -> tuple[SMPI.Comm, int | None, int | None, int | None]:
        """Weight sending comm setup."""
        cnn_color = SMPI.UNDEFINED if self.wrank not in cnn_wranks else 0
        comm = wcomm.Split(cnn_color, self.wrank)
        if comm != SMPI.COMM_NULL:
            rank = comm.Get_rank()
            size = comm.Get_size()

            # Broadcast training rank in the cnn comm to all
            cnn_training_rank = wcomm.bcast(rank, self.TRAINING_WRANK)
        else:
            comm = SMPI.COMM_NULL
            rank = None
            size = None
            cnn_training_rank = None
            cnn_training_rank = wcomm.bcast(rank, self.TRAINING_WRANK)
        return comm, rank, size, cnn_training_rank

    @classmethod
    @gracefull_error
    def local(cls, num_gpus: int, agents_per_observer: int) -> MPIHandler:
        """!!! DEPRECATED !!!.

        Will be replaced by slurm_test soon.

        Generate MPIHandler for local execution.
        """
        logger.warning("Local standard execution be replaced by slurm test soon.")

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
        self.wcomm: SMPI.Comm = SMPI.COMM_WORLD
        self.wrank: int = self.wcomm.Get_rank()
        self.wsize: int = self.wcomm.Get_size()

        # Initialize to empty comms
        self.ncomm: SMPI.Comm = SMPI.COMM_NULL
        self.nrank: int | None = None
        self.nsize: int | None = None
        self.ocomm: SMPI.Comm = SMPI.COMM_NULL
        self.orank: int | None = None
        self.osize: int | None = None
        self.cnncomm: SMPI.Comm = SMPI.COMM_NULL
        self.cnnrank: int | None = None
        self.cnnsize: int | None = None

        self.device: torch.device | None = None
        self.process: str = "UNNASSIGNED"

        self.cnn_training_rank: int | None = None
        self._init = True

        # Need to hold the data for non blocking operations
        self.send_queue: Queue[MessageItem] = Queue()

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

    def send_exit(self) -> list[SMPI.Request]:
        """Send exit message to all ranks."""
        requests: list[SMPI.Request] = []
        empty_msg = np.empty(0, dtype="b")
        for target in range(self.wsize):
            req: SMPI.Request = self.wcomm.Isend(
                [empty_msg, SMPI.BYTE], target, int(Tag.EXIT)
            )
            requests.append(req)
            item = MessageItem(req, empty_msg, int(Tag.EXIT))
            self.send_queue.add(item)
        return requests

    def probe_exit(self) -> bool:
        """Check if an exit message is sent."""
        return self.wcomm.Iprobe(SMPI.ANY_SOURCE, int(Tag.EXIT))

    def recv_exit(self) -> None:
        """Recieve an exit message."""
        empty_msg = np.empty(0, dtype="b")
        self.wcomm.Recv([empty_msg, SMPI.BYTE], SMPI.ANY_SOURCE, int(Tag.EXIT))

    def finalize(self) -> None:
        """Gracefull finalize MPI Environment."""
        while self.send_queue.poll():
            sleep(1)
        SMPI.Finalize()

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
                (
                    "states",
                    Constants.state_tensor_batch_dtype,
                    Constants.state_tensor_batch_shape,
                ),
                ("values", Constants.value_batch_dtype, Constants.value_batch_shape),
                (
                    "move_tensor",
                    Constants.move_tensor_batch_dtype,
                    Constants.move_tensor_batch_shape,
                ),
            ],
        )
        message["states"] = np_states
        message["values"] = np_values
        message["move_tensor"] = np_move_tensors
        req = self.wcomm.Isend(
            [message, SMPI.BYTE], self.TRAINING_WRANK, int(Tag.EPISODE)
        )
        item = MessageItem(req, message, int(Tag.EPISODE))
        self.send_queue.add(item)

    def probe_episode(self, source: int) -> bool:
        """Probe if episode was sent from rank."""
        return self.wcomm.Iprobe(source, int(Tag.EPISODE))

    def recv_episode(
        self, source: int
    ) -> tuple[SMPI.Request, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Recieve the episode data fom the training Rank.

        !!! Carefull: Before copy operations await data. !!!
        Using a future apporach as it allows for gathering without inhibiting.
        """
        message: np.ndarray = np.empty(
            1,
            dtype=[
                (
                    "states",
                    Constants.state_tensor_batch_dtype,
                    Constants.state_tensor_batch_shape,
                ),
                ("values", Constants.value_batch_dtype, Constants.value_batch_shape),
                (
                    "move_tensor",
                    Constants.move_tensor_batch_dtype,
                    Constants.move_tensor_batch_shape,
                ),
            ],
        )
        req: SMPI.Request = self.wcomm.Irecv(
            [message, SMPI.BYTE], source, int(Tag.EPISODE)
        )
        _ = self.send_queue.poll()
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
            req = self.cnncomm.Isend(
                [empty_msg, SMPI.BYTE], int(target), int(Tag.NOTIFY_WEIGHTS)
            )
            item = MessageItem(req, empty_msg, int(Tag.NOTIFY_WEIGHTS))
            self.send_queue.add(item)

    def probe_weights_ready(self) -> bool:
        """Check if a notification of weights ready exists."""
        return self.wcomm.Iprobe(0, int(Tag.NOTIFY_WEIGHTS))

    def recv_weights_ready(self) -> None:
        """Send data to all processes that weights are ready."""
        empty_msg = np.empty(0, dtype="b")
        self.wcomm.Irecv([empty_msg, SMPI.BYTE], 0, int(Tag.NOTIFY_WEIGHTS))

    def _bcast_size(self, size: int) -> int:
        """Blocking size distribution to allow for non blocking."""
        size = self.cnncomm.bcast(size, self.cnn_training_rank)
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
        req: SMPI.Request = self.cnncomm.Ibcast(
            [buffer, SMPI.BYTE],
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
        self.ocomm.Reduce(move_numpy, move_numpy_sum, SMPI.SUM)
        return torch.from_numpy(move_numpy_sum)

    def reduce_values(self, value: float) -> float | None:
        """Reduce values from all ocomm."""
        logger.debug("Reducing values.")
        red_value: float | None = self.ocomm.reduce(
            value, SMPI.SUM, self.OBSERVER_ORANK
        )
        return red_value

    # --- Leaves from Agents to Inference ---
    def send_leaf_batch(self, leaf_batch: torch.Tensor) -> int:
        """Send Leaf batch to inference, returning a unique Token."""
        np_leaves = leaf_batch.numpy()
        message, dtype = self.token.tokenize_data(["data"], [np_leaves])
        req = self.ncomm.Isend(
            [message, dtype], self.INFERENCE_NRANK, int(Tag.LEAF_BATCH)
        )
        item = MessageItem(req, message, int(Tag.LEAF_BATCH))
        self.send_queue.add(item)
        return int(message["token"])

    def probe_leaf_batch(self, nsource: int) -> bool:
        """Check if leaf batch from nsource exists."""
        return self.ncomm.Iprobe(nsource, int(Tag.LEAF_BATCH))

    def recv_leaf_batch(
        self, nsource: int
    ) -> tuple[SMPI.Request, int, int | np.ndarray, torch.Tensor]:
        """Recieve Leaf batch and nsource from evaluation."""
        recv_spec: FieldSpec = (
            "data",
            Constants.leaf_batch_dtype,
            Constants.leaf_batch_shape,
        )
        message, dtype = self.token.empty_message([recv_spec])
        req: SMPI.Request = self.ncomm.Irecv(
            [message, dtype], nsource, int(Tag.LEAF_BATCH)
        )
        _ = self.send_queue.poll()
        return (
            req,
            nsource,
            message["token"],
            torch.from_numpy(message["data"].squeeze(0)),
        )

    # --- Values from Inference to Agents ---
    def send_value_batch(
        self,
        ntarget: int,
        token: int,
        values: torch.Tensor,
        move_tensors: torch.Tensor,
    ) -> None:
        """Send value batch to target nrank."""
        np_values = values.numpy()
        np_move_tensors = move_tensors.numpy()
        recv_spec_value: FieldSpec = (
            "values",
            Constants.value_batch_dtype,
            Constants.value_batch_shape,
        )
        recv_spec_moves: FieldSpec = (
            "move_tensors",
            Constants.move_tensor_batch_dtype,
            Constants.move_tensor_batch_shape,
        )
        message, dtype = self.token.empty_message([recv_spec_value, recv_spec_moves])
        message["token"] = token
        message["values"] = np_values
        message["move_tensors"] = np_move_tensors
        req = self.ncomm.Isend([message, dtype], ntarget, int(Tag.VALUE_BATCH))
        item = MessageItem(req, message, int(Tag.VALUE_BATCH))
        self.send_queue.add(item)

    def probe_value_batch(self) -> bool:
        """Check if a value batch."""
        return self.ncomm.Iprobe(self.INFERENCE_NRANK, int(Tag.VALUE_BATCH))

    def recv_value_batch(self) -> tuple[int, torch.Tensor, torch.Tensor]:
        """Recieve Value and move tensor batch from Agent on nrank."""
        recv_spec_value: FieldSpec = (
            "values",
            Constants.value_batch_dtype,
            Constants.value_batch_shape,
        )
        recv_spec_moves: FieldSpec = (
            "move_tensors",
            Constants.move_tensor_batch_dtype,
            Constants.move_tensor_batch_shape,
        )
        message, dtype = self.token.empty_message([recv_spec_value, recv_spec_moves])
        self.ncomm.Recv([message, dtype], self.INFERENCE_NRANK, int(Tag.VALUE_BATCH))
        _ = self.send_queue.poll()
        return (
            int(message["token"]),
            torch.from_numpy(message["values"].squeeze(0)),
            torch.from_numpy(message["move_tensors"].squeeze(0)),
        )
