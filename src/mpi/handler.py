from __future__ import annotations
from typing import TYPE_CHECKING

from game.utils.constants import Constants
from cnn import CNN

if TYPE_CHECKING:
    from config import Settings

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
    while requests and time() - start_time <= mpi.config.MPI_SHUTDOWN_WAIT:
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

    def __init__(self) -> None:
        """Set token starting value to 0."""
        self.__current_id = 0

    def new(self) -> np.ndarray:
        """Generate a new token."""
        self.__current_id += 1
        return np.full(1, self.__current_id, dtype="i8")

    def empty_message(self, data_shape: tuple) -> np.ndarray:
        """Generate an empty message buffer."""
        message: np.ndarray = np.empty(
            1,
            dtype=[
                ("token", "i8"),
                ("data", "f4", *data_shape),
            ],
        )
        return message

    def tokenize_data(self, data: np.ndarray) -> np.ndarray:
        """Copy Data into structure array and add a token."""
        message = self.empty_message(data.shape)
        message["token"] = self.new()
        message["data"] = data
        return message


class MPIHandler:
    """MPI Handler singleton."""

    _instance = None
    token = TokenHandler()
    _init = False

    def __new__(cls, config: Settings | None = None) -> MPIHandler:
        """Singleton Instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config: Settings | None = None) -> None:
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

        if config is not None:
            self.config = config.mpi
        else:
            raise ValueError(
                "Config must be passed to the first time MPIHandler is created."
            )

        # World comm setup
        self.wcomm: MPI.Comm = MPI.COMM_WORLD
        self._wrank: int = self.wcomm.Get_rank()
        self._wsize: int = self.wcomm.Get_size()

        # Initialize to empty comms
        self.ncomm: MPI.Comm = MPI.COMM_NULL
        self._nrank: int | None = None
        self._nsize: int | None = None
        self.ocomm: MPI.Comm = MPI.COMM_NULL
        self._orank: int | None = None
        self._osize: int | None = None
        self.cnncomm: MPI.Comm = MPI.COMM_NULL
        self._cnnrank: int | None = None
        self._cnnsize: int | None = None

        self.device: torch.device | None = None

        # Node comm setup
        training_color = (
            MPI.UNDEFINED if self._wrank == self.config.TRAINING_WRANK else 0
        )
        if self._wrank == self.config.TRAINING_WRANK:
            training_name = MPI.Get_processor_name()
            if self.config.NUM_GPUS == 0:
                raise ValueError("No GPUS! Cannot continue.")
            if self.config.NUM_GPUS >= 2:
                self.device = torch.device(f"cuda:{self.config.NUM_GPUS - 1}")
            else:
                self.device = torch.device("cuda:0")
        else:
            training_name = "NON_TRAINING_NODE"

        name = MPI.Get_processor_name()
        if name == training_name:
            self.config.NUM_GPUS = self.config.NUM_GPUS - 1

        training_name = self.wcomm.bcast(training_name, self.config.TRAINING_WRANK)
        avail_comm = self.wcomm.Split(training_color, self._wrank)
        if avail_comm != MPI.COMM_NULL:
            avail_ncomm: MPI.Comm = avail_comm.Split_type(MPI.COMM_TYPE_SHARED)
            avail_size = avail_ncomm.Get_size()
            avail_rank = avail_ncomm.Get_rank()

            # Group by GPU
            num_per_gpu = avail_size // self.config.NUM_GPUS
            bp_rank = avail_size % self.config.NUM_GPUS
            ncomm_color = (
                avail_rank // (num_per_gpu + 1)
                if avail_rank <= bp_rank
                else avail_rank // num_per_gpu
            )
            self.device = torch.device(f"cuda:{ncomm_color}")
            self.ncomm: MPI.Comm = avail_ncomm.Split(ncomm_color, avail_rank)
            self._nrank = self.ncomm.Get_rank()
            self._nsize = self.ncomm.Get_size()

            # Observer-Agent comm setup
            color = MPI.UNDEFINED if self._nrank == self.config.INFERENCE_NRANK else 0
            avail_comm = self.ncomm.Split(color, self._nrank)
            if avail_comm != MPI.COMM_NULL:
                processes_per_gpu = avail_comm.Get_size()
                gpu_rank = avail_comm.Get_rank()
                number_observers = processes_per_gpu // self.config.AGENTS_PER_OBSERVER
                bp_rank = processes_per_gpu % number_observers
                observer_color = (
                    gpu_rank // (self.config.AGENTS_PER_OBSERVER + 1)
                    if gpu_rank <= bp_rank
                    else gpu_rank // self.config.AGENTS_PER_OBSERVER
                )
                self.ocomm = avail_comm.Split(observer_color, self._nrank)
                if self.ocomm != MPI.COMM_NULL:
                    self._orank = self.ocomm.Get_rank()
                    self._osize = self.ocomm.Get_size()

        # Get world comm ranks of inference ranks
        is_cnn = (
            self._nrank == self.config.INFERENCE_NRANK
            or self._wrank == self.config.TRAINING_WRANK
        )
        if is_cnn:
            logger.info(
                f"Rank {self._wrank} is using {torch.cuda.get_device_name(self.device)} for CNN."
            )
        are_cnn = self.wcomm.allgather(is_cnn)
        self.wranks = torch.where(torch.tensor(are_cnn, dtype=torch.bool))[0]

        # Weight sending comm setup
        cnn_color = MPI.UNDEFINED if self._wrank not in self.wranks else 0
        self.cnncomm = self.wcomm.Split(cnn_color, self._wrank)
        if self.cnncomm != MPI.COMM_NULL:
            self._cnnrank = self.cnncomm.Get_rank()
            self._cnnsize = self.cnncomm.Get_size()

            # Broadcast training rank in the cnn comm to all
            self._cnn_training_rank = self.wcomm.bcast(
                self._cnnrank, self.config.TRAINING_WRANK
            )

        if self._wrank == 0:
            logger.info(
                f"Started MPI with wcomm with {self._wsize} processes, inference on {self.wranks}"
            )
        self._init = True

    @gracefull_error
    def get_device(self) -> torch.device:
        """Return the current device."""
        if self.device is None:
            raise ValueError("Cuda Device not set. Exiting.")
        return self.device

    def force_exit(self):
        """Force world comm abort."""
        self.wcomm.Abort(1)

    def send_exit(self) -> list[MPI.Request]:
        """Send exit message to all ranks."""
        requests: list[MPI.Request] = []
        empty_msg = np.empty(0, dtype="b")
        for target in range(self._wsize):
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
        self.wcomm.Isend(
            [message, MPI.BYTE], self.config.TRAINING_WRANK, int(Tag.EPISODE)
        )

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
        for target in self.wranks:
            self.wcomm.Isend(
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
        size = self.cnncomm.bcast(size, self.config.TRAINING_WRANK)
        return size

    def bcast_weights(self, model: CNN) -> None:
        """Send Weights to Inference Process.

        The weights are flattened and need to be reassembled by the CNN.
        In the future it might be helpfull to switch to torch.distribution to allow
        for NCCL direct GPU-GPU interaction.
        """
        logger.debug("Broadcasting weights.")
        if self._wrank == self.config.TRAINING_WRANK:
            bytestream: bytes = model.to_bytestream()
            buffer = memoryview(bytestream)
            _ = self._bcast_size(len(bytestream))
        else:
            size = self._bcast_size(0)
            buffer = bytearray(size)
        req: MPI.Request = self.cnncomm.Ibcast(
            [buffer, MPI.BYTE],
            self._cnn_training_rank,
        )
        if not self._wrank == self.config.TRAINING_WRANK:
            req.Wait()
            model.from_bytestream(bytes(buffer))

    # --- Synchronize Observer and Agents ---
    def bcast_decided_move(
        self, move: tuple[int, int, int, int, int] | None
    ) -> tuple[int, int, int, int, int]:
        """Broadcast new game state to all agents."""
        logger.debug("Broadcasting decided move.")
        if self._orank == self.config.OBSERVER_ORANK:
            if move is None:
                raise ValueError("Attempting to Bcast None from self.OBSERVER_ORANK.")
        else:
            if move is not None:
                raise ValueError("Attempting to send data on Agent Rank.")
        return self.ocomm.bcast(move, self.config.OBSERVER_ORANK)

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
        red_value: float | None = self.ocomm.reduce(
            value, MPI.SUM, self.config.OBSERVER_ORANK
        )
        return red_value

    # --- Leaves from Agents to Inference ---
    def send_leaf_batch(self, leaf_batch: torch.Tensor) -> int:
        """Send Leaf batch to inference, returning a unique Token."""
        np_leaves = leaf_batch.numpy()
        message = self.token.tokenize_data(np_leaves)
        self.ncomm.Isend(
            [message, MPI.BYTE], self.config.INFERENCE_NRANK, int(Tag.LEAF_BATCH)
        )
        return int(message["token"])

    def probe_leaf_batch(self, nsource: int) -> bool:
        """Check if leaf batch from nsource exists."""
        return self.ncomm.Iprobe(nsource, int(Tag.LEAF_BATCH))

    def recv_leaf_batch(
        self, nsource: int
    ) -> tuple[MPI.Request, int, int, torch.Tensor]:
        """Recieve Leaf batch and nsource from evaluation."""
        message = self.token.empty_message(Constants.leaf_batch_shape)
        req: MPI.Request = self.ncomm.Irecv(message, nsource, int(Tag.LEAF_BATCH))
        return req, nsource, int(message["token"]), torch.from_numpy(message["data"])

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
        message: np.ndarray = np.empty(
            Constants.BATCH_SIZE,
            dtype=[
                ("token", "i8"),
                ("values", "f4", *Constants.value_batch_shape),
                ("move_tensors", "f4", *Constants.move_tensor_batch_shape),
            ],
        )
        message["token"] = token
        message["values"] = np_values
        message["move_tensors"] = np_move_tensors
        self.ncomm.Isend([message, MPI.BYTE], ntarget, int(Tag.VALUE_BATCH))

    def probe_value_batch(self, nsource: int) -> bool:
        """Check if a value batch."""
        return self.ncomm.Iprobe(nsource, int(Tag.VALUE_BATCH))

    def recv_value_batch(self) -> tuple[int, torch.Tensor, torch.Tensor]:
        """Recieve Value and move tensor batch from Agent on nrank."""
        message: np.ndarray = np.empty(
            Constants.BATCH_SIZE,
            dtype=[
                ("token", "i8"),
                ("values", "f4", *Constants.value_batch_shape),
                ("move_tensors", "f4", *Constants.move_tensor_batch_shape),
            ],
        )
        self.ncomm.Recv(
            [message, MPI.BYTE], self.config.INFERENCE_NRANK, int(Tag.VALUE_BATCH)
        )
        return (
            int(message["token"]),
            torch.from_numpy(message["values"].squeeze(0)),
            torch.from_numpy(message["move_tensors"].squeeze(0)),
        )
