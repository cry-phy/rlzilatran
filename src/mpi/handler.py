from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dynaconf import Dynaconf

from enum import IntEnum, auto
from game.utils import Board
from functools import wraps
from mpi4py import MPI
from time import time
import numpy as np
import traceback
import logging
import torch
import sys


logger = logging.getLogger(__name__)


class Tag(IntEnum):
    """Make tags easier to understand."""

    EXIT = auto()
    EPISODE_STATES = auto()
    EPISODE_VALUES = auto()
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
            dtype=[
                ("token", "i8"),
                ("data", "f4", *data_shape),
            ]
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

    def __new__(cls, config: Dynaconf | None = None) -> "MPIHandler":
        """Singleton Instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config: Dynaconf | None = None) -> None:
        """Initialize MPI Comms.

        Global rank 0:      Training process
        Node rank 0:        Inference process
        Observer rank 0:    Observer process
        Observer comm is needed for BCast and AllReduce of the move suggestions.
        For reference please see resources/Processes.png.
        """
        # Guard against multiple initializeations due to MPIHandler calls in class blocks.
        if self._init:
            return

        if config is not None:
            self.TRAINING_WRANK = config.mpi.TRAINING_WRANK
            self.INFERENCE_NRANK = config.mpi.INFERENCE_NRANK
            self.OBSERVER_ORANK = config.mpi.OBSERVER_ORANK
            self.MPI_SHUTDOWN_WAIT = config.mpi.MPI_SHUTDOWN_WAIT
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

        # Node comm setup
        training_color = MPI.UNDEFINED if self._wrank == self.TRAINING_WRANK else 0
        avail_comm = self.wcomm.Split(training_color, self._wrank)
        if avail_comm != MPI.COMM_NULL:
            self.ncomm: MPI.Comm = avail_comm.Split_type(MPI.COMM_TYPE_SHARED)
            self._nrank = self.ncomm.Get_rank()
            self._nsize = self.ncomm.Get_size()

            # Observer-Agent comm setup
            inference_color = (
                MPI.UNDEFINED if self._nrank == self.INFERENCE_NRANK else 0
            )
            self.ocomm = self.ncomm.Split(inference_color, self._nrank)
            if self.ocomm != MPI.COMM_NULL:
                self._orank = self.ocomm.Get_rank()
                self._osize = self.ocomm.Get_size()

        # Get world comm ranks of inference ranks
        is_inference = self._nrank == self.INFERENCE_NRANK
        logger.debug(f"Rank {self._wrank} is inference: {is_inference}")
        are_inference = self.wcomm.allgather(is_inference)
        self.inference_wranks = torch.where(
            torch.tensor(are_inference, dtype=torch.bool)
        )

        # Weight sending comm setup
        cnn_color = (
            0
            if self._wrank in self.inference_wranks
            or self._wrank == self.TRAINING_WRANK
            else MPI.UNDEFINED
        )
        self.cnncomm = self.wcomm.Split(cnn_color, self._wrank)
        if self.cnncomm != MPI.COMM_NULL:
            self._cnnrank = self.cnncomm.Get_rank()
            self._cnnsize = self.cnncomm.Get_size()

            # Broadcast training rank in the cnn comm to all
            self._cnn_training_rank = self.wcomm.bcast(
                self._cnnrank, self.TRAINING_WRANK
            )

        if self._wrank == 0:
            logger.info(
                f"Started MPI with wcomm with {self._wsize} processes, inference on {self.inference_wranks}"
            )
        self._init = True

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

    # Send Episode Data from Observer to Training
    def send_episode(
        self, states: torch.Tensor, values: torch.Tensor, move_tensor: torch.Tensor
    ) -> None:
        """Send the episode data to the training Rank."""
        np_states = states.numpy()
        np_values = values.numpy()
        np_move_tensors = move_tensor.numpy()
        message: np.ndarray = np.empty(
            dtype=[
                ("states", "f4", Board.state_tensor_batch_shape),
                ("values", "f4", Board.value_batch_shape),
                ("move_tensor", "f4", Board.move_tensor_batch_shape),
            ]
        )
        message["states"] = np_states
        message["values"] = np_values
        message["move_tensors"] = np_move_tensors
        self.wcomm.Isend(
            [message, MPI.BYTE], self.TRAINING_WRANK, int(Tag.EPISODE_STATES)
        )

    def probe_episode(self, source: int) -> bool:
        """Probe if episode was sent from rank."""
        return self.wcomm.Iprobe(source, int(Tag.EPISODE_STATES)) and self.wcomm.iprobe(
            source, int(Tag.EPISODE_VALUES)
        )

    def recv_episode(
        self, source: int
    ) -> tuple[MPI.Request, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Recieve the episode data fom the training Rank.

        !!! Carefull: Before copy operations await data. !!!
        Using a future apporach as it allows for gathering without inhibiting.
        """
        message: np.ndarray = np.empty(
            dtype=[
                ("states", "f4", Board.state_tensor_batch_shape),
                ("values", "f4", Board.value_batch_shape),
                ("move_tensor", "f4", Board.move_tensor_batch_shape),
            ]
        )
        req: MPI.Request = self.wcomm.Irecv(
            [message, MPI.BYTE], source, int(Tag.EPISODE_STATES)
        )
        return (
            req,
            torch.from_numpy(message["states"]),
            torch.from_numpy(message["values"]),
            torch.from_numpy(message["move_tensors"]),
        )

    # -- Distribute Weights from Training to Inference
    def send_weights_ready(self) -> None:
        """Send data to all processes that weights are ready."""
        empty_msg = np.empty(0, dtype="b")
        for target in self.inference_wranks:
            self.wcomm.Isend([empty_msg, MPI.BYTE], target, int(Tag.NOTIFY_WEIGHTS))

    def probe_weights_ready(self) -> bool:
        """Check if a notification of weights ready exists."""
        return self.wcomm.Iprobe(0, int(Tag.NOTIFY_WEIGHTS))

    def recv_weights_ready(self) -> None:
        """Send data to all processes that weights are ready."""
        empty_msg = np.empty(0, dtype="b")
        self.wcomm.Irecv([empty_msg, MPI.BYTE], 0, int(Tag.NOTIFY_WEIGHTS))

    def bcast_weights(self, flat_weights: np.ndarray, size: int) -> np.ndarray:
        """Send Weights to Inference Process.

        The weights are flattened and need to be reassembled by the CNN.
        In the future it might be helpfull to switch to torch.distribution to allow
        for NCCL direct GPU-GPU interaction.
        """
        logger.debug("Broadcasting weights.")
        if not self._wrank == self.TRAINING_WRANK:
            flat_weights = np.empty(size)
        req: MPI.Request = self.cnncomm.Ibcast(
            [flat_weights, MPI.FLOAT],
            self._cnn_training_rank,
        )
        if not self._wrank == self.TRAINING_WRANK:
            req.Wait()
        return flat_weights

    # -- Synchronize Observer and Agents
    def bcast_decided_move(
        self, move: tuple[int, int, int, int, int] | None
    ) -> tuple[int, int, int, int, int]:
        """Broadcast new game state to all agents."""
        logger.debug("Broadcasting decided move.")
        if self._orank == self.OBSERVER_ORANK:
            if move is None:
                raise ValueError("Attempting to Bcast None from self.OBSERVER_ORANK.")
        else:
            if move is not None:
                raise ValueError("Attempting to send data on Agent Rank.")
        return self.ocomm.bcast(move, self.OBSERVER_ORANK)

    def reduce_move_tensors(self, move_tensor: torch.Tensor) -> torch.Tensor:
        """Gather all move suggestion tensors from agents."""
        logger.debug("Reducing move Tensors")
        move_numpy_sum = np.empty(Board.move_tensor_shape, dtype=np.float32)
        move_numpy = move_tensor.numpy()
        self.ocomm.Reduce(move_numpy, move_numpy_sum, MPI.SUM)
        return torch.from_numpy(move_numpy_sum)

    def reduce_values(self, value: float) -> float | None:
        """Reduce values from all ocomm."""
        logger.debug("Reducing values.")
        red_value: float | None = self.ocomm.reduce(value, MPI.SUM, self.OBSERVER_ORANK)
        return red_value

    # -- Leaves from Agents to Inference --
    def send_leaf_batch(self, leaf_batch: torch.Tensor) -> int:
        """Send Leaf batch to inference, returning a unique Token."""
        np_leaves = leaf_batch.numpy()
        message = self.token.tokenize_data(np_leaves)
        self.ncomm.Isend([message, MPI.BYTE], self.INFERENCE_NRANK, int(Tag.LEAF_BATCH))
        return int(message["token"])

    def probe_leaf_batch(self, nsource: int) -> bool:
        """Check if leaf batch from nsource exists."""
        return self.ncomm.Iprobe(nsource, int(Tag.LEAF_BATCH))

    def recv_leaf_batch(
        self, nsource: int
    ) -> tuple[MPI.Request, int, int, torch.Tensor]:
        """Recieve Leaf batch and nsource from evaluation."""
        message = self.token.empty_message(Board.leaf_batch_shape)
        req: MPI.Request = self.ncomm.Irecv(message, nsource, int(Tag.LEAF_BATCH))
        return req, nsource, int(message["token"]), torch.from_numpy(message["data"])

    # -- Values from Inference to Agents --
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
            dtype=[
                ("token", "i8"),
                ("values", "f4", *Board.value_batch_shape),
                ("move_tensors", "f4", *Board.move_tensor_batch_shape),
            ]
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
            dtype=[
                ("token", "i8"),
                ("values", "f4", *Board.value_batch_shape),
                ("move_tensors", "f4", *Board.move_tensor_batch_shape),
            ]
        )
        self.ncomm.Recv([message, MPI.BYTE], self.INFERENCE_NRANK, int(Tag.VALUE_BATCH))
        return (
            int(message["token"]),
            torch.from_numpy(message["values"]),
            torch.from_numpy(message["move_tensors"]),
        )


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
            logger.info("In function %s: \n %s", func.__name__, tb_str)
            value = None
            send_gracefull_exit()
        return value

    return wrapper
