from enum import IntEnum, auto
from game.utils import Board
from mpi.processes import MPIProcess
from mpi4py import MPI
import torch
import numpy as np


class Tag(IntEnum):
    """Make tags easier to understand."""

    EPISODE_STATES = auto()
    EPISODE_VALUES = auto()
    NOTIFY_WEIGHTS = auto()
    LEAF_BATCH = auto()
    VALUE_BATCH = auto()


# The following 3 are all on different Communicators
TRAINING_WRANK = 0
INFERENCE_NRANK = 0
OBSERVER_ORANK = 0


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

    def empty_message(self, data_shape: tuple):
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

    def __new__(cls) -> "MPIHandler":
        """Singleton Instance."""
        if cls._instance is None:
            cls._instance = cls._instance.__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize MPI Comms.

        Global rank 0:      Training process
        Node rank 0:        Inference process
        Observer rank 0:    Observer process
        Observer comm is needed for BCast and AllReduce of the move suggestions.
        For reference please see resources/Processes.png.
        """
        # World comm setup
        self.wcomm: MPI.COMM = MPI.COMM_WORLD
        self._wrank: int = self.wcomm.Get_rank()
        self._wsize: int = self.wcomm.Get_size()

        # Node comm setup
        training_color = MPI.UNDEFINED if self._wrank == TRAINING_WRANK else 0
        avail_comm = self.wcomm.Split(training_color, self._wrank)
        self.ncomm: MPI.COMM = avail_comm.Split_type(MPI.MPI_COMM_TYPE_SHARED)
        self._nrank: int = self.ncomm.Get_rank()
        self._nsize: int = self.ncomm.Get_size()

        # Observer-Agent comm setup
        inference_color = MPI.UNDEFINED if self._wrank == OBSERVER_ORANK else 0
        self.ocomm = self.ncomm.Split(inference_color, self._nrank)
        self._orank: int = self.ocomm.Get_rank()
        self._osize: int = self.ocomm.Get_size()

        # Get world comm ranks of inference ranks
        if self._wrank == TRAINING_WRANK:
            is_inference_process = self.wcomm.allgather(self._nrank == INFERENCE_NRANK)
            self.inference_wranks = torch.where(torch.BoolTensor(is_inference_process))
        else:
            self.wcomm.allgather(self._nrank == INFERENCE_NRANK)
            self.inference_wranks = []

        self.inference_wranks = self.wcomm.bcast(self.inference_wranks, TRAINING_WRANK)

        # Weight sending comm setup
        cnn_color = (
            0
            if self._wrank in self.inference_wranks or self._wrank == TRAINING_WRANK
            else MPI.UNDEFINED
        )
        self.cnncomm = self.wcomm.Split(cnn_color, self._wrank)
        self._cnnrank: int = self.cnncomm.Get_rank()
        self._cnnsize: int = self.cnncomm.Get_size()

        # Broadcast training rank in the cnn comm to all
        self._cnn_training_rank = self.wcomm.bcast(self._cnnrank, TRAINING_WRANK)

        self.OBSERVER_ORANK = OBSERVER_ORANK

    def send_episode(self, states: torch.Tensor, values: torch.Tensor) -> None:
        """Send the episode data to the training Rank."""
        np_states = states.numpy()
        np_values = values.numpy()
        self.wcomm.Isend(
            [np_states, MPI.FLOAT], TRAINING_WRANK, int(Tag.EPISODE_STATES)
        )
        self.wcomm.Isend(
            [np_values, MPI.FLOAT], TRAINING_WRANK, int(Tag.EPISODE_VALUES)
        )

    def probe_episode(self, source: int) -> bool:
        """Probe if episode was sent from rank."""
        return self.wcomm.Iprobe(source, int(Tag.EPISODE_STATES)) and self.wcomm.iprobe(
            source, int(Tag.EPISODE_VALUES)
        )

    def recv_episode(
        self, source: int
    ) -> tuple[tuple[MPI.Request, torch.Tensor], torch.Tensor]:
        """Recieve the episode data fom the training Rank.

        !!! Carefull: Before copy operations await data. !!!
        Using a future apporach as it allows for gathering without inhibiting.
        """
        states: np.ndarray = np.empty(Board.state_tensor_shape)
        values: np.ndarray = np.empty(Board.move_tensor_shape)
        state_req: MPI.Request = self.wcomm.Irecv(
            [states, MPI.FLOAT], source, int(Tag.EPISODE_STATES)
        )
        value_req: MPI.Request = self.wcomm.Irecv(
            [values, MPI.FLOAT], source, int(Tag.EPISODE_VALUES)
        )
        return (
            (state_req, torch.from_numpy(states)),
            (value_req, torch.from_numpy(values)),
        )

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
        if not self._wrank == TRAINING_WRANK:
            flat_weights = np.empty(size)
        req: MPI.Request = self.cnncomm.Ibcast(
            [flat_weights, MPI.FLOAT],
            self._cnn_training_rank,
        )
        if not self._wrank == TRAINING_WRANK:
            req.Wait()
        return flat_weights

    def bcast_decided_move(
        self, move: tuple[int, int, int, int]
    ) -> tuple[int, int, int, int]:
        """Broadcast new game state to all agents."""
        return self.ocomm.bcast(move, OBSERVER_ORANK)

    def reduce_move_tensors(self, move_tensor: torch.Tensor) -> torch.Tensor:
        """Gather all move suggestion tensors from agents."""
        move_numpy_sum = np.empty(Board.move_tensor_shape)
        move_numpy = move_tensor.numpy()
        self.ocomm.AllReduce(move_numpy, move_numpy_sum, MPI.SUM)
        return torch.from_numpy(move_numpy_sum)

    def send_leaf_batch(self, leaf_batch: torch.Tensor) -> int:
        """Send Leaf batch to inference, returning a unique Token."""
        np_leaves = leaf_batch.numpy()
        message = self.token.tokenize_data(np_leaves)
        self.ncomm.Isend([message, MPI.BYTE], INFERENCE_NRANK, int(Tag.LEAF_BATCH))
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

    def send_value_batch(self, ntarget: int, values: torch.Tensor) -> int:
        """Send value batch to target nrank."""
        np_values = values.numpy()
        message = self.token.tokenize_data(np_values)
        self.ncomm.Isend([message, MPI.BYTE], ntarget, int(Tag.VALUE_BATCH))
        return int(message["token"])

    def probe_value_batch(self, nsource: int) -> bool:
        """Check if a value batch."""
        return self.ncomm.Iprobe(nsource, int(Tag.VALUE_BATCH))

    def recv_value_batch(self) -> tuple[int, torch.Tensor]:
        """Recieve Value batch from Agent on nrank."""
        message = self.token.empty_message(Board.value_batch_shape)
        self.ncomm.Recv([message, MPI.BYTE], INFERENCE_NRANK, int(Tag.VALUE_BATCH))
        return int(message["token"]), torch.from_numpy(message["data"])

    def get_process(self) -> MPIProcess:
        """Return the MPIProcess that is supposed to be run on this Rank."""
        raise NotImplementedError
