from enum import IntEnum, auto
from game.utils import Board
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
TRAINING_RANK = 0
INFERENCE_RANK = 0
OBSERVER_RANK = 0


class MPIHandler:
    """MPI Handler singleton."""

    _instance = None

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
        training_color = MPI.UNDEFINED if self._wrank == TRAINING_RANK else 0
        avail_comm = self.wcomm.Split(training_color, self._wrank)
        self.ncomm: MPI.COMM = avail_comm.Split_type(MPI.MPI_COMM_TYPE_SHARED)
        self._nrank: int = self.ncomm.Get_rank()
        self._nsize: int = self.ncomm.Get_size()

        # Observer-Agent comm setup
        inference_color = MPI.UNDEFINED if self._wrank == OBSERVER_RANK else 0
        self.ocomm = self.ncomm.Split(inference_color, self._nrank)
        self._nrank: int = self.ocomm.Get_rank()
        self._nsize: int = self.ocomm.Get_size()

        # Get world comm ranks of inference ranks
        if self._wrank == TRAINING_RANK:
            is_inference_process = self.wcomm.allgather(self._nrank == INFERENCE_RANK)
            self.inference_wranks = torch.where(torch.BoolTensor(is_inference_process))
        else:
            self.wcomm.allgather(self._nrank == INFERENCE_RANK)
            self.inference_wranks = []

        self.inference_wranks = self.wcomm.bcast(self.inference_wranks, TRAINING_RANK)

        # Weight sending comm setup
        cnn_color = (
            0
            if self._wrank in self.inference_wranks or self._wrank == TRAINING_RANK
            else MPI.UNDEFINED
        )
        self.cnncomm = self.wcomm.Split(cnn_color, self._wrank)
        self._cnnrank: int = self.cnncomm.Get_rank()
        self._cnnsize: int = self.cnncomm.Get_size()

        # Broadcast training rank in the cnn comm to all
        self._cnn_training_rank = self.wcomm.bcast(self._cnnrank, TRAINING_RANK)

    def send_episode(self, states: torch.Tensor, values: torch.Tensor) -> None:
        """Send the episode data to the training Rank."""
        np_states = states.numpy()
        np_values = values.numpy()
        self.wcomm.Isend([np_states, MPI.FLOAT], TRAINING_RANK, int(Tag.EPISODE_STATES))
        self.wcomm.Isend([np_values, MPI.FLOAT], TRAINING_RANK, int(Tag.EPISODE_VALUES))

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
        if not self._wrank == TRAINING_RANK:
            flat_weights = np.empty(size)
        req: MPI.Request = self.cnncomm.Ibcast(
            [flat_weights, MPI.FLOAT], self._cnn_training_rank
        )
        if not self._wrank == TRAINING_RANK:
            req.Wait()
        return flat_weights

    def bcast_decided_move(self, move: tuple[int, int, int, int]) -> None:
        """Broadcast new game state to all agents."""
        return self.ocomm.bcast(move, OBSERVER_RANK)

    def reduce_move_tensors(self, move_tensor: torch.Tensor) -> torch.Tensor:
        """Gather all move suggestion tensors from agents."""
        move_numpy_sum = np.empty(Board.move_tensor_shape)
        move_numpy = move_tensor.numpy()
        self.ocomm.AllReduce(move_numpy, move_numpy_sum, MPI.SUM)
        return torch.from_numpy(move_numpy_sum)

    def send_leaf_batch(self, leaf_batch: torch.Tensor) -> None:
        """Send Leaf batch to inference."""
        np_leaves = leaf_batch.numpy()
        self.ncomm.Isend(np_leaves, INFERENCE_RANK, int(Tag.LEAF_BATCH))

    def probe_leaf_batch(self, nsource: int) -> bool:
        """Check if leaf batch from nsource exists."""
        return self.ncomm.Iprobe(nsource, int(Tag.LEAF_BATCH))

    def recv_leaf_batch(self, nsource: int) -> tuple[int, MPI.Request, torch.Tensor]:
        """Recieve Leaf batch and nsource from evaluation."""
        np_leaves = np.empty(Board.leaf_batch_shape)
        req: MPI.Request = self.ncomm.Irecv(np_leaves, nsource, int(Tag.LEAF_BATCH))
        return nsource, req, torch.from_numpy(np_leaves)

    def send_value_batch(self, ntarget: int, values: torch.Tensor) -> None:
        """Send value batch to target nrank."""
        np_values = values.numpy()
        self.ncomm.Isend(np_values, ntarget, int(Tag.VALUE_BATCH))

    def probe_value_batch(self, nsource: int) -> bool:
        """Check if a value batch."""
        return self.ncomm.Iprobe(nsource, int(Tag.VALUE_BATCH))

    def recv_value_batch(self) -> torch.Tensor:
        """Recieve Value batch from Agent on nrank."""
        np_values = np.empty(Board.value_batch_shape)
        self.ncomm.Recv(np_values, INFERENCE_RANK, int(Tag.VALUE_BATCH))

    def get_process(self) -> "MPIProcess":
        """Return the MPIProcess that is supposed to be run on this Rank."""
        raise NotImplementedError


class MPIProcess:
    """Base Class of for Process execution.

    Allows for uniform starting of all Processes as MPIHandler returns
    an Instance of the appropriate MPIProcess, which only has to be run after.
    """

    def run(self) -> None:
        """Run the Process."""
        raise NotImplementedError


class TrainingProcess(MPIProcess):
    """Initialize process runng batching and training CNN."""


class InferenceProcess(MPIProcess):
    """Initialize process running Inference of the CNN."""


class AgentProcess(MPIProcess):
    """Initialize Process acting only as Observer."""


class ObserverProcess(MPIProcess):
    """Initialize Process acting as Observer.

    This is part of the ocomm and therefore also runs Agent search.
    """
