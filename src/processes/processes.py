from __future__ import annotations
from typing import TYPE_CHECKING

from dynaconf import Dynaconf

from src.cnn.loader import get_leaf_loader
from src.cnn.queue import Queue
from src.cnn.small import SmallCNN
import numpy as np
from src.game.utils.constants import Constants
from src.mpi.handler import MPIHandler

if TYPE_CHECKING:
    from agents import BaseAgent

from typing import cast

import torch
from game import Game
from observer import BaseObserver, UIObserver

import logging

logger: logging.Logger = logging.getLogger(__name__)


class ProcessError(Exception):
    """Exeption handling errors with processes."""


def get_process(config: Dynaconf, handler: MPIHandler) -> MPIProcess:
    """Return the MPIProcess that is supposed to be run on this Rank."""
    logger.debug("Getting processes.")
    if handler._wrank == handler.TRAINING_WRANK:
        return TrainingProcess()
    elif handler._nrank == handler.INFERENCE_NRANK:
        return InferenceProcess(config, handler)
    game: Game = Game.new_game()
    agent_cls1: type[BaseAgent] = config.agent_class_dict[config.obs.agent1]
    agent_cls2: type[BaseAgent] = config.agent_class_dict[config.obs.agent2]
    if handler._orank == handler.OBSERVER_ORANK:
        if config.obs.GUI:
            return UIObserverProcess(
                game=game,
                agent_cls1=agent_cls1,
                agent_cls2=agent_cls2,
            )
        else:
            return ObserverProcess(
                game=game,
                agent_cls1=agent_cls1,
                agent_cls2=agent_cls2,
            )
    else:
        return AgentProcess(
            game=game,
            agent_cls1=agent_cls1,
            agent_cls2=agent_cls2,
        )


class MPIProcess:
    """Base Class of for Process execution.

    Allows for uniform starting of all Processes as MPIHandler returns
    an Instance of the appropriate MPIProcess, which only has to be run after.
    """

    def __init__(self) -> None:
        """Initialize MPI handler here.

        doing it in the class block provides some issues at import time.
        """
        logger.debug("Initialized %s on rank %i", self.__class__.__name__)

    def run(self) -> None:
        """Run the Process."""
        raise NotImplementedError

    def main_step(self) -> None:
        """Run a single step in the process loop."""
        raise NotImplementedError


class TrainingProcess(MPIProcess):
    """Initialize process running batching and training CNN."""

    # Setup CNN

    # While True:
    #   if new episode:
    #       irecv new epsiode
    #       add episode to DataLoader
    #   if DataLoader enough data:
    #       CNN training
    #       for all processes in cnncomm
    #           isend notify new weights
    #       ibcast new weights
    def run(self) -> None:
        """Run the Process."""
        while True:
            self.main_step()

    def main_step(self) -> None:
        """Run a single step in the process loop."""
        ...


class InferenceProcess(MPIProcess):
    """Initialize process running Inference of the CNN."""

    def __init__(self, config: Dynaconf, handler: MPIHandler) -> None:
        """Initialize CNN inference."""
        super().__init__()
        self.queue = Queue(Constants.BATCH_SIZE)
        self.dataset, self.dataloader = get_leaf_loader(
            self.queue, Constants.BATCH_SIZE
        )
        self.cnn = SmallCNN(
            128,
            Constants.state_tensor_shape,
            Constants.move_tensor_shape,
        )
        self.mpi = handler
        self.super_batch_size = config.super_batch_size
        self.leaf_batch_num = config.leaf_batch_num

    def run(self) -> None:
        """Run the Process."""
        while True:
            self.main_step()

    def main_step(self) -> None:
        """Run a single step in the process loop."""
        if self.mpi.probe_weights_ready():
            self.mpi.recv_weights_ready()
            flat_weights, size = np.empty(0), 0
            self.mpi.bcast_weights(flat_weights, size)
            self.cnn.from_flat(flat_weights)

        for nrank in range(cast(int, self.mpi._nsize)):
            if self.mpi.probe_leaf_batch(nrank):
                leaf_data = self.mpi.recv_leaf_batch(nrank)
                self.queue.add(leaf_data)

        if self.queue.num_ready() >= self.leaf_batch_num:
            moves = []
            values = []
            for data in self.dataloader():
                moves, values = self.cnn.forward(data)
                moves.append(moves)
                values.append(values)
            self.cnn.forward()
            pass


class AgentProcess(MPIProcess):
    """Initialize Process acting only as Observer."""

    def __init__(
        self,
        game: Game,
        agent_cls1: type[BaseAgent],
        agent_cls2: type[BaseAgent],
    ) -> None:
        """Set Agent environment and recieve agents and Game."""
        super().__init__()
        self.observer = None
        self.game = game
        self.obs = BaseObserver.any_vs_any(game, agent_cls1, agent_cls2)

    def run(self) -> None:
        """Loop the control flow indefenitely."""
        while True:
            while not self.game.game_over():
                self.main_step()
            self.obs.reset()

    def main_step(self) -> None:
        """Agent Control Flow."""
        current_agent = self.obs.current_agent()
        current_agent.generate_move_tensor()
        logger.debug("Generated Move Tensor.")
        _ = current_agent.reduce_move_tensors()
        _ = current_agent.reduce_values()
        move = None
        logger.debug("Awaiting Move")
        bcast_move = current_agent.bcast_move(move)
        current_agent.make_move(*bcast_move)
        logger.info("Move Executed: %s", bcast_move)


class ObserverProcess(MPIProcess):
    """Initialize Process acting as Observer.

    This is part of the ocomm and therefore also runs Agent search.
    """

    def __init__(
        self,
        game: Game,
        agent_cls1: type[BaseAgent],
        agent_cls2: type[BaseAgent],
    ) -> None:
        """Set up observer and distribute game anda agents."""
        super().__init__()
        self.obs = BaseObserver.any_vs_any(game, agent_cls1, agent_cls2)

    def run(self) -> None:
        """Loop the control flow indefenitely."""
        while True:
            while not self.obs.game.game_over():
                self.main_step()
                self._send_episode()
            logger.info("Game over.")
            self.obs.reset()

    def main_step(self) -> None:
        """Observer rank Control Flow."""
        state_tensor = self.obs.game.state_tensor
        current_agent = self.obs.agents[self.obs.game.state.player]
        current_agent.generate_move_tensor()
        logger.debug("Generated Move")
        sum_move_tensor = current_agent.reduce_move_tensors()
        sum_value = cast(float, current_agent.reduce_values())
        move_tensor = sum_move_tensor
        move = current_agent.generate_move(move_tensor)
        logger.debug("Sending Move.")
        bcast_move = current_agent.bcast_move(move)
        current_agent.make_move(*bcast_move)
        logger.info("Move Executed: %s", bcast_move)
        self.obs.add_data(state_tensor, sum_value, move_tensor)

    def _send_episode(self) -> None:
        """Send episode batch to Training."""
        if self.obs.batch_ready():
            self.obs.send_epsiode()
            self.obs.reset_data()


class UIObserverProcess(MPIProcess):
    """Observer process able to handle User Interface.

    !!! Do not run on Cluster since main loop is not secured against crashes.
    """

    def __init__(
        self,
        game: Game,
        agent_cls1: type[BaseAgent],
        agent_cls2: type[BaseAgent],
    ) -> None:
        """Set up observer and distribute game anda agents."""
        super().__init__()
        self.obs: UIObserver = cast(
            UIObserver, UIObserver.any_vs_any(game, agent_cls1, agent_cls2)
        )

    def run(self) -> None:
        """Loop the control flow indefenitely."""
        self.main_step()
        self.obs.root.mainloop()

    def main_step(self) -> None:
        """Observer rank Control Flow."""
        state_tensor = self.obs.game.state_tensor
        current_agent: BaseAgent = self.obs.agents[self.obs.game.state.player]

        if current_agent.clickable:
            self.obs.pre_clickable_input()

        if current_agent.move_ready:
            current_agent.generate_move_tensor()
            logger.debug("Generated Move")
            sum_move_tensor = current_agent.reduce_move_tensors()
            sum_value = cast(float, current_agent.reduce_values())
            move_tensor = sum_move_tensor
            torch.save(move_tensor, "temp/move_tensor1.txt")
            self.obs.draw_move_tensor(move_tensor)
            move = current_agent.generate_move(move_tensor)
            logger.debug("Sending Move.")
            bcast_move = current_agent.bcast_move(move)
            logger.debug("Decided Move %s.", bcast_move)
            current_agent.make_move(*bcast_move)
            logger.info("Move Executed: %s", bcast_move)
            self.obs.redraw_board()
            self.obs.add_data(state_tensor, sum_value, move_tensor)

            if current_agent.clickable:
                self.obs.post_clickable_input()

        if not self.obs.game.game_over():
            self.obs.wait(self.main_step)
        else:
            logger.info("Game over.")
            self.obs.reset()
            self.obs.wait(self.main_step)
