from __future__ import annotations
from time import monotonic
from typing import TYPE_CHECKING


from config import Settings
from cnn.loader import get_episode_loader, get_leaf_loader
from cnn.queue import EpisodeItem, LeafItem, Queue
from cnn.small import SmallCNN
from cnn.trainer import Trainer
from game.utils import Constants
from mpi.handler import MPIHandler, gracefull_error
from cnn.inferrer import Inferrer

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


def get_process(config: Settings, handler: MPIHandler) -> MPIProcess:
    """Return the MPIProcess that is supposed to be run on this Rank."""
    logger.debug("Getting processes.")

    if handler.is_training():
        return TrainingProcess(config, handler)
    elif handler.is_inference():
        return InferenceProcess(config, handler)
    if handler.orank == handler.OBSERVER_ORANK:
        if config.obs.GUI:
            return UIObserverProcess(config)
        else:
            return ObserverProcess(config)
    else:
        return AgentProcess(config)


class MPIProcess:
    """Base Class of for Process execution.

    Allows for uniform starting of all Processes as MPIHandler returns
    an Instance of the appropriate MPIProcess, which only has to be run after.
    """

    def __init__(self) -> None:
        """Initialize MPI handler here.

        doing it in the class block provides some issues at import time.
        """
        logger.info("Initialized %s.", self.__class__.__name__)

    def run(self) -> None:
        """Run the Process."""
        raise NotImplementedError

    def main_step(self) -> None:
        """Run a single step in the process loop."""
        raise NotImplementedError


class TrainingProcess(MPIProcess):
    """Initialize process running batching and training CNN."""

    def __init__(self, config: Settings, handler: MPIHandler) -> None:
        """Initialize CNN Training."""
        super().__init__()
        self.queue = Queue()
        self.dataset, self.dataloader = get_episode_loader(
            self.queue, config.cnn.super_batch_size, config.cnn.replay_buffer_size
        )
        self.cnn = SmallCNN.create(
            handler.get_device(),
            128,
            Constants.state_tensor_shape,
            Constants.move_tensor_shape,
        )
        self.trainer = Trainer.from_config(self.cnn, config)
        self.mpi = handler
        self.episode_batch_num = config.cnn.episode_batch_num

    def run(self) -> None:
        """Run the Process."""
        while True:
            self.main_step()

    @gracefull_error
    def main_step(self) -> None:
        """Run a single step in the process loop."""
        for source in range(self.mpi.wsize):
            if self.mpi.probe_episode(source):
                req, states, values, move_tensor = self.mpi.recv_episode(source)
                item = EpisodeItem(req, states, values, move_tensor)
                self.queue.add(item)
                logger.info("Recieved Episode data from %i", source)

        if self.queue.num_ready() >= self.episode_batch_num:
            logger.info(
                "Starting Training with %i.",
                self.episode_batch_num + self.dataset.replay_buffer_len(),
            )
            self.trainer.train(self.dataloader)
            self.mpi.send_weights_ready()
            self.mpi.bcast_weights(self.cnn)


class InferenceProcess(MPIProcess):
    """Initialize process running Inference of the CNN."""

    def __init__(self, config: Settings, handler: MPIHandler) -> None:
        """Initialize CNN inference."""
        super().__init__()
        self.queue = Queue()
        self.dataset, self.dataloader = get_leaf_loader(
            self.queue, config.cnn.super_batch_size
        )
        self.cnn = SmallCNN.create(
            handler.get_device(),
            128,
            Constants.state_tensor_shape,
            Constants.move_tensor_shape,
        )
        self.inferrer = Inferrer(self.cnn, self.cnn.device)
        self.mpi = handler
        self.leaf_batch_num = config.cnn.leaf_batch_num

        self._last_inference = monotonic()
        self._inference_timing = config.cnn.inference_timing

    def run(self) -> None:
        """Run the Process."""
        while True:
            self.main_step()

    @gracefull_error
    def main_step(self) -> None:
        """Run a single step in the process loop."""
        if self.mpi.probe_weights_ready():
            logger.info("Recieving Weights.")
            self.mpi.recv_weights_ready()
            self.mpi.bcast_weights(self.cnn)
            logger.info("Recieved Weights.")

        for nrank in range(cast(int, self.mpi.nsize)):
            if self.mpi.probe_leaf_batch(nrank):
                req, source, token, data = self.mpi.recv_leaf_batch(nrank)
                item = LeafItem(request=req, data=data, source=source, token=token)
                logger.debug("Recieved leaf data from %s", source)
                self.queue.add(item)

        current = monotonic()
        should_run_inference = (
            self.queue.num_ready() >= self.leaf_batch_num
            or self._last_inference + self._inference_timing <= current
        )
        if should_run_inference:
            logger.debug("Starting inference batch processing via Inferrer.")
            for source, token, moves, values in self.inferrer.infer_and_chunk(
                self.dataloader
            ):
                self.mpi.send_value_batch(
                    ntarget=source,
                    token=token,
                    values=values,
                    move_tensors=moves,
                )
            self._last_inference = current
            logger.debug("Completed inference batch and sent results.")


class AgentProcess(MPIProcess):
    """Initialize Process acting only as Observer."""

    def __init__(self, config: Settings) -> None:
        """Set Agent environment and recieve agents and Game."""
        super().__init__()
        game: Game = Game.new_game()
        agent_cls1: type[BaseAgent] = config.agent_class_dict[config.obs.agent1]
        agent_cls2: type[BaseAgent] = config.agent_class_dict[config.obs.agent2]
        self.observer = None
        self.game = game
        self.obs = BaseObserver.any_vs_any(game, agent_cls1, agent_cls2)
        self.random_prep_moves = config.obs.random_prep_moves

    def run(self) -> None:
        """Loop the control flow indefenitely."""
        while True:
            self.obs.make_random_moves(self.random_prep_moves)
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
        self.num_moves = 0
        logger.debug("Awaiting Move")
        bcast_move = current_agent.bcast_move(move)
        current_agent.make_move(*bcast_move)
        logger.info("Move Executed: %s", bcast_move)


class ObserverProcess(MPIProcess):
    """Initialize Process acting as Observer.

    This is part of the ocomm and therefore also runs Agent search.
    """

    def __init__(self, config: Settings) -> None:
        """Set up observer and distribute game anda agents."""
        super().__init__()
        game: Game = Game.new_game()
        agent_cls1: type[BaseAgent] = config.agent_class_dict[config.obs.agent1]
        agent_cls2: type[BaseAgent] = config.agent_class_dict[config.obs.agent2]
        self.obs = BaseObserver.any_vs_any(game, agent_cls1, agent_cls2)
        self.random_prep_moves = config.obs.random_prep_moves

    def run(self) -> None:
        """Loop the control flow indefenitely."""
        while True:
            self.obs.make_random_moves(self.random_prep_moves)
            while not self.obs.game.game_over():
                self.main_step()
                self._send_episode()
            logger.info("Game over.")
            self.obs.reset()

    def main_step(self) -> None:
        """Observer rank Control Flow."""
        state_tensor = self.obs.game.state_tensor
        current_agent = self.obs.agents[self.obs.game.state.player]
        move_tensor, value = self.generate_move_tensor(current_agent)
        self.decide_make_move(current_agent, move_tensor)
        self.obs.add_data(state_tensor, value, move_tensor)

    def _send_episode(self) -> None:
        """Send episode batch to Training."""
        if self.obs.batch_ready():
            logger.info("Sending Episode.")
            self.obs.send_episode()
            self.obs.reset_data()

    @gracefull_error
    def generate_move_tensor(self, agent: BaseAgent) -> tuple[torch.Tensor, float]:
        """Centralize move tensor generation."""
        agent.generate_move_tensor()
        logger.debug("Generated Move Tensor.")
        move_tensor = agent.reduce_move_tensors()
        value = cast(float, agent.reduce_values())
        return move_tensor, value

    @gracefull_error
    def decide_make_move(self, agent: BaseAgent, move_tensor: torch.Tensor) -> None:
        """Centralize move decision and making."""
        move = agent.generate_move(move_tensor)
        logger.debug("Sending Move.")
        bcast_move = agent.bcast_move(move)
        agent.make_move(*bcast_move)
        logger.info("Move Executed: %s", bcast_move)


class UIObserverProcess(ObserverProcess):
    """Observer process able to handle User Interface.

    !!! Do not run on Cluster since main loop is not properly secured against crashes.
    """

    def __init__(self, config: Settings) -> None:
        """Set up observer and distribute game and agents."""
        logger.info("Initialized %s.", self.__class__.__name__)
        game: Game = Game.new_game()
        agent_cls1: type[BaseAgent] = config.agent_class_dict[config.obs.agent1]
        agent_cls2: type[BaseAgent] = config.agent_class_dict[config.obs.agent2]
        self.obs: UIObserver = cast(
            UIObserver, UIObserver.any_vs_any(game, agent_cls1, agent_cls2)
        )

    def run(self) -> None:
        """Loop the control flow indefenitely."""
        self.obs.make_random_moves(self.random_prep_moves)
        self.main_step()
        self.obs.root.mainloop()

    def main_step(self) -> None:
        """Observer rank Control Flow."""
        state_tensor = self.obs.game.state_tensor
        current_agent: BaseAgent = self.obs.agents[self.obs.game.state.player]

        if current_agent.clickable:
            self.obs.pre_clickable_input()

        if current_agent.move_ready:
            move_tensor, value = self.generate_move_tensor(current_agent)
            self.obs.draw_move_tensor(move_tensor)
            self.decide_make_move(current_agent, move_tensor)
            self.obs.redraw_board()
            self.obs.add_data(state_tensor, value, move_tensor)
            self._send_episode()
            self.num_moves += 1

            if current_agent.clickable:
                self.obs.post_clickable_input()

        if not self.obs.game.game_over():
            self.obs.wait(self.main_step)
        else:
            logger.info("Game over. Total moves: %i", self.num_moves)
            self.obs.reset()
            self.obs.make_random_moves(self.random_prep_moves)
            self.obs.wait(self.main_step)
