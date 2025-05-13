import logging
from typing import cast

import torch
from config import get_config
from game.game import Game
from mpi.handler import MPIHandler, gracefull_error
from observer.ui import BaseObserver, UIObserver
from agents.base import BaseAgent

logger: logging.Logger = logging.getLogger(__name__)


class ProcessError(Exception):
    """Exeption handling errors with processes."""


def get_process() -> "MPIProcess":
    """Return the MPIProcess that is supposed to be run on this Rank."""
    logger.debug("Getting processes.")
    handler = MPIHandler()
    config = get_config()
    if handler._wrank == handler.TRAINING_WRANK:
        return TrainingProcess()
    elif handler._nrank == handler.INFERENCE_NRANK:
        return InferenceProcess()
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
        self.mpi = MPIHandler()
        logger.debug(
            "Initialized %s on rank %i", self.__class__.__name__, self.mpi._wrank
        )

    def run(self) -> None:
        """Run the Process."""
        while True:
            self.main_step()

    @gracefull_error
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

    @gracefull_error
    def main_step(self) -> None:
        """Run a single step in the process loop."""
        ...


class InferenceProcess(MPIProcess):
    """Initialize process running Inference of the CNN."""

    # Setup CNN

    # While True:
    #   if notify new weights exist:
    #       recv notify new weights
    #       ibcast new weights
    #   if leaf batch:
    #       irecv leaf batch
    #       add leaf batch to DataLoader
    #   if DataLoader enough data:
    #       Inference on DataLoader
    #       for value batch in infered data:
    #           isend value batch

    @gracefull_error
    def main_step(self) -> None:
        """Run a single step in the process loop."""
        ...


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
        temp_obs = BaseObserver.any_vs_any(game, agent_cls1, agent_cls2)
        self.agents: tuple[BaseAgent, BaseAgent] = temp_obs.agents

    def run(self) -> None:
        """Loop the control flow indefenitely."""
        while not self.game.game_over():
            self.main_step()

    @gracefull_error
    def main_step(self) -> None:
        """Agent Control Flow."""
        current_agent = self.agents[self.game.state.player]
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
        if self.mpi._osize is not None:
            self.osize = self.mpi._osize
        else:
            raise ProcessError("Observer on bad rank.")

    def run(self) -> None:
        """Loop the control flow indefenitely."""
        while not self.obs.game.game_over():
            self.main_step()
            self._send_episode()

    @gracefull_error
    def main_step(self) -> None:
        """Observer rank Control Flow."""
        state_tensor = self.obs.game.state_tensor
        current_agent = self.obs.agents[self.obs.game.state.player]
        current_agent.generate_move_tensor()
        logger.debug("Generated Move")
        sum_move_tensor = current_agent.reduce_move_tensors()
        sum_value = current_agent.reduce_values()
        move_tensor = sum_move_tensor / self.osize
        if sum_value is None:
            raise ValueError("Value should not be None on OBSERVER_ORANK")
        else:
            value = sum_value / self.osize
        move = current_agent.generate_move(move_tensor)
        logger.debug("Sending Move.")
        bcast_move = current_agent.bcast_move(move)
        current_agent.make_move(*bcast_move)
        logger.info("Move Executed: %s", bcast_move)
        self.obs.add_data(state_tensor, value, move_tensor)

    @gracefull_error
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
        if self.mpi._osize is not None:
            self.osize = self.mpi._osize
        else:
            raise ProcessError("Observer on bad rank.")

    def run(self) -> None:
        """Loop the control flow indefenitely."""
        self.main_step()
        self.obs.root.mainloop()

    @gracefull_error
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
            sum_value = current_agent.reduce_values()
            move_tensor = sum_move_tensor / self.osize
            torch.save(move_tensor, "temp/move_tensor1.txt")
            self.obs.draw_move_tensor(move_tensor)
            if sum_value is None:
                raise ValueError("Value should not be None on OBSERVER_ORANK")
            else:
                value = sum_value / self.osize
            move = current_agent.generate_move(move_tensor)
            logger.debug("Sending Move.")
            bcast_move = current_agent.bcast_move(move)
            current_agent.make_move(*bcast_move)
            logger.info("Move Executed: %s", bcast_move)
            self.obs.redraw_board()
            self.obs.add_data(state_tensor, value, move_tensor)

            if current_agent.clickable:
                self.obs.post_clickable_input()

        if not self.obs.game.game_over():
            self.obs.wait(self.main_step)
        else:
            logger.info("Game over.")
