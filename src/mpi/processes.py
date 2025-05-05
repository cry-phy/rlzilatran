from agents.human import HumanAgent
from mpi.handler import MPIHandler
from observer.ui import UIObserver


class MPIProcess:
    """Base Class of for Process execution.

    Allows for uniform starting of all Processes as MPIHandler returns
    an Instance of the appropriate MPIProcess, which only has to be run after.
    """

    mpi: MPIHandler = MPIHandler()

    def run(self) -> None:
        """Run the Process."""
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


class AgentProcess(MPIProcess):
    """Initialize Process acting only as Observer."""

    def __init__(self) -> None:
        """Set Agent environment and recieve agents and Game."""
        super().__init__()
        self.observer = None

        # Observer bcast the Agents on ocomm
        self.game, self.agents = self.mpi.bcast_observer_agents(self.observer)

    def run(self) -> None:
        """Loop the control flow indefenitely."""
        super().run()
        while not self.game.game_over():
            current_agent = self.agents[self.game.state.player]
            current_agent.generate_move_tensor()
            _ = current_agent.reduce_move_tensors()
            move = None
            bcast_move = current_agent.bcast_move(move)
            current_agent.make_move(*bcast_move)


class ObserverProcess(MPIProcess):
    """Initialize Process acting as Observer.

    This is part of the ocomm and therefore also runs Agent search.
    """

    def __init__(self) -> None:
        """Set up observer and distribute game anda agents."""
        super().__init__()
        self.observer = UIObserver.any_vs_any(HumanAgent, HumanAgent, (0, 1))

        # Observer bcast the Agents on ocomm
        self.game, self.agents = self.mpi.bcast_observer_agents(self.observer)

    def run(self) -> None:
        """Loop the control flow indefenitely."""
        super().run()
        while not self.game.game_over():
            state_tensor = self.game.state_tensor()
            current_agent = self.agents[self.game.state.player]
            current_agent.generate_move_tensor()
            sum_move_tensor = current_agent.reduce_move_tensors()
            sum_value = current_agent.reduce_values()
            move_tensor = sum_move_tensor / self.mpi._osize
            if sum_value is None:
                raise ValueError("Value should not be None on OBSERVER_ORANK")
            else:
                value = sum_value / self.mpi._osize
            move = current_agent.generate_move(move_tensor)
            bcast_move = current_agent.bcast_move(move)
            current_agent.make_move(*bcast_move)
            self.observer.add_data(state_tensor, value, move_tensor)
            if self.observer.batch_ready():
                self.observer.send_epsiode()
                self.observer.reset_data()
