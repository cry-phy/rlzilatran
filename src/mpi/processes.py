class MPIProcess:
    """Base Class of for Process execution.

    Allows for uniform starting of all Processes as MPIHandler returns
    an Instance of the appropriate MPIProcess, which only has to be run after.
    """

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

    # Observer bcast the Agents on ocomm

    # While game != done:
    #   Current Agents: get move tensor
    #   Current Agents: reduce move tensor
    #   Current Agents: bcast move tuple
    #   Current Agents: make move
    #   Switch Agents


class ObserverProcess(MPIProcess):
    """Initialize Process acting as Observer.

    This is part of the ocomm and therefore also runs Agent search.
    """

    # Observer setup

    # Observer bcast the Agents on ocomm

    # While game != done:
    #   Current Agents: get move tensor:
    #       while searching:
    #           rollout
    #           leaf batch append leaf
    #           if leaf batch enough:
    #               isend leaf batch
    #           while value batch exist:
    #               recv value batch
    #           Adjust Tree weights
    #       Return value, move tensor
    #   Current Agents: reduce move tensor
    #   On Observer: Current Agent: generate move
    #   Current Agents: bcast move tuple
    #   On Observer: episode append state, move_tensor, value
    #   Current Agents: make move
    #   Switch Agents
    # ISend Episode
