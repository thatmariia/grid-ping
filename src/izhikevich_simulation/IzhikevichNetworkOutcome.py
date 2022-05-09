import numpy as np


class IzhikevichNetworkOutcome:
    """
    This class contains the collected information from the simulation of the Izhikevich network.

    :param spikes: indices of spikes.
    :type spikes: list[tuple[int, int]]

    :param potentials: potentials of neurons throughout the simulation.
    :type potentials: list[numpy.ndarray[int, int]]

    :ivar spikes: indices of spikes.
    :ivar potentials: potentials of neurons throughout the simulation.
    """

    def __init__(self, spikes: list[tuple[int, int]], potentials: list[np.ndarray[int, int]]):
        self.spikes: list[tuple[int, int]] = spikes
        self.potentials: list[np.ndarray[int, int]] = potentials