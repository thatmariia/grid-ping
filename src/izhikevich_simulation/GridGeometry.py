from src.izhikevich_simulation.PINGNetworkNeurons import *
from src.params.NeuronTypes import *

import numpy as np


class GridGeometry:
    """
    This class contains information about grid locations of PING networks and neurons located in them.

    :param ping_networks: list of PING networks.
    :type ping_networks: list[PINGNetworkNeurons]

    :param neuron_distances: distances between neurons in the visual cortex.
    :type neuron_distances: numpy.ndarray[(int, int), float]

    :ivar ping_networks: list of PING networks.
    :ivar neuron_distances: distances between neurons in the visual cortex.
    """

    def __init__(
            self,
            ping_networks: list[PINGNetworkNeurons],
            neuron_distances: np.ndarray[(int, int), float]
    ):

        self.ping_networks: list[PINGNetworkNeurons] = ping_networks
        self.neuron_distances: np.ndarray[(int, int), float] = neuron_distances

    def __eq__(self, other):
        return np.all(self.ping_networks == other.ping_networks) and np.all(self.neuron_distances == other.neuron_distances)

