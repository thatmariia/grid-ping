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
            ping_networks,
            neuron_distances
    ):

        self.ping_networks: list[PINGNetworkNeurons] = ping_networks
        self.neuron_distances: np.ndarray[(int, int), float] = neuron_distances

