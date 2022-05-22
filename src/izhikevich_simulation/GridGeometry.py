from src.izhikevich_simulation.PINGNetworkNeurons import *
from src.params.NeuronTypes import *

import numpy as np


class GridGeometry:
    """
    This class contains information about grid locations of PING networks and neurons located in them.

    :param ping_networks: list of PING networks.
    :type ping_networks: list[PINGNetworkNeurons]

    :param neuron_ping_map: dictionary mapping a neuron to the PING network it belongs to.
    :type neuron_ping_map: dict[NeuronTypes, dict[int, int]]
    """

    def __init__(
            self,
            ping_networks: list[PINGNetworkNeurons],
            neuron_locations: np.ndarray[int, int]
    ):

        self.ping_networks: list[PINGNetworkNeurons] = ping_networks
        self.neuron_locations: np.ndarray[int, int] = neuron_locations

