from typing import Any
from src.NeuronTypes import *

import numpy as np

class Connectivity:
    """
    This class contains information about connectivity between neurons in the oscillatory network.

    The interaction strength of lateral connections is represented by a matrix :math:`K` of pairwise coupling weights
    defined by an exponential function decaying by the Euclidean distance between the PING networks they belong to:

    :math:`K_{v, w} = C_{ \mathsf{type}(v), \mathsf{type}(w)} \exp (-\| \mathsf{cmap}(v), \mathsf{cmap}(w) \| / s_{v, w}),`

    where
    * :math:`v, w` are two arbitrary neurons in the network,
    * :math:`\mathsf{type}(v)` maps a neuron to its type (see :obj:`NeuronTypes`),
    * :math:`\mathsf{cmap}(v)` maps a neuron to its location in the visual cortex (see :obj:`StimulusLocations`),
    * :math:`s_{v, w}` is the spatial constant (see :obj:`constants.SPATIAL_CONST`).

    This equation was introduced in :cite:p:`Izhikevich2003`.

    This class performs the assignment of neurons to relevant PING networks arranged in a grid and computes the matrix
    of coupling weights.

    :param nr_neurons: dictionary of number of neurons of each type and the total number of neurons.
    :type nr_neurons: dict[Any, int]

    :param neur_slice: indices of each type of neurons.
    :type neur_slice: dict[NeuronTypes, slice]

    :param nr_ping_networks: number of PING networks.
    :type nr_ping_networks: int

    :param coupling_weights: coupling weights between all pairs of neurons.
    :type coupling_weights: numpy.ndarray[(int, int), float]


    :ivar nr_neurons: dictionary of number of neurons of each type and the total number of neurons.
    :ivar neur_slice: indices of each type of neurons.
    :ivar nr_ping_networks: number of PING networks.
    :ivar coupling_weights: coupling weights between all pairs of neurons.
    """

    def __init__(
            self, nr_neurons: dict[Any, int], neur_slice: dict[NeuronTypes, slice], nr_ping_networks: int,
            coupling_weights: np.ndarray[(int, int), float]
    ):
        self.nr_neurons: dict[Any, int] = nr_neurons
        self.neur_slice: dict[NeuronTypes, slice] = neur_slice
        self.nr_ping_networks: int = nr_ping_networks
        self.coupling_weights: np.ndarray[(int, int), float] = coupling_weights

