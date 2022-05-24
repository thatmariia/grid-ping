from src.params.ParamsPING import *
from src.izhikevich_simulation.GridGeometry import *

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
    * :math:`\mathsf{cmap}(v)` maps a neuron to its grid_location in the visual cortex (see :obj:`StimulusLocations`),
    * :math:`s_{v, w}` is the spatial constant (see :obj:`ParamsConnectivity`).

    The coupling weights are a part of current (see :obj:`CurrentComponents`).
    This equation was introduced in :cite:p:`Izhikevich2003`. This class performs the assignment of neurons to relevant PING networks arranged in a grid and computes the matrix
    of coupling weights.

    :param params_ping: parameters describing PING networks and their composition.
    :type params_ping: ParamsPING

    :param coupling_weights: coupling weights between all pairs of neurons.
    :type coupling_weights: numpy.ndarray[(int, int), float]

    :param grid_geometry: contains information about grid locations of PING networks and neurons located in them.
    :type grid_geometry: GridGeometry

    :ivar params_ping: parameters describing PING networks and their composition.
    :ivar coupling_weights: coupling weights between all pairs of neurons.
    :ivar grid_geometry: contains information about grid locations of PING networks and neurons located in them.
    """

    def __init__(
            self, params_ping: ParamsPING, coupling_weights: np.ndarray[(int, int), float],
            grid_geometry: GridGeometry = None
    ):
        self.params_ping: ParamsPING = params_ping
        self.coupling_weights: np.ndarray[(int, int), float] = coupling_weights
        self.grid_geometry = grid_geometry

    def __eq__(self, other):
        return (self.params_ping == other.params_ping) and \
               np.all(self.coupling_weights == other.coupling_weights) and \
               (self.grid_geometry == other.grid_geometry)
