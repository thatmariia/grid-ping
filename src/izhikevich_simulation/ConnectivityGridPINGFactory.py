from src.params.ParamsConnectivity import *

from src.izhikevich_simulation.Connectivity import *
from src.izhikevich_simulation.GridGeometryFactory import *
from src.izhikevich_simulation.RingGeometryFactory import *
from src.izhikevich_simulation.GridGeometry import *

from itertools import product
import numpy as np
from tqdm import tqdm


class ConnectivityGridPINGFactory:
    """
    This class determines the connectivity between neurons in the oscillatory network.
    """

    def create(
            self, params_ping: ParamsPING, params_connectivity: ParamsConnectivity,
            cortical_distances: np.ndarray[(int, int), float]
    ) -> Connectivity:
        """
        Determines the connectivity between neurons in the oscillatory network.

        :param params_ping: parameters describing PING networks and their composition.
        :type params_ping: ParamsPING

        :param params_connectivity: parameters of the network connectivity.
        :type params_connectivity: ParamsConnectivity

        :param cortical_distances: distances between PING networks in the visual cortex.
        :type cortical_distances: numpy.ndarray[(int, int), float]

        :return: connectivity between neurons in the oscillatory network.
        :rtype: Connectivity
        """

        grid_geometry = GridGeometryFactory().create(params_ping, cortical_distances)
        #grid_geometry = RingGeometryFactory().create(params_ping)
        coupling_weights = self._compute_coupling_weights(
            params_ping, params_connectivity, grid_geometry
        )

        connectivity = Connectivity(
            params_ping=params_ping,
            coupling_weights=coupling_weights,
            grid_geometry=grid_geometry
        )

        return connectivity

    def _compute_coupling_weights(
            self,
            params_ping: ParamsPING, params_connectivity: ParamsConnectivity, grid_geometry: GridGeometry
    ) -> np.ndarray[(int, int), float]:
        """
        Computes the coupling weights between all neurons.

        Essentially, this method computes the full matrix :math:`K` of coupling weights.
        The approach is derived from :cite:p:`Lowet2015`.

        :param params_ping: parameters describing PING networks and their composition.
        :type params_ping: ParamsPING

        :param params_connectivity: parameters of the network connectivity.
        :type params_connectivity: ParamsConnectivity

        :param grid_geometry: contains information about grid locations of PING networks and neurons located in them.
        :type grid_geometry: GridGeometry

        :return: matrix of all coupling weights.
        :rtype: numpy.ndarray[(int, int), float]
        """

        coupling_weights = np.zeros((params_ping.nr_neurons["total"], params_ping.nr_neurons["total"]))

        for nts in (pbar := tqdm(list(product([NeuronTypes.EX, NeuronTypes.IN], repeat=2)))):
            pbar.set_description("Computing coupling weights")

            dist = grid_geometry.neuron_distances[params_ping.neur_slice[nts[0]], params_ping.neur_slice[nts[1]]]
            types_coupling_weights = self._compute_type_coupling_weights(
                dist=dist,
                max_connect_strength=params_connectivity.max_connect_strength[(nts[0], nts[1])],
                spatial_const=params_connectivity.spatial_const[(nts[0], nts[1])]
            )

            if nts[0] == nts[1]:
                coupling_weights[params_ping.neur_slice[nts[0]], params_ping.neur_slice[nts[1]]] = \
                    types_coupling_weights
            else:
                coupling_weights[params_ping.neur_slice[nts[1]], params_ping.neur_slice[nts[0]]] = \
                    types_coupling_weights.T

        return np.nan_to_num(coupling_weights)

    def _compute_type_coupling_weights(
            self,
            dist: np.ndarray[(int, int), float], max_connect_strength: float, spatial_const: float
    ) -> np.ndarray[(int, int), float]:
        """
        Computes the coupling weights for connections between two types of neurons.

        This method computes a matrix of :math:`K_{v, w}` between neurons :math:`v` and
        :math:`w` of given types.

        :param dist: distance matrix with pairwise distances between neurons.
        :type dist: numpy.ndarray[(int, int), float]

        :param max_connect_strength: max connection strength between neuron types.
        :type max_connect_strength: float

        :param spatial_const: spatial constant for the neuron types.
        :type spatial_const: float

        :return: the matrix of coupling weights of size nr1 x nr2, where n1 and nr2 - number of neurons of
            each neuron_type in the coupling of interest.
        :rtype: numpy.ndarray[(int, int), float]
        """

        coupling_weights_type = max_connect_strength * np.exp(np.true_divide(-np.array(dist), spatial_const))
        return coupling_weights_type