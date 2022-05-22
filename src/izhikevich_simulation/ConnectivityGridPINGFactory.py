from src.params.ParamsConnectivity import *

from src.izhikevich_simulation.PINGNetworkNeurons import *
from src.izhikevich_simulation.Connectivity import *
from src.misc import *
from src.izhikevich_simulation.GridGeometryFactory import *
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
            cortical_distances: list[list[float]]
    ) -> Connectivity:
        """
        Determines the connectivity between neurons in the oscillatory network.

        :param params_ping: parameters describing PING networks and their composition.
        :type params_ping: ParamsPING

        :param params_connectivity: parameters of the network connectivity.
        :type params_connectivity: ParamsConnectivity

        :param cortical_distances: coordinates of the points in the visual cortex.
        :type cortical_distances: list[list[float]]

        :return: connectivity between neurons in the oscillatory network.
        :rtype: Connectivity
        """

        grid_geometry = GridGeometryFactory().create(params_ping)
        coupling_weights = self._compute_coupling_weights(
            params_ping, params_connectivity, grid_geometry, cortical_distances
        )

        connectivity = Connectivity(
            params_ping=params_ping,
            coupling_weights=coupling_weights,
            grid_geometry=grid_geometry
        )

        return connectivity

    def _compute_coupling_weights(
            self,
            params_ping: ParamsPING, params_connectivity: ParamsConnectivity,
            grid_geometry: GridGeometry,
            cortical_distances: list[list[float]]
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

        :param cortical_distances: coordinates of the points in the visual cortex.
        :type cortical_distances: list[list[float]]

        :return: matrix of all coupling weights.
        :rtype: numpy.ndarray[(int, int), float]
        """

        coupling_weights = np.zeros((params_ping.nr_neurons["total"], params_ping.nr_neurons["total"]))

        for nts in (pbar := tqdm(list(product([NeuronTypes.EX, NeuronTypes.IN], repeat=2)))):
            pbar.set_description("Computing coupling weights")

            dist = self._get_neurons_dist(
                neuron_type1=nts[0],
                neuron_type2=nts[1],
                nr1=params_ping.nr_neurons[nts[0]],
                nr2=params_ping.nr_neurons[nts[1]],
                params_ping=params_ping,
                grid_geometry=grid_geometry,
                cortical_distances=cortical_distances
            )
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

    def _get_neurons_dist(
            self,
            neuron_type1: NeuronTypes, neuron_type2: NeuronTypes, nr1: int, nr2: int,
            params_ping: ParamsPING,
            grid_geometry: GridGeometry,
            cortical_distances: list[list[float]]
    ) -> np.ndarray[(int, int), float]:
        """
        Computes the matrix of Euclidian distances between two types of neurons.

        This method computes a matrix of :math:`\| \mathsf{loc}(v), \mathsf{loc}(w) \|` between neurons :math:`v` and
        :math:`w` of given types.

        :param neuron_type1: neurons neuron_type 1
        :type neuron_type1: NeuronTypes

        :param neuron_type2: neurons neuron_type 2
        :type neuron_type2: NeuronTypes

        :param nr1: number of neurons of neuron_type 1
        :type nr1: int

        :param nr2: number of neurons of neuron_type 2
        :type nr2: int

        :param params_ping: parameters describing PING networks and their composition.
        :type params_ping: ParamsPING

        :param grid_geometry: contains information about grid locations of PING networks and neurons located in them.
        :type grid_geometry: GridGeometry

        :param cortical_distances: coordinates of the points in the visual cortex.
        :type cortical_distances: list[list[float]]

        :return: The matrix nr1 x nr2 of pairwise distances between neurons.
        :rtype: numpy.ndarray[(int, int), float]
        """

        def get_cortical_dist(id1, id2):
            # finding to which ping networks the neurons belong
            ping_network1 = grid_geometry.neuron_locations[id1]
            ping_network2 = grid_geometry.neuron_locations[id2]
            # print(ping_network1, ping_network2)

            # computing the distance between the found PING networks
            # (which = the distance between neurons in those PING networks)
            return cortical_distances[ping_network1][ping_network2]


        # find cortical distances between neurons of different types
        dist = np.array([
            [get_cortical_dist(id1, id2) for id2 in params_ping.neur_indices[neuron_type2]]
            for id1 in params_ping.neur_indices[neuron_type1]
        ])

        return dist

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