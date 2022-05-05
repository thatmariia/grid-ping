from src.params.ParamsConnectivity import *

from src.izhikevich_simulation.PINGNetworkNeurons import *
from src.izhikevich_simulation.Connectivity import *
from src.misc import *

from itertools import product
import numpy as np


class ConnectivityGridPINGFactory:
    """
    This class determines the connectivity between neurons in the oscillatory network.
    """

    def __init__(self):
        pass

    def create(
            self, params_ping: ParamsPING, params_connectivity: ParamsConnectivity,
            cortical_coords: list[list[tuple[float, float]]]
    ) -> Connectivity:
        """
        Determines the connectivity between neurons in the oscillatory network.

        :param params_ping: parameters describing PING networks and their composition.
        :type params_ping: ParamsPING

        :param params_connectivity: parameters of the network connectivity.
        :type params_connectivity: ParamsConnectivity

        :param cortical_coords: coordinates of the points in the visual cortex.
        :type cortical_coords: list[list[tuple[float, float]]]

        :return: connectivity between neurons in the oscillatory network.
        :rtype: Connectivity
        """

        ping_networks, neuron_ping_map = self._assign_ping_networks(params_ping)
        coupling_weights = self._compute_coupling_weights(
            params_ping, params_connectivity, ping_networks, neuron_ping_map, cortical_coords
        )

        connectivity = Connectivity(
            params_ping=params_ping,
            coupling_weights=coupling_weights
        )

        return connectivity

    def _assign_ping_networks(
            self, params_ping: ParamsPING,
    ) -> tuple[list[PINGNetworkNeurons], dict[NeuronTypes, dict[int, int]]]:
        """
        Creates PING networks, assigns grid locations to them, and adds the same number of neurons of each neuron type
        to them.

        In other words, this function creates a map that can be used as function :math:`\mathsf{loc}`.

        :param params_ping: parameters describing PING networks and their composition.
        :type params_ping: ParamsPING

        :return: list of PING networks in the network and a dictionary mapping a neuron to the PING network it belongs to.
        :rtype: tuple[list[PINGNetworkNeurons], dict[NeuronTypes, dict[int, int]]]
        """

        ping_networks = []
        neuron_ping_map = {
            NeuronTypes.EX: {},
            NeuronTypes.IN: {}
        }

        grid_size = int(math.sqrt(params_ping.nr_ping_networks))  # now assuming the grid is square

        #  number of neurons of each neuron_type in each PING network
        nr_ex_per_ping_network = params_ping.nr_neurons[NeuronTypes.EX] // params_ping.nr_ping_networks
        nr_in_per_ping_network = params_ping.nr_neurons[NeuronTypes.IN] // params_ping.nr_ping_networks

        for i in range(params_ping.nr_ping_networks):
            x = i // grid_size
            y = i % grid_size

            ex_ids = []
            for neuron_id in range(i * nr_ex_per_ping_network, (i + 1) * nr_ex_per_ping_network):
                ex_ids.append(neuron_id)
                neuron_ping_map[NeuronTypes.EX][neuron_id] = i

            in_ids = []
            for neuron_id in range(i * nr_in_per_ping_network, (i + 1) * nr_in_per_ping_network):
                in_ids.append(neuron_id)
                neuron_ping_map[NeuronTypes.IN][neuron_id] = i

            ping_network = PINGNetworkNeurons(
                grid_location=(x, y),
                excit_ids=ex_ids,
                inhibit_ids=in_ids
            )

            ping_networks.append(ping_network)

        return ping_networks, neuron_ping_map

    def _compute_coupling_weights(
            self,
            params_ping: ParamsPING, params_connectivity: ParamsConnectivity,
            ping_networks: list[PINGNetworkNeurons], neuron_ping_map: dict[NeuronTypes, dict[int, int]],
            cortical_coords: list[list[tuple[float, float]]]
    ) -> np.ndarray[(int, int), float]:
        """
        Computes the coupling weights between all neurons.

        Essentially, this method computes the full matrix :math:`K` of coupling weights.
        The approach is derived from :cite:p:`Lowet2015`.

        :param params_ping: parameters describing PING networks and their composition.
        :type params_ping: ParamsPING

        :param params_connectivity: parameters of the network connectivity.
        :type params_connectivity: ParamsConnectivity

        :param ping_networks: list of PING networks in the network.
        :type ping_networks: list[PINGNetworkNeurons]

        :param neuron_ping_map: a dictionary mapping a neuron to the PING network it belongs to.
        :type neuron_ping_map: dict[NeuronTypes, dict[int, int]]

        :param cortical_coords: coordinates of the points in the visual cortex.
        :type cortical_coords: list[list[tuple[float, float]]]

        :return: matrix of all coupling weights.
        :rtype: numpy.ndarray[(int, int), float]
        """

        coupling_weights = np.zeros((params_ping.nr_neurons["total"], params_ping.nr_neurons["total"]))

        for nts in list(product([NeuronTypes.EX, NeuronTypes.IN], repeat=2)):
            dist = self._get_neurons_dist(
                neuron_type1=nts[0],
                neuron_type2=nts[1],
                nr1=params_ping.nr_neurons[nts[0]],
                nr2=params_ping.nr_neurons[nts[1]],
                ping_networks=ping_networks,
                neuron_ping_map=neuron_ping_map,
                cortical_coords=cortical_coords
            )
            types_coupling_weights = self._compute_type_coupling_weights(
                dist=dist,
                max_connect_strength=params_connectivity.max_connect_strength[(nts[0], nts[1])],
                spatial_const=params_connectivity.spatial_consts[(nts[0], nts[1])]
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
            ping_networks: list[PINGNetworkNeurons], neuron_ping_map: dict[NeuronTypes, dict[int, int]],
            cortical_coords: list[list[tuple[float, float]]]
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

        :param ping_networks: list of PING networks in the network.
        :type ping_networks: list[PINGNetworkNeurons]

        :param neuron_ping_map: a dictionary mapping a neuron to the PING network it belongs to.
        :type neuron_ping_map: dict[NeuronTypes, dict[int, int]]

        :param cortical_coords: coordinates of the points in the visual cortex.
        :type cortical_coords: list[list[tuple[float, float]]]

        :return: The matrix nr1 x nr2 of pairwise distances between neurons.
        :rtype: numpy.ndarray[(int, int), float]
        """

        dist = np.zeros((nr1, nr2))

        for id1 in range(nr1):
            for id2 in range(nr2):

                # finding to which ping_networks the neurons belong
                ping_network1 = ping_networks[neuron_ping_map[neuron_type1][id1]]
                ping_network2 = ping_networks[neuron_ping_map[neuron_type2][id2]]

                # computing the distance between the found PING networks
                # (which = the distance between neurons in those PING networks)
                cortical_coord1 = cortical_coords[ping_network1.grid_location[0]][ping_network1.grid_location[1]]
                cortical_coord2 = cortical_coords[ping_network2.grid_location[0]][ping_network2.grid_location[1]]
                dist[id1][id2] = euclidian_dist(
                    cortical_coord1, cortical_coord2
                )

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