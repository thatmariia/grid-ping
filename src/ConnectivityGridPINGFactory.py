from src.PINGNetworkNeurons import *
from src.NeuronTypes import *
from src.Connectivity import *
from src.misc import *
from src.constants import *

from itertools import product


class ConnectivityGridPINGFactory:
    """
    This class determines the connectivity between neurons in the oscillatory network.
    """

    def __init__(self):
        pass

    def create(
            self,
            nr_excitatory: int, nr_inhibitory: int, nr_ping_networks: int,
            cortical_coords: list[list[tuple[float, float]]]
    ) -> Connectivity:
        """
        Determines the connectivity between neurons in the oscillatory network.

        :param nr_excitatory: number of excitatory neurons in the network.
        :type nr_excitatory: int

        :param nr_inhibitory: number of inhibitory neurons in the network.
        :type nr_inhibitory: int

        :param nr_ping_networks: number of PING networks in the network.
        :type nr_ping_networks: int

        :param cortical_coords: coordinates of the points in the visual cortex.
        :type cortical_coords: list[list[tuple[float, float]]]

        :raises:
            AssertionError: if the number of excitatory neurons is smaller than 2.
        :raises:
            AssertionError: if the number of inhibitory neurons is smaller than 2.

        :return: connectivity between neurons in the oscillatory network.
        :rtype: Connectivity
        """

        # FIXME:: this assertions are only there because of the stim_input?
        assert nr_excitatory >= 2, "Number of excitatory neurons cannot be smaller than 2."
        assert nr_inhibitory >= 2, "Number of inhibitory neurons cannot be smaller than 2."

        nr_neurons = {
            NeuronTypes.E: nr_excitatory,
            NeuronTypes.I: nr_inhibitory,
            "total": nr_excitatory + nr_inhibitory
        }
        neur_slice = {
            NeuronTypes.E: slice(nr_excitatory),
            NeuronTypes.I: slice(nr_excitatory, nr_excitatory + nr_inhibitory),
        }
        ping_networks, neuron_ping_map = self._assign_ping_networks(nr_neurons, nr_ping_networks)
        coupling_weights = self._compute_coupling_weights(
            nr_neurons, neur_slice, ping_networks, neuron_ping_map, cortical_coords
        )

        connectivity = Connectivity(
            nr_neurons=nr_neurons,
            neur_slice=neur_slice,
            nr_ping_networks=nr_ping_networks,
            coupling_weights=coupling_weights
        )

        return connectivity

    def _assign_ping_networks(
            self, nr_neurons: dict[Any, int], nr_ping_networks: int
    ) -> tuple[list[PINGNetworkNeurons], dict[NeuronTypes, dict[int, int]]]:
        """
        Creates PING networks, assigns grid locations to them, and adds the same number of neurons of each neuron type
        to them.

        In other words, this function creates a map that can be used as function :math:`\mathsf{loc}`.

        :param nr_neurons: dictionary of number of neurons of each type and the total number of neurons.
        :type nr_neurons: dict[Any, int]

        :param nr_ping_networks: number of PING networks.
        :type nr_ping_networks: int

        :return: list of PING networks in the network and a dictionary mapping a neuron to the PING network it belongs to.
        :rtype: tuple[list[PINGNetworkNeurons], dict[NeuronTypes, dict[int, int]]]
        """

        ping_networks = []
        neuron_ping_map = {
            NeuronTypes.E: {},
            NeuronTypes.I: {}
        }

        grid_size = int(math.sqrt(nr_ping_networks))  # now assuming the grid is square

        #  number of neurons of each neuron_type in each PING network
        nr_ex_per_ping_network = nr_neurons[NeuronTypes.E] // nr_ping_networks
        nr_in_per_ping_network = nr_neurons[NeuronTypes.I] // nr_ping_networks

        for i in range(nr_ping_networks):
            x = i // grid_size
            y = i % grid_size

            ex_ids = []
            for neuron_id in range(i * nr_ex_per_ping_network, (i + 1) * nr_ex_per_ping_network):
                ex_ids.append(neuron_id)
                neuron_ping_map[NeuronTypes.E][neuron_id] = i

            in_ids = []
            for neuron_id in range(i * nr_in_per_ping_network, (i + 1) * nr_in_per_ping_network):
                in_ids.append(neuron_id)
                neuron_ping_map[NeuronTypes.I][neuron_id] = i

            ping_network = PINGNetworkNeurons(
                location=(x, y),
                excit_ids=ex_ids,
                inhibit_ids=in_ids
            )

            ping_networks.append(ping_network)

        return ping_networks, neuron_ping_map

    def _compute_coupling_weights(
            self,
            nr_neurons: dict[Any, int], neur_slice: dict[NeuronTypes, slice],
            ping_networks: list[PINGNetworkNeurons], neuron_ping_map: dict[NeuronTypes, dict[int, int]],
            cortical_coords: list[list[tuple[float, float]]]
    ) -> np.ndarray[(int, int), float]:
        """
        Computes the coupling weights between all neurons.

        Essentially, this method computes the full matrix :math:`K` of coupling weights.
        The approach is derived from :cite:p:`Lowet2015`.

        :param nr_neurons: dictionary of number of neurons of each type and the total number of neurons.
        :type nr_neurons: dict[Any, int]

        :param neur_slice: indices of each type of neurons.
        :type neur_slice: dict[NeuronType, slice]

        :param ping_networks: list of PING networks in the network.
        :type ping_networks: list[PINGNetworkNeurons]

        :param neuron_ping_map: a dictionary mapping a neuron to the PING network it belongs to.
        :type neuron_ping_map: dict[NeuronTypes, dict[int, int]]

        :param cortical_coords: coordinates of the points in the visual cortex.
        :type cortical_coords: list[list[tuple[float, float]]]

        :return: matrix of all coupling weights.
        :rtype: numpy.ndarray[(int, int), float]
        """

        all_coupling_weights = np.zeros((nr_neurons["total"], nr_neurons["total"]))

        for nts in list(product([NeuronTypes.E, NeuronTypes.I], repeat=2)):
            dist = self._get_neurons_dist(
                neuron_type1=nts[0],
                neuron_type2=nts[1],
                nr1=nr_neurons[nts[0]],
                nr2=nr_neurons[nts[1]],
                ping_networks=ping_networks,
                neuron_ping_map=neuron_ping_map,
                cortical_coords=cortical_coords
            )
            types_coupling_weights = self._compute_type_coupling_weights(
                dist=dist,
                max_connect_strength=MAX_CONNECT_STRENGTH[(nts[0], nts[1])],
                spatial_const=SPATIAL_CONST[(nts[0], nts[1])]
            )
            # TODO:: why is this?
            if nts[0] == nts[1]:
                all_coupling_weights[neur_slice[nts[0]], neur_slice[nts[1]]] = types_coupling_weights
            else:
                all_coupling_weights[neur_slice[nts[1]], neur_slice[nts[0]]] = types_coupling_weights.T

        return np.nan_to_num(all_coupling_weights)

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
                cortical_coord1 = cortical_coords[ping_network1.location[0]][ping_network1.location[1]]
                cortical_coord2 = cortical_coords[ping_network2.location[0]][ping_network2.location[1]]
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