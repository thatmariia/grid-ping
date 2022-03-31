from src.PINGNetwork import *
from src.NeuronTypes import *
from src.misc import *
from src.constants import *

from itertools import product


class GridConnectivity:
    """
    This class constructs the connectivity matrix for the oscillatory network.

    The interaction strength of lateral connections is represented by a matrix :math:`K` of pairwise coupling weights
    defined by an exponential function decaying by the Euclidean distance between the PING networks they belong to:

    :math:`K_{v, w} = C_{ \mathsf{type}(v), \mathsf{type}(w)} \exp (-\| \mathsf{loc}(v), \mathsf{loc}(w) \| / s_{v, w}),`

    where

    * :math:`v, w` are two arbitrary neurons in the network,
    * :math:`\mathsf{type}(v)` maps a neuron to its type (see :obj:`NeuronTypes`),
    * :math:`\mathsf{loc}(v)` maps a neuron to its location on the grid,
    * :math:`s_{v, w}` is the spatial constant (see :obj:`constants.SPATIAL_CONST`).

    This equation was introduced in :cite:p:`Izhikevich2003`.

    This class performs the assignment of neurons to relevant PING networks arranged in a grid and computes the matrix
    of coupling weights.


    :param nr_neurons: number of neurons of each type in the network.
    :type nr_neurons: dict[NeuronTypes, int]

    :param nr_ping_networks: number of PING networks in the network.
    :type nr_ping_networks: int


    :raises:
        AssertionError: If the number of PING networks is smaller than 1.
    :raises:
        AssertionError: if number of excitatory neurons doesn't divide the number of PING networks as there should be
        an equal number of excitatory neurons in each PING network.
    :raises:
        AssertionError: if number of inhibitory neurons doesn't divide the number of PING networks as there should be
        an equal number of inhibitory neurons in each PING network.
    :raises:
        AssertionError: if the number of PING networks is not a square as PING networks should be arranged in a square
        grid.


    :ivar _nr_neurons: number of neurons of each type in the network.
    :type _nr_neurons: dict[NeuronTypes, int]

    :ivar _nr_ping_networks: number of ping_networks in the network.
    :type _nr_ping_networks: int

    :ivar coupling_weights: Matrix of all coupling weights.
    :type coupling_weights: np.ndarray[(int, int), float]
    """

    def __init__(self, nr_neurons, nr_ping_networks):

        assert nr_ping_networks >= 1, \
            "Number of PING networks cannot be smaller than 1."
        assert nr_neurons[NeuronTypes.E] % nr_ping_networks == 0, \
            "Cannot allocated equal number of excitatory neurons to each PING network. Make sure the number of " \
            "PING networks divides the number of excitatory neurons."
        assert nr_neurons[NeuronTypes.I] % nr_ping_networks == 0, \
            "Cannot allocated equal number of inhibitory neurons to each PING network. Make sure the number of " \
            "PING networks divides the number of inhibitory neurons."
        assert int(math.sqrt(nr_ping_networks)) == math.sqrt(nr_ping_networks), \
            "The PING networks should be arranged in a square grid. Make sure the number of PING networks is a " \
            "perfect square."

        self._nr_neurons = nr_neurons
        self._nr_ping_networks = nr_ping_networks

        ping_networks, neuron_ping_map = self._assign_ping_networks()

        self.coupling_weights = self._compute_coupling_weights(
            ping_networks=ping_networks,
            neuron_ping_map=neuron_ping_map
        )

    def _assign_ping_networks(self) -> tuple[list[PINGNetwork], dict[NeuronTypes, dict[int, int]]]:
        """
        Creates PING networks, assigns grid locations to them, and adds the same number of neurons of each neuron type
        to them.

        In other words, this function creates a map that can be used as function :math:`\mathsf{loc}`.

        :return: list of PING networks in the network and a dictionary mapping a neuron to the PING network it belongs
            to.
        :rtype: tuple[list[PINGNetwork], dict[NeuronTypes, dict[int, int]]]
        """

        ping_networks = []
        neuron_ping_map = {
            NeuronTypes.E: {},
            NeuronTypes.I: {}
        }

        grid_size = int(math.sqrt(self._nr_ping_networks))  # now assuming the grid is square

        #  number of neurons of each neuron_type in each PING network
        nr_ex_per_ping_network = self._nr_neurons[NeuronTypes.E] // self._nr_ping_networks
        nr_in_per_ping_network = self._nr_neurons[NeuronTypes.I] // self._nr_ping_networks

        for i in range(self._nr_ping_networks):
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

            ping_network = PINGNetwork(
                location=(x, y),
                excit_ids=ex_ids,
                inhibit_ids=in_ids
            )

            ping_networks.append(ping_network)

        return ping_networks, neuron_ping_map

    def _compute_coupling_weights(
            self,
            ping_networks: list[PINGNetwork], neuron_ping_map: dict[NeuronTypes, dict[int, int]]
    ) -> np.ndarray[(int, int), float]:
        """
        Computes the coupling weights between all neurons.

        Essentially, this method computes the full matrix :math:`K` of coupling weights.

        :param ping_networks: list of PING networks in the network.
        :type ping_networks: list[PINGNetwork]

        :param neuron_ping_map: a dictionary mapping a neuron to the PING network it belongs to.
        :type neuron_ping_map: dict[NeuronTypes, dict[int, int]]

        :return: matrix of all coupling weights.
        :rtype: numpy.ndarray[(int, int), float]
        """

        nr_neurons = self._nr_neurons[NeuronTypes.E] + self._nr_neurons[NeuronTypes.I]
        all_coupling_weights = np.zeros((nr_neurons, nr_neurons))

        for neuron_types in list(product([NeuronTypes.E, NeuronTypes.I], repeat=2)):
            dist = self._get_neurons_dist(
                neuron_type1=neuron_types[0],
                neuron_type2=neuron_types[1],
                nr1=self._nr_neurons[neuron_types[0]],
                nr2=self._nr_neurons[neuron_types[1]],
                ping_networks=ping_networks,
                neuron_ping_map=neuron_ping_map
            )
            types_coupling_weights = self._compute_type_coupling_weights(
                dist=dist,
                max_connect_strength=MAX_CONNECT_STRENGTH[(neuron_types[0], neuron_types[1])],
                spatial_const=SPATIAL_CONST[(neuron_types[0], neuron_types[1])]
            )
            # TODO:: why is this?
            if neuron_types[0] == neuron_types[1]:
                all_coupling_weights[
                    neur_slice(neuron_types[0], self._nr_neurons[NeuronTypes.E], self._nr_neurons[NeuronTypes.I]),
                    neur_slice(neuron_types[1], self._nr_neurons[NeuronTypes.E], self._nr_neurons[NeuronTypes.I])
                ] = types_coupling_weights
            else:
                all_coupling_weights[
                    neur_slice(neuron_types[1], self._nr_neurons[NeuronTypes.E], self._nr_neurons[NeuronTypes.I]),
                    neur_slice(neuron_types[0], self._nr_neurons[NeuronTypes.E], self._nr_neurons[NeuronTypes.I])
                ] = types_coupling_weights.T

        return np.nan_to_num(all_coupling_weights)

    def _get_neurons_dist(
            self,
            neuron_type1: NeuronTypes, neuron_type2: NeuronTypes, nr1: int, nr2: int,
            ping_networks: list[PINGNetwork], neuron_ping_map: dict[NeuronTypes, dict[int, int]]
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
        :type ping_networks: list[PINGNetwork]

        :param neuron_ping_map: a dictionary mapping a neuron to the PING network it belongs to.
        :type neuron_ping_map: dict[NeuronTypes, dict[int, int]]

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
                # FIXME:: assuming unit distance for now
                dist[id1][id2] = euclidian_dist(
                    p1=(ping_network1.location[0], ping_network1.location[1]),
                    p2=(ping_network2.location[0], ping_network2.location[1])
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







