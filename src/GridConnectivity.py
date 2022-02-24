from src.Oscillator import *
from src.NeuronTypes import *
from src.misc import *
from src.constants import *

from itertools import product


class GridConnectivity:
    """
    This class constructs the connectivity matrix for the oscillatory network.

    The interaction strength of lateral connections is represented by a matrix :math:`K` of pairwise coupling weights
    defined by an exponential function decaying by the euclidean distance between the oscillators they belong to:

    :math:`K_{v, w} = C_{ \mathsf{type}(v), \mathsf{type}(w)} \exp (-\| \mathsf{loc}(v), \mathsf{loc}(w) \| / s_{v, w}),`

    where

    * :math:`v, w` are two arbitrary neurons in the network,
    * :math:`\mathsf{type}(v)` maps a neuron to its type (excitatory or inhibitory),
    * :math:`\mathsf{loc}(v)` maps a neuron to its location on the grid,
    * :math:`s_{v, w}` is the spatial constant (see :obj:`constants.SPATIAL_CONST`).

    This equation was introduced in :cite:p:`Izhikevich2003`.

    This class performs the assignment of neurons to relevant oscillators arranged in a grid and computes the matrix
    of coupling weights.


    :param nr_neurons: number of neurons of each type in the network.
    :type nr_neurons: dict[NeuronTypes: int]

    :param nr_oscillators: number of oscillators in the network.
    :type nr_oscillators: int


    :raises:
        AssertionError: If the number of oscillators is smaller than 1.
    :raises:
        AssertionError: if number of excitatory neurons doesn't divide the number of oscillators as there should be
        an equal number of excitatory neurons in each oscillator.
    :raises:
        AssertionError: if number of inhibitory neurons doesn't divide the number of oscillators as there should be
        an equal number of inhibitory neurons in each oscillator.
    :raises:
        AssertionError: if the number of oscillators is not a square as oscillators should be arranged in a square grid.


    :ivar nr_neurons: number of neurons of each type in the network.
    :type nr_neurons: dict[NeuronTypes: int]

    :ivar nr_oscillators: number of oscillators in the network.
    :type nr_oscillators: int

    :ivar coupling_weights: Matrix of all coupling weights.
    :type coupling_weights: ndarray[ndarray[float]]
    """

    def __init__(self, nr_neurons, nr_oscillators):

        assert nr_oscillators >= 1, "Number of oscillators cannot be smaller than 1."
        assert nr_neurons[NeuronTypes.E] % nr_oscillators == 0, \
            "Cannot allocated equal number of excitatory neurons to each oscillator. Make sure the number of " \
            "oscillators divides the number of excitatory neurons. "
        assert nr_neurons[NeuronTypes.I] % nr_oscillators == 0, \
            "Cannot allocated equal number of inhibitory neurons to each oscillator. Make sure the number of " \
            "oscillators divides the number of inhibitory neurons. "
        assert int(math.sqrt(nr_oscillators)) == math.sqrt(nr_oscillators), \
            "The oscillators should be arranged in a square grid. Make sure the number of oscillators is a perfect " \
            "square. "

        self.nr_neurons = nr_neurons
        self.nr_oscillators = nr_oscillators

        oscillators, neuron_oscillator_map = self._assign_oscillators()

        self.coupling_weights = self._compute_coupling_weights(
            oscillators=oscillators,
            neuron_oscillator_map=neuron_oscillator_map
        )

    def _assign_oscillators(self):
        """
        Creates oscillators, assigns grid locations to them, and adds the same number of neurons of each neuron type to them.

        In other words, this function creates a map that can be used as function :math:`\mathsf{loc}`.

        :return: list of oscillators in the network and a dictionary mapping a neuron to the oscillator it belongs to.
        :rtype: tuple[list[Oscillator], dict[NeuronTypes: dict[int, int]]]
        """

        oscillators = []
        neuron_oscillator_map = {
            NeuronTypes.E: {},
            NeuronTypes.I: {}
        }

        grid_size = int(math.sqrt(self.nr_oscillators))  # now assuming the grid is square

        #  number of neurons of each neuron_type in each oscillator
        nr_ex_per_oscillator = self.nr_neurons[NeuronTypes.E] // self.nr_oscillators
        nr_in_per_oscillator = self.nr_neurons[NeuronTypes.I] // self.nr_oscillators

        for i in range(self.nr_oscillators):
            x = i // grid_size
            y = i % grid_size

            ex_ids = []
            for neuron_id in range(i * nr_ex_per_oscillator, (i + 1) * nr_ex_per_oscillator):
                ex_ids.append(neuron_id)
                neuron_oscillator_map[NeuronTypes.E][neuron_id] = i

            in_ids = []
            for neuron_id in range(i * nr_in_per_oscillator, (i + 1) * nr_in_per_oscillator):
                in_ids.append(neuron_id)
                neuron_oscillator_map[NeuronTypes.I][neuron_id] = i

            oscillator = Oscillator(
                location=(x, y),
                excit_ids=ex_ids,
                inhibit_ids=in_ids
            )
            oscillators.append(oscillator)

        return oscillators, neuron_oscillator_map

    def _compute_coupling_weights(self, oscillators, neuron_oscillator_map):
        """
        Computes the coupling weights between all neurons.

        Essentially, this method computes the full matrix :math:`K` of coupling weights.

        :param oscillators: list of oscillators in the network.
        :type oscillators: list[Oscillator]

        :param neuron_oscillator_map: a dictionary mapping a neuron to the oscillator it belongs to.
        :type neuron_oscillator_map: dict[NeuronTypes: dict[int, int]]

        :return: matrix of all coupling weights.
        :rtype: ndarray[ndarray[float]]
        """

        nr_neurons = self.nr_neurons[NeuronTypes.E] + self.nr_neurons[NeuronTypes.I]
        all_coupling_weights = np.zeros((nr_neurons, nr_neurons))

        for neuron_types in list(product([NeuronTypes.E, NeuronTypes.I], repeat=2)):
            dist = self._get_neurons_dist(
                neuron_type1=neuron_types[0],
                neuron_type2=neuron_types[1],
                nr1=self.nr_neurons[neuron_types[0]],
                nr2=self.nr_neurons[neuron_types[1]],
                oscillators=oscillators,
                neuron_oscillator_map=neuron_oscillator_map
            )
            types_coupling_weights = self._compute_type_coupling_weights(
                dist=dist,
                max_connect_strength=MAX_CONNECT_STRENGTH[(neuron_types[0], neuron_types[1])],
                spatial_const=SPATIAL_CONST[(neuron_types[0], neuron_types[1])]
            )
            # TODO:: why is this?
            if neuron_types[0] == neuron_types[1]:
                all_coupling_weights[
                    neur_slice(neuron_types[0], self.nr_neurons[NeuronTypes.E], self.nr_neurons[NeuronTypes.I]),
                    neur_slice(neuron_types[1], self.nr_neurons[NeuronTypes.E], self.nr_neurons[NeuronTypes.I])
                ] = types_coupling_weights
            else:
                all_coupling_weights[
                    neur_slice(neuron_types[1], self.nr_neurons[NeuronTypes.E], self.nr_neurons[NeuronTypes.I]),
                    neur_slice(neuron_types[0], self.nr_neurons[NeuronTypes.E], self.nr_neurons[NeuronTypes.I])
                ] = types_coupling_weights.T

        return np.nan_to_num(all_coupling_weights)

    def _get_neurons_dist(self, neuron_type1, neuron_type2, nr1, nr2, oscillators, neuron_oscillator_map):
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

        :param oscillators: list of oscillators in the network.
        :type oscillators: list[Oscillator]

        :param neuron_oscillator_map: a dictionary mapping a neuron to the oscillator it belongs to.
        :type neuron_oscillator_map: dict[NeuronTypes: dict[int, int]]

        :return: The matrix nr1 x nr2 of pairwise distances between neurons.
        :rtype: list[list[float]]
        """

        dist = np.zeros((nr1, nr2))

        for id1 in range(nr1):
            for id2 in range(nr2):

                # finding to which oscillators the neurons belong
                oscillator1 = oscillators[neuron_oscillator_map[neuron_type1][id1]]
                oscillator2 = oscillators[neuron_oscillator_map[neuron_type2][id2]]

                # computing the distance between the found oscillators
                # (which = the distance between neurons in those oscillators)
                # FIXME:: assuming unit distance for now
                dist[id1][id2] = euclidian_dist_R2(
                    p1=(oscillator1.location[0], oscillator1.location[1]),
                    p2=(oscillator2.location[0], oscillator2.location[1])
                )

        return dist

    def _compute_type_coupling_weights(self, dist, max_connect_strength, spatial_const):
        """
        Computes the coupling weights for connections between two types of neurons.

        This method computes a matrix of :math:`K_{v, w}` between neurons :math:`v` and
        :math:`w` of given types.

        :param dist: distance matrix with pairwise distances between neurons.
        :type dist: list[list[float]]

        :param max_connect_strength: max connection strength between neuron types.
        :type max_connect_strength: float

        :param spatial_const: spatial constant for the neuron types.
        :type spatial_const: float

        :return: the matrix of coupling weights of size nr1 x nr2, where n1 and nr2 - number of neurons of
            each neuron_type in the coupling of interest.
        :rtype: list[list[float]]
        """

        coupling_weights_type = max_connect_strength * np.exp(np.true_divide(-np.array(dist), spatial_const))
        return coupling_weights_type







