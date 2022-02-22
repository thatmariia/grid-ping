from src.Oscillator import *
from src.NeuronTypes import *
from src.misc import *
from src.constants import *

from itertools import product


class GridConnectivity:
    """
    This class constructs the connectivity for the oscillatory network.

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

    :ivar coupling_weights: Matrix of all coupling weights.
    :type coupling_weights: ndarray[ndarray[float]]
    """

    def __init__(self):

        assert NR_OSCILLATORS >= 1, "Number of oscillators cannot be smaller than 1."
        assert NR_NEURONS[NeuronTypes.E] % NR_OSCILLATORS == 0, \
            "Cannot allocated equal number of excitatory neurons to each oscillator. Make sure the number of " \
            "oscillators divides the number of excitatory neurons. "
        assert NR_NEURONS[NeuronTypes.I] % NR_OSCILLATORS == 0, \
            "Cannot allocated equal number of inhibitory neurons to each oscillator. Make sure the number of " \
            "oscillators divides the number of inhibitory neurons. "
        assert int(math.sqrt(NR_OSCILLATORS)) == math.sqrt(NR_OSCILLATORS), \
            "The oscillators should be arranged in a square grid. Make sure the number of oscillators is a perfect " \
            "square. "

        oscillators, neuron_oscillator_map = self._assign_oscillators()

        self.coupling_weights = self.compute_coupling_weights(
            oscillators=oscillators,
            neuron_oscillator_map=neuron_oscillator_map
        )

    def _assign_oscillators(self):
        """
        Creates oscillators, assigns grid locations to them, and adds the same number of neurons of each neuron_type to them.

        :return: list of oscillators in the network and a dictionary mapping a neuron to the oscillator it belongs to.
        :rtype: tuple[list[Oscillator], dict[NeuronTypes: dict[int, int]]]
        """

        oscillators = []
        neuron_oscillator_map = {
            NeuronTypes.E: {},
            NeuronTypes.I: {}
        }

        grid_size = int(math.sqrt(NR_OSCILLATORS))  # now assuming the grid is square

        #  number of neurons of each neuron_type in each oscillator
        nr_excit_per_oscillator = NR_NEURONS[NeuronTypes.E] // NR_OSCILLATORS
        nr_inhibit_per_oscillator = NR_NEURONS[NeuronTypes.I] // NR_OSCILLATORS

        for i in range(NR_OSCILLATORS):
            x = i // grid_size
            y = i % grid_size

            excit_ids = []

            for neuron_id in range(i * nr_excit_per_oscillator, (i + 1) * nr_excit_per_oscillator):
                excit_ids.append(neuron_id)
                neuron_oscillator_map[NeuronTypes.E][neuron_id] = i

            inhibit_ids = []

            for neuron_id in range(i * nr_inhibit_per_oscillator, (i + 1) * nr_inhibit_per_oscillator):
                inhibit_ids.append(neuron_id)
                neuron_oscillator_map[NeuronTypes.I][neuron_id] = i

            oscillator = Oscillator(
                location=(x, y),
                excit_ids=excit_ids,
                inhibit_ids=inhibit_ids
            )
            oscillators.append(oscillator)

        return oscillators, neuron_oscillator_map

    def compute_coupling_weights(self, oscillators, neuron_oscillator_map):
        """
        Computes the coupling weights between all neurons.

        :param oscillators: list of oscillators in the network.
        :type oscillators: list[Oscillator]

        :param neuron_oscillator_map: a dictionary mapping a neuron to the oscillator it belongs to.
        :type neuron_oscillator_map: dict[NeuronTypes: dict[int, int]]

        :return: matrix of all coupling weights.
        :rtype: ndarray[ndarray[float]]
        """

        nr_neurons = NR_NEURONS[NeuronTypes.E] + NR_NEURONS[NeuronTypes.I]
        all_coupling_weights = np.zeros((nr_neurons, nr_neurons))

        for neuron_types in list(product([NeuronTypes.E, NeuronTypes.I], repeat=2)):
            dist = self._get_neurons_dist(
                neuron_type1=neuron_types[0],
                neuron_type2=neuron_types[1],
                nr1=NR_NEURONS[neuron_types[0]],
                nr2=NR_NEURONS[neuron_types[1]],
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
                all_coupling_weights[neur_slice(neuron_types[0]), neur_slice(neuron_types[1])] = types_coupling_weights
            else:
                all_coupling_weights[neur_slice(neuron_types[1]), neur_slice(neuron_types[0])] = types_coupling_weights.T

        return np.nan_to_num(all_coupling_weights)

    def _get_neurons_dist(self, neuron_type1, neuron_type2, nr1, nr2, oscillators, neuron_oscillator_map):
        """
        Computes the matrix of Euclidian distances between two types of neurons.

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







