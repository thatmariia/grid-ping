from src.Oscillator import *
from src.NeuronTypes import *
from src.misc import *
from src.constants import *


class GridConnectivity:
    """
    This class constructs the connectivity for the oscillatory network.

    :param nr_excit: number of excitatory neurons in the network.
    :type nr_excit: int

    :param nr_inhibit: number of inhibitory neurons in the network.
    :type nr_inhibit: int

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

    :ivar nr_excit: number of excitatory neurons in the network.
    :type nr_excit: int

    :ivar nr_inhibit: number of inhibitory neurons in the network.
    :type nr_inhibit: int

    :ivar nr_oscillators: number of oscillators in the network.
    :type nr_oscillators: int

    :ivar K: Matrix of all coupling weights.
    :type K: ndarray[ndarray[float]]
    """

    def __init__(self, nr_excit, nr_inhibit, nr_oscillators):

        assert nr_oscillators >= 1, "Number of oscillators cannot be smaller than 1."
        assert nr_excit % nr_oscillators == 0, \
            "Cannot allocated equal number of excitatory neurons to each oscillator. Make sure the number of " \
            "oscillators divides the number of excitatory neurons. "
        assert nr_inhibit % nr_oscillators == 0, \
            "Cannot allocated equal number of inhibitory neurons to each oscillator. Make sure the number of " \
            "oscillators divides the number of inhibitory neurons. "
        assert int(math.sqrt(nr_oscillators)) == math.sqrt(nr_oscillators), \
            "The oscillators should be arranged in a square grid. Make sure the number of oscillators is a perfect " \
            "square. "

        self.nr_excit = nr_excit
        self.nr_inhibit = nr_inhibit
        self.nr_oscillators = nr_oscillators

        oscillators, neuron_oscillator_map = self._assign_oscillators()

        # coupling weights
        KEE, KII, KEI, KIE = self._get_KXXs(
            oscillators=oscillators,
            neuron_oscillator_map=neuron_oscillator_map
        )

        self.K = self._get_K(
            KEE=KEE,
            KII=KII,
            KEI=KEI,
            KIE=KIE
        )

    def _assign_oscillators(self):
        """
        Creates oscillators, assigns grid locations to them, and adds the same number of neurons of each type to them.

        :return: list of oscillators in the network and a dictionary mapping a neuron to the oscillator it belongs to.
        :rtype: tuple[list[Oscillator], dict[NeuronTypes: dict[int, int]]]
        """

        oscillators = []
        neuron_oscillator_map = {
            NeuronTypes.E: {},
            NeuronTypes.I: {}
        }

        grid_size = int(math.sqrt(self.nr_oscillators))  # now assuming the grid is square

        #  number of neurons of each type in each oscillator
        nr_excit_per_oscillator = self.nr_excit // self.nr_oscillators
        nr_inhibit_per_oscillator = self.nr_inhibit // self.nr_oscillators

        for i in range(self.nr_oscillators):
            x = i // grid_size
            y = i % grid_size

            excit_ids = []

            for id in range(i * nr_excit_per_oscillator, (i + 1) * nr_excit_per_oscillator):
                excit_ids.append(id)
                neuron_oscillator_map[NeuronTypes.E][id] = i

            inhibit_ids = []

            for id in range(i * nr_inhibit_per_oscillator, (i + 1) * nr_inhibit_per_oscillator):
                inhibit_ids.append(id)
                neuron_oscillator_map[NeuronTypes.I][id] = i

            oscillator = Oscillator(
                location=(x, y),
                excit_ids=excit_ids,
                inhibit_ids=inhibit_ids
            )
            oscillators.append(oscillator)

        return oscillators, neuron_oscillator_map

    def _get_KXXs(self, oscillators, neuron_oscillator_map):
        """
        Computes the coupling weights between all neurons.

        :param nr_excit: number of excitatory neurons.
        :param nr_inhibit: number of inhibitory neurons.

        :param oscillators: list of oscillators in the network.
        :type oscillators: list[Oscillator]

        :param neuron_oscillator_map: a dictionary mapping a neuron to the oscillator it belongs to.
        :type neuron_oscillator_map: dict[NeuronTypes: dict[int, int]]

        :return: coupling strengths for EE, II, EI, IE connections.
        :rtype: tuple[list[list[int]]]
        """

        dist_EE = self._get_neurons_dist(
            X1=NeuronTypes.E,
            X2=NeuronTypes.E,
            nr1=self.nr_excit,
            nr2=self.nr_excit,
            oscillators=oscillators,
            neuron_oscillator_map=neuron_oscillator_map
        )
        dist_II = self._get_neurons_dist(
            X1=NeuronTypes.I,
            X2=NeuronTypes.I,
            nr1=self.nr_inhibit,
            nr2=self.nr_inhibit,
            oscillators=oscillators,
            neuron_oscillator_map=neuron_oscillator_map
        )
        dist_EI = self._get_neurons_dist(
            X1=NeuronTypes.E,
            X2=NeuronTypes.I,
            nr1=self.nr_excit,
            nr2=self.nr_inhibit,
            oscillators=oscillators,
            neuron_oscillator_map=neuron_oscillator_map
        )
        dist_IE = self._get_neurons_dist(
            X1=NeuronTypes.I,
            X2=NeuronTypes.E,
            nr1=self.nr_inhibit,
            nr2=self.nr_excit,
            oscillators=oscillators,
            neuron_oscillator_map=neuron_oscillator_map
        )

        KEE =  self._compute_KXX(
            dist=dist_EE,
            XX=MAX_CONNECT_STRENGTH[(NeuronTypes.E, NeuronTypes.E)],
            sXX=SPATIAL_CONST[(NeuronTypes.E, NeuronTypes.E)]
        )
        KII = self._compute_KXX(
            dist=dist_II,
            XX=MAX_CONNECT_STRENGTH[(NeuronTypes.I, NeuronTypes.I)],
            sXX=SPATIAL_CONST[(NeuronTypes.I, NeuronTypes.I)]
        )
        KEI = self._compute_KXX(
            dist=dist_EI,
            XX=MAX_CONNECT_STRENGTH[(NeuronTypes.E, NeuronTypes.I)],
            sXX=SPATIAL_CONST[(NeuronTypes.E, NeuronTypes.I)]
        )
        KIE = self._compute_KXX(
            dist=dist_IE,
            XX=MAX_CONNECT_STRENGTH[(NeuronTypes.I, NeuronTypes.E)],
            sXX=SPATIAL_CONST[(NeuronTypes.I, NeuronTypes.E)]
        )

        return KEE, KII, KEI, KIE

    def _get_neurons_dist(self, X1, X2, nr1, nr2, oscillators, neuron_oscillator_map):
        """
        Computes the matrix of Euclidian distances between two types of neurons.

        :param X1: neurons type 1
        :type X1: NeuronTypes

        :param X2: neurons type 2
        :type X2: NeuronTypes

        :param nr1: number of neurons of type 1
        :type nr1: int

        :param nr2: number of neurons of type 2
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
                oscillator1 = oscillators[neuron_oscillator_map[X1][id1]]
                oscillator2 = oscillators[neuron_oscillator_map[X2][id2]]

                # computing the distance between the found oscillators
                # (which = the distance between neurons in those oscillators)
                # FIXME:: assuming unit distance for now
                dist[id1][id2] = euclidian_dist_R2(
                    p1=(oscillator1.location[0], oscillator1.location[1]),
                    p2=(oscillator2.location[0], oscillator2.location[1])
                )

        return dist

    def _compute_KXX(self, dist, XX, sXX):
        """
        Computes the coupling weights for connections between two types of neurons.

        :param dist: distance matrix with pairwise distances between neurons.
        :type dist: list[list[float]]

        :param XX: max connection strength between neuron types.
        :type XX: float

        :param sXX: spatial constant for the neuron types.
        :type sXX: float

        :return: the matrix of coupling weights of size nr1 x nr2, where n1 and nr2 - number of neurons of
        each type in the coupling of ineterest.
        :rtype: list[list[float]]
        """

        KXX = XX * np.exp(np.true_divide(-np.array(dist), sXX))
        return KXX

    def _get_K(self, KEE, KII, KEI, KIE):
        """
        Assigns coupling weights.

        :param nr_excit: number of excitatory neurons in the network.
        :type nr_excit: int

        :param nr_inhibit: number of inhibitory neurons in the network.
        :type nr_inhibit: int

        :param KEE, KII, KEI, KIE: coupling weights of respective connections.
        :type KEE, KII, KEI, KIE: list[list[int]]

        :return: matrix of all coupling weights.
        :rtype: ndarray[ndarray[float]]
        """

        nr_neurons = self.nr_excit + self.nr_inhibit

        S = np.zeros((nr_neurons, nr_neurons))

        S[:self.nr_excit, :self.nr_excit] = KEE
        S[self.nr_excit:nr_neurons, self.nr_excit:nr_neurons] = KII
        S[:self.nr_excit, self.nr_excit:nr_neurons] = KIE.T
        S[self.nr_excit:nr_neurons, :self.nr_excit] = KEI.T

        return np.nan_to_num(S)






