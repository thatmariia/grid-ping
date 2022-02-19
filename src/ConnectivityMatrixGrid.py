from src.Oscillator import *
from src.NeuronTypes import *

from src.misc import *


class ConnectivityMatrixGrid():
    """
    This class constructs the connectivity matrix for the oscillatory network.

    :param nr_excit: number of excitatory neurons in the network.
    :type nr_excit: int

    :param nr_inhibit: number of inhibitory neurons in the network.
    :type nr_inhibit: int

    :param nr_oscillators: number of oscillators in the network.
    :type nr_oscillators: int

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

    :ivar nr_oscillators: list of oscillators in the network
    :type nr_oscillators: list[Oscillator]

    :ivar nr_excit_per_oscillator: number of excitatory neurons in each oscillator.
    :type nr_oscillators: int

    :ivar nr_inhibit_per_oscillator: number of inhibitory neurons in each oscillator.
    :type nr_oscillators: int

    :ivar neuron_oscillator_map: A dictionary mapping a neuron to the oscillator it belongs to.
    :type nr_oscillators: dict[NeuronTypes: dict[int, int]]

    :ivar grid_size: The size of a side of the network grid.
    :type nr_oscillators: int
    """

    def __init__(self, nr_excit, nr_inhibit, nr_oscillators):

        assert nr_excit % nr_oscillators == 0, "Cannot allocated equal number of excitatory neurons to each oscillator. Make sure the number of oscillators divides the number of excitatory neurons."
        assert nr_inhibit % nr_oscillators == 0, "Cannot allocated equal number of inhibitory neurons to each oscillator. Make sure the number of oscillators divides the number of inhibitory neurons."
        assert int(math.sqrt(nr_oscillators)) == math.sqrt(nr_oscillators), "The oscillators should be arranged in a square grid. Make sure the number of oscillators is a perfect square."

        self.nr_oscillators = nr_oscillators
        self.oscillators = []

        # number of neurons of each type in each oscillator
        self.nr_excit_per_oscillator = nr_excit // nr_oscillators
        self.nr_inhibit_per_oscillator = nr_inhibit // nr_oscillators

        # maps the neuron ID to an oscillator it belongs to
        self.neuron_oscillator_map = {
            NeuronTypes.E: {},
            NeuronTypes.I: {}
        }

        self.grid_size = int(math.sqrt(nr_oscillators))  # now assuming the grid is square

        self.assign_oscillators()

        # excitatory  to excitatory
        self.EE = 0.004
        self.sEE = 0.4

        # excitatory  to inhibitory
        self.EI = 0.07
        self.sEI = 0.3

        # inhibitory to excitatory
        self.IE = -0.04
        self.sIE = 0.3

        # inhibitory to inhibitory
        self.II = -0.015
        self.sII = 0.3

        self.KEE, self.KII, self.KEI, self.KIE = self.get_KXXs(nr_excit=nr_excit, nr_inhibit=nr_inhibit)

        # TODO:: compute self.S
        # self.S = self.get_S(
        #     nr_neurons=nr_neurons,
        #     nr_excit=nr_excit,
        #     nr_inhibit=nr_inhibit
        # )

    def assign_oscillators(self):
        """
        Creates oscillators, assigns grid locations to them, and adds the same number of neurons of each type to them.
        """
        for i in range(self.nr_oscillators):
            x = i // self.grid_size
            y = i % self.grid_size

            excit_ids = []

            for id in range(i * self.nr_excit_per_oscillator, (i + 1) * self.nr_excit_per_oscillator + 1):
                excit_ids.append(id)
                self.neuron_oscillator_map[NeuronTypes.E][id] = i

            inhibit_ids = []

            for id in range(i * self.nr_inhibit_per_oscillator, (i + 1) * self.nr_inhibit_per_oscillator + 1):
                inhibit_ids.append(id)
                self.neuron_oscillator_map[NeuronTypes.I][id] = i

            oscillator = Oscillator(
                location=(x, y),
                excit_ids=excit_ids,
                inhibit_ids=inhibit_ids
            )
            self.oscillators.append(oscillator)

    def get_KXXs(self, nr_excit, nr_inhibit):
        """
        Computes the coupling weights between all neurons.

        :param nr_excit: number of excitatory neurons
        :param nr_inhibit: number of inhibitory neurons

        :return: coupling strengths for EE, II, EI, IE connections
        :rtype: tuple[list[list[int]]]
        """
        dist_EE = self._get_neurons_dist(
            X1=NeuronTypes.E,
            X2=NeuronTypes.E,
            nr1=nr_excit,
            nr2=nr_excit
        )
        dist_II = self._get_neurons_dist(
            X1=NeuronTypes.I,
            X2=NeuronTypes.I,
            nr1=nr_inhibit,
            nr2=nr_inhibit
        )
        dist_EI = self._get_neurons_dist(
            X1=NeuronTypes.E,
            X2=NeuronTypes.I,
            nr1=nr_excit,
            nr2=nr_inhibit
        )
        dist_IE = self._get_neurons_dist(
            X1=NeuronTypes.I,
            X2=NeuronTypes.E,
            nr1=nr_inhibit,
            nr2=nr_excit
        )

        return self._compute_KXX(dist=dist_EE, XX=self.EE, sXX=self.sEE), \
               self._compute_KXX(dist=dist_II, XX=self.II, sXX=self.sII), \
               self._compute_KXX(dist=dist_EI, XX=self.EI, sXX=self.sEI), \
               self._compute_KXX(dist=dist_IE, XX=self.IE, sXX=self.sIE)

    def _compute_KXX(dist, XX, sXX):
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
        KXX = XX * np.exp(np.true_divide(-dist, sXX))
        return KXX

    def _get_neurons_dist(self, X1, X2, nr1, nr2):
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

        :return: dist: The matrix nr1 x nr2 of pairwise distances between neurons.
        :rtype: list[list[float]]
        """
        dist = np.zeros((nr1, nr2))

        for id1 in range(nr1):
            for id2 in range(nr2):

                # finding to which oscillators the neurons belong
                oscillator1 = self.oscillators[self.neuron_oscillator_map[X1][id1]]
                oscillator2 = self.oscillators[self.neuron_oscillator_map[X2][id2]]

                # computing the distance between the found oscillators
                # (which = the distance between neurons in those oscillators)
                # FIXME:: assuming unit distance for now
                dist[id1][id2] = euclidian_dist_R2(
                    x=oscillator1.location[0] - oscillator2.location[0],
                    y=oscillator1.location[1] - oscillator2.location[1]
                )
        return dist






