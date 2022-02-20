from src.GridConnectivity import *
from src.Oscillator import *
from src.NeuronTypes import *
from src.constants import *
from tests.globals import *

import numpy as np
from math import sqrt

class TestGridConnectivity:

    def test_assign_oscillators(self):

        oscillators1, neuron_oscillator_map1 = connectivity1._assign_oscillators()
        oscillators1_expected = [
            Oscillator((0, 0), [0, 1], [0, 1])
        ]
        neuron_oscillator_map1_expected = {
            NeuronTypes.E: {0: 0, 1: 0},
            NeuronTypes.I: {0: 0, 1: 0},
        }
        assert oscillators1 == oscillators1_expected
        assert neuron_oscillator_map1 == neuron_oscillator_map1_expected

        # -----------------------------------------------------------------------

        oscillators2, neuron_oscillator_map2 = connectivity2._assign_oscillators()
        oscillators2_expected = [
            Oscillator((0, 0), [0, 1], [0]),
            Oscillator((0, 1), [2, 3], [1]),
            Oscillator((1, 0), [4, 5], [2]),
            Oscillator((1, 1), [6, 7], [3]),
        ]
        neuron_oscillator_map2_expected = {
            NeuronTypes.E: {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5:2, 6: 3, 7: 3},
            NeuronTypes.I: {0: 0, 1: 1, 2: 2, 3: 3},
        }
        assert oscillators2 == oscillators2_expected
        assert neuron_oscillator_map2 == neuron_oscillator_map2_expected

    def test_get_neurons_dist(self):

        oscillators1, neuron_oscillator_map1 = connectivity1._assign_oscillators()
        distEE1 = connectivity1._get_neurons_dist(
            X1=NeuronTypes.E,
            X2=NeuronTypes.E,
            nr1=connectivity1.nr_excit,
            nr2=connectivity1.nr_excit,
            oscillators=oscillators1,
            neuron_oscillator_map=neuron_oscillator_map1
        )
        distEE1_expected = [[0.0, 0.0], [0.0, 0.0]]

        distII1 = connectivity1._get_neurons_dist(
            X1=NeuronTypes.I,
            X2=NeuronTypes.I,
            nr1=connectivity1.nr_inhibit,
            nr2=connectivity1.nr_inhibit,
            oscillators=oscillators1,
            neuron_oscillator_map=neuron_oscillator_map1
        )
        distII1_expected = [[0.0, 0.0], [0.0, 0.0]]

        distEI1 = connectivity1._get_neurons_dist(
            X1=NeuronTypes.E,
            X2=NeuronTypes.I,
            nr1=connectivity1.nr_excit,
            nr2=connectivity1.nr_inhibit,
            oscillators=oscillators1,
            neuron_oscillator_map=neuron_oscillator_map1
        )
        distEI1_expected = [[0.0, 0.0], [0.0, 0.0]]

        distIE1 = connectivity1._get_neurons_dist(
            X1=NeuronTypes.I,
            X2=NeuronTypes.E,
            nr1=connectivity1.nr_inhibit,
            nr2=connectivity1.nr_excit,
            oscillators=oscillators1,
            neuron_oscillator_map=neuron_oscillator_map1
        )
        distIE1_expected = [[0, 0], [0, 0]]

        assert np.array_equal(distEE1, distEE1_expected)
        assert np.array_equal(distII1, distII1_expected)
        assert np.array_equal(distEI1, distEI1_expected)
        assert np.array_equal(distIE1, distIE1_expected)

        # -----------------------------------------------------------------------

        oscillators2, neuron_oscillator_map2 = connectivity2._assign_oscillators()
        distEE2 = connectivity2._get_neurons_dist(
            X1=NeuronTypes.E,
            X2=NeuronTypes.E,
            nr1=connectivity2.nr_excit,
            nr2=connectivity2.nr_excit,
            oscillators=oscillators2,
            neuron_oscillator_map=neuron_oscillator_map2
        )
        distEE2_expected = [
            [0, 0, 1, 1, 1, 1, sqrt(2), sqrt(2)],
            [0, 0, 1, 1, 1, 1, sqrt(2), sqrt(2)],
            [1, 1, 0, 0, sqrt(2), sqrt(2), 1, 1],
            [1, 1, 0, 0, sqrt(2), sqrt(2), 1, 1],
            [1, 1, sqrt(2), sqrt(2), 0, 0, 1, 1],
            [1, 1, sqrt(2), sqrt(2), 0, 0, 1, 1],
            [sqrt(2), sqrt(2), 1, 1, 1, 1, 0, 0],
            [sqrt(2), sqrt(2), 1, 1, 1, 1, 0, 0]
        ]

        distII2 = connectivity2._get_neurons_dist(
            X1=NeuronTypes.I,
            X2=NeuronTypes.I,
            nr1=connectivity2.nr_inhibit,
            nr2=connectivity2.nr_inhibit,
            oscillators=oscillators2,
            neuron_oscillator_map=neuron_oscillator_map2
        )
        distII2_expected = [
            [0, 1, 1, sqrt(2)],
            [1, 0, sqrt(2), 1],
            [1, sqrt(2), 0, 1],
            [sqrt(2), 1, 1, 0]
        ]

        distEI2 = connectivity2._get_neurons_dist(
            X1=NeuronTypes.E,
            X2=NeuronTypes.I,
            nr1=connectivity2.nr_excit,
            nr2=connectivity2.nr_inhibit,
            oscillators=oscillators2,
            neuron_oscillator_map=neuron_oscillator_map2
        )
        distEI2_expected = [
            [0, 1, 1, sqrt(2)],
            [0, 1, 1, sqrt(2)],
            [1, 0, sqrt(2), 1],
            [1, 0, sqrt(2), 1],
            [1, sqrt(2), 0, 1],
            [1, sqrt(2), 0, 1],
            [sqrt(2), 1, 1, 0],
            [sqrt(2), 1, 1, 0]
        ]

        distIE2 = connectivity2._get_neurons_dist(
            X1=NeuronTypes.I,
            X2=NeuronTypes.E,
            nr1=connectivity2.nr_inhibit,
            nr2=connectivity2.nr_excit,
            oscillators=oscillators2,
            neuron_oscillator_map=neuron_oscillator_map2
        )
        distIE2_expected = [
            [0, 0, 1, 1, 1, 1, sqrt(2), sqrt(2)],
            [1, 1, 0, 0, sqrt(2), sqrt(2), 1, 1],
            [1, 1, sqrt(2), sqrt(2), 0, 0, 1, 1],
            [sqrt(2), sqrt(2), 1, 1, 1, 1, 0, 0]
        ]

        assert np.array_equal(distEE2, distEE2_expected)
        assert np.array_equal(distII2, distII2_expected)
        assert np.array_equal(distEI2, distEI2_expected)
        assert np.array_equal(distIE2, distIE2_expected)