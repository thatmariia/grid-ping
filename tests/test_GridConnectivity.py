from src.GridConnectivity import *
from src.Oscillator import *
from src.NeuronTypes import *
from src.constants import *

import numpy as np
from math import sqrt, exp


class TestGridConnectivity:

    connectivity1 = GridConnectivity(
        nr_neurons={
            NeuronTypes.E: 2,
            NeuronTypes.I: 2
        },
        nr_oscillators=1
    )

    connectivity2 = GridConnectivity(
        nr_neurons={
            NeuronTypes.E: 8,
            NeuronTypes.I: 4
        },
        nr_oscillators=4
    )

    def test_assign_oscillators(self):
        oscillators1, neuron_oscillator_map1 = self.connectivity1._assign_oscillators()
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

        oscillators2, neuron_oscillator_map2 = self.connectivity2._assign_oscillators()
        oscillators2_expected = [
            Oscillator((0, 0), [0, 1], [0]),
            Oscillator((0, 1), [2, 3], [1]),
            Oscillator((1, 0), [4, 5], [2]),
            Oscillator((1, 1), [6, 7], [3]),
        ]
        neuron_oscillator_map2_expected = {
            NeuronTypes.E: {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3},
            NeuronTypes.I: {0: 0, 1: 1, 2: 2, 3: 3},
        }
        assert oscillators2 == oscillators2_expected
        assert neuron_oscillator_map2 == neuron_oscillator_map2_expected

    def test_get_neurons_dist(self):
        oscillators1, neuron_oscillator_map1 = self.connectivity1._assign_oscillators()
        distEE1 = self.connectivity1._get_neurons_dist(
            neuron_type1=NeuronTypes.E,
            neuron_type2=NeuronTypes.E,
            nr1=self.connectivity1.nr_neurons[NeuronTypes.E],
            nr2=self.connectivity1.nr_neurons[NeuronTypes.E],
            oscillators=oscillators1,
            neuron_oscillator_map=neuron_oscillator_map1
        )
        distEE1_expected = [[0.0, 0.0], [0.0, 0.0]]

        distII1 = self.connectivity1._get_neurons_dist(
            neuron_type1=NeuronTypes.I,
            neuron_type2=NeuronTypes.I,
            nr1=self.connectivity1.nr_neurons[NeuronTypes.I],
            nr2=self.connectivity1.nr_neurons[NeuronTypes.I],
            oscillators=oscillators1,
            neuron_oscillator_map=neuron_oscillator_map1
        )
        distII1_expected = [[0.0, 0.0], [0.0, 0.0]]

        distEI1 = self.connectivity1._get_neurons_dist(
            neuron_type1=NeuronTypes.E,
            neuron_type2=NeuronTypes.I,
            nr1=self.connectivity1.nr_neurons[NeuronTypes.E],
            nr2=self.connectivity1.nr_neurons[NeuronTypes.I],
            oscillators=oscillators1,
            neuron_oscillator_map=neuron_oscillator_map1
        )
        distEI1_expected = [[0.0, 0.0], [0.0, 0.0]]

        distIE1 = self.connectivity1._get_neurons_dist(
            neuron_type1=NeuronTypes.I,
            neuron_type2=NeuronTypes.E,
            nr1=self.connectivity1.nr_neurons[NeuronTypes.I],
            nr2=self.connectivity1.nr_neurons[NeuronTypes.E],
            oscillators=oscillators1,
            neuron_oscillator_map=neuron_oscillator_map1
        )
        distIE1_expected = [[0, 0], [0, 0]]

        assert np.array_equal(distEE1, distEE1_expected)
        assert np.array_equal(distII1, distII1_expected)
        assert np.array_equal(distEI1, distEI1_expected)
        assert np.array_equal(distIE1, distIE1_expected)

        # -----------------------------------------------------------------------

        oscillators2, neuron_oscillator_map2 = self.connectivity2._assign_oscillators()
        distEE2 = self.connectivity2._get_neurons_dist(
            neuron_type1=NeuronTypes.E,
            neuron_type2=NeuronTypes.E,
            nr1=self.connectivity2.nr_neurons[NeuronTypes.E],
            nr2=self.connectivity2.nr_neurons[NeuronTypes.E],
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

        distII2 = self.connectivity2._get_neurons_dist(
            neuron_type1=NeuronTypes.I,
            neuron_type2=NeuronTypes.I,
            nr1=self.connectivity2.nr_neurons[NeuronTypes.I],
            nr2=self.connectivity2.nr_neurons[NeuronTypes.I],
            oscillators=oscillators2,
            neuron_oscillator_map=neuron_oscillator_map2
        )
        distII2_expected = [
            [0, 1, 1, sqrt(2)],
            [1, 0, sqrt(2), 1],
            [1, sqrt(2), 0, 1],
            [sqrt(2), 1, 1, 0]
        ]

        distEI2 = self.connectivity2._get_neurons_dist(
            neuron_type1=NeuronTypes.E,
            neuron_type2=NeuronTypes.I,
            nr1=self.connectivity2.nr_neurons[NeuronTypes.E],
            nr2=self.connectivity2.nr_neurons[NeuronTypes.I],
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

        distIE2 = self.connectivity2._get_neurons_dist(
            neuron_type1=NeuronTypes.I,
            neuron_type2=NeuronTypes.E,
            nr1=self.connectivity2.nr_neurons[NeuronTypes.I],
            nr2=self.connectivity2.nr_neurons[NeuronTypes.E],
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

    def test_compute_type_coupling_weights(self):
        oscillators1, neuron_oscillator_map1 = self.connectivity1._assign_oscillators()

        distEE1 = self.connectivity1._get_neurons_dist(
            neuron_type1=NeuronTypes.E,
            neuron_type2=NeuronTypes.E,
            nr1=self.connectivity1.nr_neurons[NeuronTypes.E],
            nr2=self.connectivity1.nr_neurons[NeuronTypes.E],
            oscillators=oscillators1,
            neuron_oscillator_map=neuron_oscillator_map1
        )
        type_coupling_weights_EE1 = self.connectivity1._compute_type_coupling_weights(
            dist=distEE1,
            max_connect_strength=MAX_CONNECT_STRENGTH[(NeuronTypes.E, NeuronTypes.E)],
            spatial_const=SPATIAL_CONST[(NeuronTypes.E, NeuronTypes.E)]
        )
        type_coupling_weights_EE1_expected = [
            [0.004, 0.004],
            [0.004, 0.004]
        ]

        distII1 = self.connectivity1._get_neurons_dist(
            neuron_type1=NeuronTypes.I,
            neuron_type2=NeuronTypes.I,
            nr1=self.connectivity1.nr_neurons[NeuronTypes.I],
            nr2=self.connectivity1.nr_neurons[NeuronTypes.I],
            oscillators=oscillators1,
            neuron_oscillator_map=neuron_oscillator_map1
        )
        type_coupling_weights_II1 = self.connectivity1._compute_type_coupling_weights(
            dist=distII1,
            max_connect_strength=MAX_CONNECT_STRENGTH[(NeuronTypes.I, NeuronTypes.I)],
            spatial_const=SPATIAL_CONST[(NeuronTypes.I, NeuronTypes.I)]
        )
        type_coupling_weights_II1_expected = [
            [-0.015, -0.015],
            [-0.015, -0.015]
        ]

        distEI1 = self.connectivity1._get_neurons_dist(
            neuron_type1=NeuronTypes.E,
            neuron_type2=NeuronTypes.I,
            nr1=self.connectivity1.nr_neurons[NeuronTypes.E],
            nr2=self.connectivity1.nr_neurons[NeuronTypes.I],
            oscillators=oscillators1,
            neuron_oscillator_map=neuron_oscillator_map1
        )
        type_coupling_weights_EI1 = self.connectivity1._compute_type_coupling_weights(
            dist=distEI1,
            max_connect_strength=MAX_CONNECT_STRENGTH[(NeuronTypes.E, NeuronTypes.I)],
            spatial_const=SPATIAL_CONST[(NeuronTypes.E, NeuronTypes.I)]
        )
        type_coupling_weights_EI1_expected = [
            [0.07, 0.07],
            [0.07, 0.07]
        ]

        distIE1 = self.connectivity1._get_neurons_dist(
            neuron_type1=NeuronTypes.I,
            neuron_type2=NeuronTypes.E,
            nr1=self.connectivity1.nr_neurons[NeuronTypes.I],
            nr2=self.connectivity1.nr_neurons[NeuronTypes.E],
            oscillators=oscillators1,
            neuron_oscillator_map=neuron_oscillator_map1
        )
        type_coupling_weights_IE1 = self.connectivity1._compute_type_coupling_weights(
            dist=distIE1,
            max_connect_strength=MAX_CONNECT_STRENGTH[(NeuronTypes.I, NeuronTypes.E)],
            spatial_const=SPATIAL_CONST[(NeuronTypes.I, NeuronTypes.E)]
        )
        type_coupling_weights_IE1_expected = [
            [-0.04, -0.04],
            [-0.04, -0.04]
        ]

        assert np.array_equal(type_coupling_weights_EE1, type_coupling_weights_EE1_expected)
        assert np.array_equal(type_coupling_weights_II1, type_coupling_weights_II1_expected)
        assert np.array_equal(type_coupling_weights_EI1, type_coupling_weights_EI1_expected)
        assert np.array_equal(type_coupling_weights_IE1, type_coupling_weights_IE1_expected)

        # -----------------------------------------------------------------------

        oscillators2, neuron_oscillator_map2 = self.connectivity2._assign_oscillators()

        distEE2 = self.connectivity2._get_neurons_dist(
            neuron_type1=NeuronTypes.E,
            neuron_type2=NeuronTypes.E,
            nr1=self.connectivity2.nr_neurons[NeuronTypes.E],
            nr2=self.connectivity2.nr_neurons[NeuronTypes.E],
            oscillators=oscillators2,
            neuron_oscillator_map=neuron_oscillator_map2
        )
        type_coupling_weights_EE2 = self.connectivity2._compute_type_coupling_weights(
            dist=distEE2,
            max_connect_strength=MAX_CONNECT_STRENGTH[(NeuronTypes.E, NeuronTypes.E)],
            spatial_const=SPATIAL_CONST[(NeuronTypes.E, NeuronTypes.E)]
        )
        type_coupling_weights_EE2_expected = [
            [0.004, 0.004, 0.004 * exp(-2.5), 0.004 * exp(-2.5), 0.004 * exp(-2.5), 0.004 * exp(-2.5),
             0.004 * exp(-sqrt(2) / 0.4), 0.004 * exp(-sqrt(2) / 0.4)],
            [0.004, 0.004, 0.004 * exp(-2.5), 0.004 * exp(-2.5), 0.004 * exp(-2.5), 0.004 * exp(-2.5),
             0.004 * exp(-sqrt(2) / 0.4), 0.004 * exp(-sqrt(2) / 0.4)],

            [0.004 * exp(-2.5), 0.004 * exp(-2.5), 0.004, 0.004, 0.004 * exp(-sqrt(2) / 0.4),
             0.004 * exp(-sqrt(2) / 0.4), 0.004 * exp(-2.5), 0.004 * exp(-2.5)],
            [0.004 * exp(-2.5), 0.004 * exp(-2.5), 0.004, 0.004, 0.004 * exp(-sqrt(2) / 0.4),
             0.004 * exp(-sqrt(2) / 0.4), 0.004 * exp(-2.5), 0.004 * exp(-2.5)],

            [0.004 * exp(-2.5), 0.004 * exp(-2.5), 0.004 * exp(-sqrt(2) / 0.4), 0.004 * exp(-sqrt(2) / 0.4), 0.004,
             0.004, 0.004 * exp(-2.5), 0.004 * exp(-2.5)],
            [0.004 * exp(-2.5), 0.004 * exp(-2.5), 0.004 * exp(-sqrt(2) / 0.4), 0.004 * exp(-sqrt(2) / 0.4), 0.004,
             0.004, 0.004 * exp(-2.5), 0.004 * exp(-2.5)],

            [0.004 * exp(-sqrt(2) / 0.4), 0.004 * exp(-sqrt(2) / 0.4), 0.004 * exp(-2.5), 0.004 * exp(-2.5),
             0.004 * exp(-2.5), 0.004 * exp(-2.5), 0.004, 0.004],
            [0.004 * exp(-sqrt(2) / 0.4), 0.004 * exp(-sqrt(2) / 0.4), 0.004 * exp(-2.5), 0.004 * exp(-2.5),
             0.004 * exp(-2.5), 0.004 * exp(-2.5), 0.004, 0.004]
        ]

        distII2 = self.connectivity2._get_neurons_dist(
            neuron_type1=NeuronTypes.I,
            neuron_type2=NeuronTypes.I,
            nr1=self.connectivity2.nr_neurons[NeuronTypes.I],
            nr2=self.connectivity2.nr_neurons[NeuronTypes.I],
            oscillators=oscillators2,
            neuron_oscillator_map=neuron_oscillator_map2
        )
        type_coupling_weights_II2 = self.connectivity2._compute_type_coupling_weights(
            dist=distII2,
            max_connect_strength=MAX_CONNECT_STRENGTH[(NeuronTypes.I, NeuronTypes.I)],
            spatial_const=SPATIAL_CONST[(NeuronTypes.I, NeuronTypes.I)]
        )
        type_coupling_weights_II2_expected = [
            [-0.015, -0.015 * exp(-1 / 0.3), -0.015 * exp(-1 / 0.3), -0.015 * exp(-sqrt(2) / 0.3)],
            [-0.015 * exp(-1 / 0.3), -0.015, -0.015 * exp(-sqrt(2) / 0.3), -0.015 * exp(-1 / 0.3)],
            [-0.015 * exp(-1 / 0.3), -0.015 * exp(-sqrt(2) / 0.3), -0.015, -0.015 * exp(-1 / 0.3)],
            [-0.015 * exp(-sqrt(2) / 0.3), -0.015 * exp(-1 / 0.3), -0.015 * exp(-1 / 0.3), -0.015]
        ]

        distEI2 = self.connectivity2._get_neurons_dist(
            neuron_type1=NeuronTypes.E,
            neuron_type2=NeuronTypes.I,
            nr1=self.connectivity2.nr_neurons[NeuronTypes.E],
            nr2=self.connectivity2.nr_neurons[NeuronTypes.I],
            oscillators=oscillators2,
            neuron_oscillator_map=neuron_oscillator_map2
        )
        type_coupling_weights_EI2 = self.connectivity2._compute_type_coupling_weights(
            dist=distEI2,
            max_connect_strength=MAX_CONNECT_STRENGTH[(NeuronTypes.E, NeuronTypes.I)],
            spatial_const=SPATIAL_CONST[(NeuronTypes.E, NeuronTypes.I)]
        )
        type_coupling_weights_EI2_expected = [
            [0.07, 0.07 * exp(-1 / 0.3), 0.07 * exp(-1 / 0.3), 0.07 * exp(-sqrt(2) / 0.3)],
            [0.07, 0.07 * exp(-1 / 0.3), 0.07 * exp(-1 / 0.3), 0.07 * exp(-sqrt(2) / 0.3)],

            [0.07 * exp(-1 / 0.3), 0.07, 0.07 * exp(-sqrt(2) / 0.3), 0.07 * exp(-1 / 0.3)],
            [0.07 * exp(-1 / 0.3), 0.07, 0.07 * exp(-sqrt(2) / 0.3), 0.07 * exp(-1 / 0.3)],

            [0.07 * exp(-1 / 0.3), 0.07 * exp(-sqrt(2) / 0.3), 0.07, 0.07 * exp(-1 / 0.3)],
            [0.07 * exp(-1 / 0.3), 0.07 * exp(-sqrt(2) / 0.3), 0.07, 0.07 * exp(-1 / 0.3)],

            [0.07 * exp(-sqrt(2) / 0.3), 0.07 * exp(-1 / 0.3), 0.07 * exp(-1 / 0.3), 0.07],
            [0.07 * exp(-sqrt(2) / 0.3), 0.07 * exp(-1 / 0.3), 0.07 * exp(-1 / 0.3), 0.07]
        ]

        distIE2 = self.connectivity2._get_neurons_dist(
            neuron_type1=NeuronTypes.I,
            neuron_type2=NeuronTypes.E,
            nr1=self.connectivity2.nr_neurons[NeuronTypes.I],
            nr2=self.connectivity2.nr_neurons[NeuronTypes.E],
            oscillators=oscillators2,
            neuron_oscillator_map=neuron_oscillator_map2
        )
        type_coupling_weights_IE2 = self.connectivity2._compute_type_coupling_weights(
            dist=distIE2,
            max_connect_strength=MAX_CONNECT_STRENGTH[(NeuronTypes.I, NeuronTypes.E)],
            spatial_const=SPATIAL_CONST[(NeuronTypes.I, NeuronTypes.E)]
        )
        type_coupling_weights_IE2_expected = [
            [-0.04, -0.04, -0.04 * exp(-1 / 0.3), -0.04 * exp(-1 / 0.3), -0.04 * exp(-1 / 0.3), -0.04 * exp(-1 / 0.3),
             -0.04 * exp(-sqrt(2) / 0.3), -0.04 * exp(-sqrt(2) / 0.3)],

            [-0.04 * exp(-1 / 0.3), -0.04 * exp(-1 / 0.3), -0.04, -0.04, -0.04 * exp(-sqrt(2) / 0.3),
             -0.04 * exp(-sqrt(2) / 0.3), -0.04 * exp(-1 / 0.3), -0.04 * exp(-1 / 0.3)],

            [-0.04 * exp(-1 / 0.3), -0.04 * exp(-1 / 0.3), -0.04 * exp(-sqrt(2) / 0.3), -0.04 * exp(-sqrt(2) / 0.3),
             -0.04, -0.04, -0.04 * exp(-1 / 0.3), -0.04 * exp(-1 / 0.3)],

            [-0.04 * exp(-sqrt(2) / 0.3), -0.04 * exp(-sqrt(2) / 0.3), -0.04 * exp(-1 / 0.3), -0.04 * exp(
                -1 / 0.3), -0.04 * exp(-1 / 0.3), -0.04 * exp(-1 / 0.3), -0.04, -0.04]
        ]

        assert np.array_equal(type_coupling_weights_EE2, type_coupling_weights_EE2_expected)
        assert np.array_equal(type_coupling_weights_II2, type_coupling_weights_II2_expected)
        assert np.array_equal(type_coupling_weights_EI2, type_coupling_weights_EI2_expected)
        assert np.array_equal(type_coupling_weights_IE2, type_coupling_weights_IE2_expected)

    def test_compute_coupling_weights(self):
        oscillators1, neuron_oscillator_map1 = self.connectivity1._assign_oscillators()
        all_coupling_weights1 = self.connectivity1.compute_coupling_weights(oscillators1, neuron_oscillator_map1)
        all_coupling_weights1_expected = [
            [0.004, 0.004, -0.04, -0.04],
            [0.004, 0.004, -0.04, -0.04],
            [0.07, 0.07, -0.015, -0.015],
            [0.07, 0.07, -0.015, -0.015]
        ]

        assert np.array_equal(all_coupling_weights1, all_coupling_weights1_expected)

        # -----------------------------------------------------------------------

        oscillators2, neuron_oscillator_map2 = self.connectivity2._assign_oscillators()
        all_coupling_weights2 = self.connectivity2.compute_coupling_weights(oscillators2, neuron_oscillator_map2)
        all_coupling_weights2_expected = [
            [0.004, 0.004, 0.004 * exp(-2.5), 0.004 * exp(-2.5), 0.004 * exp(-2.5), 0.004 * exp(-2.5),
             0.004 * exp(-sqrt(2) / 0.4), 0.004 * exp(-sqrt(2) / 0.4), -0.04, -0.04 * exp(-1 / 0.3),
             -0.04 * exp(-1 / 0.3), -0.04 * exp(-sqrt(2) / 0.3)],
            [0.004, 0.004, 0.004 * exp(-2.5), 0.004 * exp(-2.5), 0.004 * exp(-2.5), 0.004 * exp(-2.5),
             0.004 * exp(-sqrt(2) / 0.4), 0.004 * exp(-sqrt(2) / 0.4), -0.04, -0.04 * exp(-1 / 0.3),
             -0.04 * exp(-1 / 0.3), -0.04 * exp(-sqrt(2) / 0.3)],

            [0.004 * exp(-2.5), 0.004 * exp(-2.5), 0.004, 0.004, 0.004 * exp(-sqrt(2) / 0.4),
             0.004 * exp(-sqrt(2) / 0.4), 0.004 * exp(-2.5), 0.004 * exp(-2.5), -0.04 * exp(-1 / 0.3), -0.04,
             -0.04 * exp(-sqrt(2) / 0.3), -0.04 * exp(-1 / 0.3)],
            [0.004 * exp(-2.5), 0.004 * exp(-2.5), 0.004, 0.004, 0.004 * exp(-sqrt(2) / 0.4),
             0.004 * exp(-sqrt(2) / 0.4), 0.004 * exp(-2.5), 0.004 * exp(-2.5), -0.04 * exp(-1 / 0.3), -0.04,
             -0.04 * exp(-sqrt(2) / 0.3), -0.04 * exp(-1 / 0.3)],

            [0.004 * exp(-2.5), 0.004 * exp(-2.5), 0.004 * exp(-sqrt(2) / 0.4), 0.004 * exp(-sqrt(2) / 0.4), 0.004,
             0.004, 0.004 * exp(-2.5), 0.004 * exp(-2.5),  -0.04 * exp(-1 / 0.3), -0.04 * exp(-sqrt(2) / 0.3), -0.04,
             -0.04 * exp(-1 / 0.3)],
            [0.004 * exp(-2.5), 0.004 * exp(-2.5), 0.004 * exp(-sqrt(2) / 0.4), 0.004 * exp(-sqrt(2) / 0.4), 0.004,
             0.004, 0.004 * exp(-2.5), 0.004 * exp(-2.5),  -0.04 * exp(-1 / 0.3), -0.04 * exp(-sqrt(2) / 0.3), -0.04,
             -0.04 * exp(-1 / 0.3)],

            [0.004 * exp(-sqrt(2) / 0.4), 0.004 * exp(-sqrt(2) / 0.4), 0.004 * exp(-2.5), 0.004 * exp(-2.5),
             0.004 * exp(-2.5), 0.004 * exp(-2.5), 0.004, 0.004, -0.04 * exp(-sqrt(2) / 0.3), -0.04 * exp(-1 / 0.3),
             -0.04 * exp(-1 / 0.3), -0.04],
            [0.004 * exp(-sqrt(2) / 0.4), 0.004 * exp(-sqrt(2) / 0.4), 0.004 * exp(-2.5), 0.004 * exp(-2.5),
             0.004 * exp(-2.5), 0.004 * exp(-2.5), 0.004, 0.004, -0.04 * exp(-sqrt(2) / 0.3), -0.04 * exp(-1 / 0.3),
             -0.04 * exp(-1 / 0.3), -0.04],

            [0.07, 0.07, 0.07 * exp(-1 / 0.3), 0.07 * exp(-1 / 0.3), 0.07 * exp(-1 / 0.3), 0.07 * exp(-1 / 0.3),
             0.07 * exp(-sqrt(2) / 0.3), 0.07 * exp(-sqrt(2) / 0.3), -0.015, -0.015 * exp(-1 / 0.3),
             -0.015 * exp(-1 / 0.3), -0.015 * exp(-sqrt(2) / 0.3)],

            [0.07 * exp(-1 / 0.3), 0.07 * exp(-1 / 0.3), 0.07, 0.07, 0.07 * exp(-sqrt(2) / 0.3),
             0.07 * exp(-sqrt(2) / 0.3), 0.07 * exp(-1 / 0.3), 0.07 * exp(-1 / 0.3), -0.015 * exp(-1 / 0.3), -0.015,
             -0.015 * exp(-sqrt(2) / 0.3), -0.015 * exp(-1 / 0.3)],

            [0.07 * exp(-1 / 0.3), 0.07 * exp(-1 / 0.3), 0.07 * exp(-sqrt(2) / 0.3), 0.07 * exp(-sqrt(2) / 0.3), 0.07,
             0.07, 0.07 * exp(-1 / 0.3), 0.07 * exp(-1 / 0.3), -0.015 * exp(-1 / 0.3), -0.015 * exp(-sqrt(2) / 0.3),
             -0.015, -0.015 * exp(-1 / 0.3)],

            [0.07 * exp(-sqrt(2) / 0.3), 0.07 * exp(-sqrt(2) / 0.3), 0.07 * exp(-1 / 0.3), 0.07 * exp(-1 / 0.3),
             0.07 * exp(-1 / 0.3), 0.07 * exp(-1 / 0.3), 0.07, 0.07, -0.015 * exp(-sqrt(2) / 0.3),
             -0.015 * exp(-1 / 0.3), -0.015 * exp(-1 / 0.3), -0.015]
        ]

        assert np.array_equal(all_coupling_weights1, all_coupling_weights1_expected)
        assert np.array_equal(all_coupling_weights2, all_coupling_weights2_expected)