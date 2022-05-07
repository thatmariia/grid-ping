from src.GridConnectivity import *
from src.PINGNetworkNeurons import *
from src.NeuronTypes import *
from src.constants import *

import numpy as np
from math import sqrt, exp


class TestGridConnectivity:

    connectivity1 = GridConnectivity(
        nr_neurons={
            NeuronTypes.EX: 2,
            NeuronTypes.IN: 2
        },
        nr_ping_networks=1
    )

    connectivity2 = GridConnectivity(
        nr_neurons={
            NeuronTypes.EX: 8,
            NeuronTypes.IN: 4
        },
        nr_ping_networks=4
    )

    def test_assign_ping_networks(self):
        ping_networks1, neuron_ping_map1 = self.connectivity1._assign_ping_networks()
        ping_networks1_expected = [
            PINGNetworkNeurons((0, 0), [0, 1], [0, 1])
        ]
        neuron_ping_map1_expected = {
            NeuronTypes.EX: {0: 0, 1: 0},
            NeuronTypes.IN: {0: 0, 1: 0},
        }
        assert ping_networks1 == ping_networks1_expected
        assert neuron_ping_map1 == neuron_ping_map1_expected

        # -----------------------------------------------------------------------

        ping_networks2, neuron_ping_map2 = self.connectivity2._assign_ping_networks()
        ping_networks2_expected = [
            PINGNetworkNeurons((0, 0), [0, 1], [0]),
            PINGNetworkNeurons((0, 1), [2, 3], [1]),
            PINGNetworkNeurons((1, 0), [4, 5], [2]),
            PINGNetworkNeurons((1, 1), [6, 7], [3]),
        ]
        neuron_ping_map2_expected = {
            NeuronTypes.EX: {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3},
            NeuronTypes.IN: {0: 0, 1: 1, 2: 2, 3: 3},
        }
        assert ping_networks2 == ping_networks2_expected
        assert neuron_ping_map2 == neuron_ping_map2_expected

    def test_get_neurons_dist(self):
        ping_networks1, neuron_ping_map1 = self.connectivity1._assign_ping_networks()
        distEE1 = self.connectivity1._get_neurons_dist(
            neuron_type1=NeuronTypes.EX,
            neuron_type2=NeuronTypes.EX,
            nr1=self.connectivity1._nr_neurons[NeuronTypes.EX],
            nr2=self.connectivity1._nr_neurons[NeuronTypes.EX],
            ping_networks=ping_networks1,
            neuron_ping_map=neuron_ping_map1
        )
        distEE1_expected = [[0.0, 0.0], [0.0, 0.0]]

        distII1 = self.connectivity1._get_neurons_dist(
            neuron_type1=NeuronTypes.IN,
            neuron_type2=NeuronTypes.IN,
            nr1=self.connectivity1._nr_neurons[NeuronTypes.IN],
            nr2=self.connectivity1._nr_neurons[NeuronTypes.IN],
            ping_networks=ping_networks1,
            neuron_ping_map=neuron_ping_map1
        )
        distII1_expected = [[0.0, 0.0], [0.0, 0.0]]

        distEI1 = self.connectivity1._get_neurons_dist(
            neuron_type1=NeuronTypes.EX,
            neuron_type2=NeuronTypes.IN,
            nr1=self.connectivity1._nr_neurons[NeuronTypes.EX],
            nr2=self.connectivity1._nr_neurons[NeuronTypes.IN],
            ping_networks=ping_networks1,
            neuron_ping_map=neuron_ping_map1
        )
        distEI1_expected = [[0.0, 0.0], [0.0, 0.0]]

        distIE1 = self.connectivity1._get_neurons_dist(
            neuron_type1=NeuronTypes.IN,
            neuron_type2=NeuronTypes.EX,
            nr1=self.connectivity1._nr_neurons[NeuronTypes.IN],
            nr2=self.connectivity1._nr_neurons[NeuronTypes.EX],
            ping_networks=ping_networks1,
            neuron_ping_map=neuron_ping_map1
        )
        distIE1_expected = [[0, 0], [0, 0]]

        assert np.array_equal(distEE1, distEE1_expected)
        assert np.array_equal(distII1, distII1_expected)
        assert np.array_equal(distEI1, distEI1_expected)
        assert np.array_equal(distIE1, distIE1_expected)

        # -----------------------------------------------------------------------

        ping_networks2, neuron_ping_map2 = self.connectivity2._assign_ping_networks()
        distEE2 = self.connectivity2._get_neurons_dist(
            neuron_type1=NeuronTypes.EX,
            neuron_type2=NeuronTypes.EX,
            nr1=self.connectivity2._nr_neurons[NeuronTypes.EX],
            nr2=self.connectivity2._nr_neurons[NeuronTypes.EX],
            ping_networks=ping_networks2,
            neuron_ping_map=neuron_ping_map2
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
            neuron_type1=NeuronTypes.IN,
            neuron_type2=NeuronTypes.IN,
            nr1=self.connectivity2._nr_neurons[NeuronTypes.IN],
            nr2=self.connectivity2._nr_neurons[NeuronTypes.IN],
            ping_networks=ping_networks2,
            neuron_ping_map=neuron_ping_map2
        )
        distII2_expected = [
            [0, 1, 1, sqrt(2)],
            [1, 0, sqrt(2), 1],
            [1, sqrt(2), 0, 1],
            [sqrt(2), 1, 1, 0]
        ]

        distEI2 = self.connectivity2._get_neurons_dist(
            neuron_type1=NeuronTypes.EX,
            neuron_type2=NeuronTypes.IN,
            nr1=self.connectivity2._nr_neurons[NeuronTypes.EX],
            nr2=self.connectivity2._nr_neurons[NeuronTypes.IN],
            ping_networks=ping_networks2,
            neuron_ping_map=neuron_ping_map2
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
            neuron_type1=NeuronTypes.IN,
            neuron_type2=NeuronTypes.EX,
            nr1=self.connectivity2._nr_neurons[NeuronTypes.IN],
            nr2=self.connectivity2._nr_neurons[NeuronTypes.EX],
            ping_networks=ping_networks2,
            neuron_ping_map=neuron_ping_map2
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
        ping_networks1, neuron_ping_map1 = self.connectivity1._assign_ping_networks()

        distEE1 = self.connectivity1._get_neurons_dist(
            neuron_type1=NeuronTypes.EX,
            neuron_type2=NeuronTypes.EX,
            nr1=self.connectivity1._nr_neurons[NeuronTypes.EX],
            nr2=self.connectivity1._nr_neurons[NeuronTypes.EX],
            ping_networks=ping_networks1,
            neuron_ping_map=neuron_ping_map1
        )
        type_coupling_weights_EE1 = self.connectivity1._compute_type_coupling_weights(
            dist=distEE1,
            max_connect_strength=MAX_CONNECT_STRENGTH[(NeuronTypes.EX, NeuronTypes.EX)],
            spatial_const=SPATIAL_CONST[(NeuronTypes.EX, NeuronTypes.EX)]
        )
        type_coupling_weights_EE1_expected = [
            [0.004, 0.004],
            [0.004, 0.004]
        ]

        distII1 = self.connectivity1._get_neurons_dist(
            neuron_type1=NeuronTypes.IN,
            neuron_type2=NeuronTypes.IN,
            nr1=self.connectivity1._nr_neurons[NeuronTypes.IN],
            nr2=self.connectivity1._nr_neurons[NeuronTypes.IN],
            ping_networks=ping_networks1,
            neuron_ping_map=neuron_ping_map1
        )
        type_coupling_weights_II1 = self.connectivity1._compute_type_coupling_weights(
            dist=distII1,
            max_connect_strength=MAX_CONNECT_STRENGTH[(NeuronTypes.IN, NeuronTypes.IN)],
            spatial_const=SPATIAL_CONST[(NeuronTypes.IN, NeuronTypes.IN)]
        )
        type_coupling_weights_II1_expected = [
            [-0.015, -0.015],
            [-0.015, -0.015]
        ]

        distEI1 = self.connectivity1._get_neurons_dist(
            neuron_type1=NeuronTypes.EX,
            neuron_type2=NeuronTypes.IN,
            nr1=self.connectivity1._nr_neurons[NeuronTypes.EX],
            nr2=self.connectivity1._nr_neurons[NeuronTypes.IN],
            ping_networks=ping_networks1,
            neuron_ping_map=neuron_ping_map1
        )
        type_coupling_weights_EI1 = self.connectivity1._compute_type_coupling_weights(
            dist=distEI1,
            max_connect_strength=MAX_CONNECT_STRENGTH[(NeuronTypes.EX, NeuronTypes.IN)],
            spatial_const=SPATIAL_CONST[(NeuronTypes.EX, NeuronTypes.IN)]
        )
        type_coupling_weights_EI1_expected = [
            [0.07, 0.07],
            [0.07, 0.07]
        ]

        distIE1 = self.connectivity1._get_neurons_dist(
            neuron_type1=NeuronTypes.IN,
            neuron_type2=NeuronTypes.EX,
            nr1=self.connectivity1._nr_neurons[NeuronTypes.IN],
            nr2=self.connectivity1._nr_neurons[NeuronTypes.EX],
            ping_networks=ping_networks1,
            neuron_ping_map=neuron_ping_map1
        )
        type_coupling_weights_IE1 = self.connectivity1._compute_type_coupling_weights(
            dist=distIE1,
            max_connect_strength=MAX_CONNECT_STRENGTH[(NeuronTypes.IN, NeuronTypes.EX)],
            spatial_const=SPATIAL_CONST[(NeuronTypes.IN, NeuronTypes.EX)]
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

        ping_networks2, neuron_ping_map2 = self.connectivity2._assign_ping_networks()

        distEE2 = self.connectivity2._get_neurons_dist(
            neuron_type1=NeuronTypes.EX,
            neuron_type2=NeuronTypes.EX,
            nr1=self.connectivity2._nr_neurons[NeuronTypes.EX],
            nr2=self.connectivity2._nr_neurons[NeuronTypes.EX],
            ping_networks=ping_networks2,
            neuron_ping_map=neuron_ping_map2
        )
        type_coupling_weights_EE2 = self.connectivity2._compute_type_coupling_weights(
            dist=distEE2,
            max_connect_strength=MAX_CONNECT_STRENGTH[(NeuronTypes.EX, NeuronTypes.EX)],
            spatial_const=SPATIAL_CONST[(NeuronTypes.EX, NeuronTypes.EX)]
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
            neuron_type1=NeuronTypes.IN,
            neuron_type2=NeuronTypes.IN,
            nr1=self.connectivity2._nr_neurons[NeuronTypes.IN],
            nr2=self.connectivity2._nr_neurons[NeuronTypes.IN],
            ping_networks=ping_networks2,
            neuron_ping_map=neuron_ping_map2
        )
        type_coupling_weights_II2 = self.connectivity2._compute_type_coupling_weights(
            dist=distII2,
            max_connect_strength=MAX_CONNECT_STRENGTH[(NeuronTypes.IN, NeuronTypes.IN)],
            spatial_const=SPATIAL_CONST[(NeuronTypes.IN, NeuronTypes.IN)]
        )
        type_coupling_weights_II2_expected = [
            [-0.015, -0.015 * exp(-1 / 0.3), -0.015 * exp(-1 / 0.3), -0.015 * exp(-sqrt(2) / 0.3)],
            [-0.015 * exp(-1 / 0.3), -0.015, -0.015 * exp(-sqrt(2) / 0.3), -0.015 * exp(-1 / 0.3)],
            [-0.015 * exp(-1 / 0.3), -0.015 * exp(-sqrt(2) / 0.3), -0.015, -0.015 * exp(-1 / 0.3)],
            [-0.015 * exp(-sqrt(2) / 0.3), -0.015 * exp(-1 / 0.3), -0.015 * exp(-1 / 0.3), -0.015]
        ]

        distEI2 = self.connectivity2._get_neurons_dist(
            neuron_type1=NeuronTypes.EX,
            neuron_type2=NeuronTypes.IN,
            nr1=self.connectivity2._nr_neurons[NeuronTypes.EX],
            nr2=self.connectivity2._nr_neurons[NeuronTypes.IN],
            ping_networks=ping_networks2,
            neuron_ping_map=neuron_ping_map2
        )
        type_coupling_weights_EI2 = self.connectivity2._compute_type_coupling_weights(
            dist=distEI2,
            max_connect_strength=MAX_CONNECT_STRENGTH[(NeuronTypes.EX, NeuronTypes.IN)],
            spatial_const=SPATIAL_CONST[(NeuronTypes.EX, NeuronTypes.IN)]
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
            neuron_type1=NeuronTypes.IN,
            neuron_type2=NeuronTypes.EX,
            nr1=self.connectivity2._nr_neurons[NeuronTypes.IN],
            nr2=self.connectivity2._nr_neurons[NeuronTypes.EX],
            ping_networks=ping_networks2,
            neuron_ping_map=neuron_ping_map2
        )
        type_coupling_weights_IE2 = self.connectivity2._compute_type_coupling_weights(
            dist=distIE2,
            max_connect_strength=MAX_CONNECT_STRENGTH[(NeuronTypes.IN, NeuronTypes.EX)],
            spatial_const=SPATIAL_CONST[(NeuronTypes.IN, NeuronTypes.EX)]
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
        ping_networks1, neuron_ping_map1 = self.connectivity1._assign_ping_networks()
        all_coupling_weights1 = self.connectivity1._compute_coupling_weights(ping_networks1, neuron_ping_map1)
        all_coupling_weights1_expected = [
            [0.004, 0.004, -0.04, -0.04],
            [0.004, 0.004, -0.04, -0.04],
            [0.07, 0.07, -0.015, -0.015],
            [0.07, 0.07, -0.015, -0.015]
        ]

        # -----------------------------------------------------------------------

        ping_networks2, neuron_ping_map2 = self.connectivity2._assign_ping_networks()
        all_coupling_weights2 = self.connectivity2._compute_coupling_weights(ping_networks2, neuron_ping_map2)
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
