import numpy as np

from src.params.ParamsPING import *
from src.izhikevich_simulation.GridGeometryFactory import *
from src.izhikevich_simulation.PINGNetworkNeurons import *
from src.izhikevich_simulation.GridGeometry import *


class TestGridGeometryFactory:

    params_ping = ParamsPING(
        nr_excitatory=8,
        nr_inhibitory=4,
        nr_ping_networks=4
    )

    cortical_distances1 = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])

    cortical_distances2 = np.array([
        [0, 1, 2, 3],
        [1, 0, 3, 1],
        [2, 3, 0, 2],
        [3, 1, 2, 0]
    ])

    def test_refresh_neuron_ping_ids(self):
        grid_geometry_factory = GridGeometryFactory()

        # Test 1

        ping_id1 = 0
        ping_ids1 = {}
        refreshed_neuron_ping_ids1 = grid_geometry_factory._refresh_neuron_ping_ids(ping_id1, ping_ids1, self.params_ping)
        refreshed_neuron_ping_ids1_exp = {
            0: {
                NeuronTypes.EX: [0, 1],
                NeuronTypes.IN: [8]
            }
        }

        assert refreshed_neuron_ping_ids1 == refreshed_neuron_ping_ids1_exp

        # Test 2

        ping_id2 = 2
        refreshed_neuron_ping_ids2 = grid_geometry_factory._refresh_neuron_ping_ids(ping_id2, ping_ids1, self.params_ping)
        refreshed_neuron_ping_ids2_exp = {
            0: {
                NeuronTypes.EX: [0, 1],
                NeuronTypes.IN: [8]
            },
            2: {
                NeuronTypes.EX: [4, 5],
                NeuronTypes.IN: [10]
            }
        }

        assert refreshed_neuron_ping_ids2 == refreshed_neuron_ping_ids2_exp

    def test_create(self):

        # Test 1

        distances1_exp = np.zeros((self.params_ping.nr_neurons["total"], self.params_ping.nr_neurons["total"]))
        ping_networks_exp = [
            PINGNetworkNeurons(
                grid_location=(0, 0),
                ids_ex=[0, 1],
                ids_in=[8]
            ),
            PINGNetworkNeurons(
                grid_location=(0, 1),
                ids_ex=[2, 3],
                ids_in=[9]
            ),
            PINGNetworkNeurons(
                grid_location=(1, 0),
                ids_ex=[4, 5],
                ids_in=[10]
            ),
            PINGNetworkNeurons(
                grid_location=(1, 1),
                ids_ex=[6, 7],
                ids_in=[11]
            )
        ]

        grid_geometry1_exp = GridGeometry(
            neuron_distances=distances1_exp,
            ping_networks=ping_networks_exp
        )
        grid_geometry1 = GridGeometryFactory().create(self.params_ping, self.cortical_distances1)

        assert grid_geometry1 == grid_geometry1_exp

        # Test 2

        distances2_exp = np.array([
            [0, 0, 1, 1, 2, 2, 3, 3, 0, 1, 2, 3],
            [0, 0, 1, 1, 2, 2, 3, 3, 0, 1, 2, 3],
            [1, 1, 0, 0, 3, 3, 1, 1, 1, 0, 3, 1],
            [1, 1, 0, 0, 3, 3, 1, 1, 1, 0, 3, 1],
            [2, 2, 3, 3, 0, 0, 2, 2, 2, 3, 0, 2],
            [2, 2, 3, 3, 0, 0, 2, 2, 2, 3, 0, 2],
            [3, 3, 1, 1, 2, 2, 0, 0, 3, 1, 2, 0],
            [3, 3, 1, 1, 2, 2, 0, 0, 3, 1, 2, 0],
            [0, 0, 1, 1, 2, 2, 3, 3, 0, 1, 2, 3],
            [1, 1, 0, 0, 3, 3, 1, 1, 1, 0, 3, 1],
            [2, 2, 3, 3, 0, 0, 2, 2, 2, 3, 0, 2],
            [3, 3, 1, 1, 2, 2, 0, 0, 3, 1, 2, 0]
        ])

        grid_geometry2_exp = GridGeometry(
            neuron_distances=distances2_exp,
            ping_networks=ping_networks_exp
        )
        grid_geometry2 = GridGeometryFactory().create(self.params_ping, self.cortical_distances2)

        assert grid_geometry2 == grid_geometry2_exp





