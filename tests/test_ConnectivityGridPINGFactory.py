from src.params.ParamsPING import *
from src.params.ParamsConnectivity import *
from src.izhikevich_simulation.GridGeometryFactory import *
from src.izhikevich_simulation.ConnectivityGridPINGFactory import *
from src.izhikevich_simulation.Connectivity import *

import numpy as np


class TestConnectivityGridPINGFactory:
    params_ping = ParamsPING(
        nr_excitatory=8,
        nr_inhibitory=4,
        nr_ping_networks=4
    )

    cortical_distances = np.array([
        [0, 1, 2, 3],
        [1, 0, 3, 1],
        [2, 3, 0, 2],
        [3, 1, 2, 0]
    ])

    params_connectivity = ParamsConnectivity(
        max_connect_strength_EE=0.5,
        max_connect_strength_EI=0.5,
        max_connect_strength_IE=0.5,
        max_connect_strength_II=0.5,
        spatial_const_EE=0.5,
        spatial_const_EI=0.5,
        spatial_const_IE=0.5,
        spatial_const_II=0.5
    )

    def test_compute_type_coupling_weights(self):

        conectivity_grid_ping_factory = ConnectivityGridPINGFactory()

        type_cw_exp = 0.5 * np.exp(-np.array([1, 2]) / 0.5)
        type_cw = conectivity_grid_ping_factory._compute_type_coupling_weights(
            np.array([1, 2]), 0.5, 0.5
        )

        assert np.all(type_cw_exp == type_cw)

    def test_create(self):
        connectivity = ConnectivityGridPINGFactory().create(
            self.params_ping, self.params_connectivity, self.cortical_distances
        )

        grid_geometry = GridGeometryFactory().create(self.params_ping, self.cortical_distances)
        coupling_weights = np.zeros((self.params_ping.nr_neurons["total"], self.params_ping.nr_neurons["total"]))
        for nts in list(product([NeuronTypes.EX, NeuronTypes.IN], repeat=2)):

            dist = grid_geometry.neuron_distances[self.params_ping.neur_slice[nts[0]], self.params_ping.neur_slice[nts[1]]]
            types_coupling_weights = 0.5 * np.exp(-dist / 0.5)

            if nts[0] == nts[1]:
                coupling_weights[self.params_ping.neur_slice[nts[0]], self.params_ping.neur_slice[nts[1]]] = \
                    types_coupling_weights
            else:
                coupling_weights[self.params_ping.neur_slice[nts[1]], self.params_ping.neur_slice[nts[0]]] = \
                    types_coupling_weights.T
        connectivity_exp = Connectivity(
            self.params_ping, coupling_weights, grid_geometry
        )

        assert connectivity == connectivity_exp


