from src.izhikevich_simulation.CurrentComponentsGridPING import *
from src.params.ParamsSynaptic import *
from src.params.ParamsConnectivity import *
from src.izhikevich_simulation.Connectivity import *
from src.izhikevich_simulation.ConnectivityGridPINGFactory import *

import numpy as np


class TestCurrentComponentsGridPING:
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

    params_synaptic = ParamsSynaptic(
        rise_E=1,
        decay_E=2,
        rise_I=2,
        decay_I=4,
        conductance_EE=0.5,
        conductance_EI=0.5,
        conductance_IE=0.5,
        conductance_II=0.5,
        reversal_potential_E=-80,
        reversal_potential_I=0
    )

    stimulus_currents = np.array([1, 2, 3, 4])

    dt = 1

    potentials = np.array([-100, 30, -50, 0, 0, -50, 30, -100, -60, 0, 30, 0])

    def test_get_stimulus_input(self):
        current_components_grid_ping = CurrentComponentsGridPING(
            connectivity=ConnectivityGridPINGFactory().create(self.params_ping, self.params_connectivity, self.cortical_distances),
            params_synaptic=self.params_synaptic,
            stimulus_currents=self.stimulus_currents
        )
        stimulus_input = current_components_grid_ping._get_stimulus_input()
        stimulus_input_exp = np.array([
            1, 1, 2, 2, 3, 3, 4, 4, 0, 0, 0, 0
        ])

        assert np.all(stimulus_input == stimulus_input_exp)

    def test_get_gatings(self):
        current_components_grid_ping = CurrentComponentsGridPING(
            connectivity=ConnectivityGridPINGFactory().create(self.params_ping, self.params_connectivity,
                                                              self.cortical_distances),
            params_synaptic=self.params_synaptic,
            stimulus_currents=self.stimulus_currents
        )
        gatings = current_components_grid_ping._get_gatings(self.dt, self.potentials)
        gatings_exp = np.array([
            1 - np.tanh(25),
            1 + np.tanh(15 / 2),
            1 - np.tanh(25 / 2),
            1,
            1,
            1 - np.tanh(25 / 2),
            1 + np.tanh(15 / 2),
            1 - np.tanh(25),
            2 * (1 - np.tanh(15)),
            2,
            2 * (1 + np.tanh(15 / 2)),
            2
        ])

        assert np.all(gatings == gatings_exp)

    def test_get_synaptic_currents(self):
        current_components_grid_ping = CurrentComponentsGridPING(
            connectivity=ConnectivityGridPINGFactory().create(self.params_ping, self.params_connectivity,
                                                              self.cortical_distances),
            params_synaptic=self.params_synaptic,
            stimulus_currents=self.stimulus_currents
        )
        synaptic_currents = current_components_grid_ping.get_synaptic_currents(self.dt, self.potentials)
        current_components_grid_ping._gatings = np.zeros(self.params_ping.nr_neurons["total"])
        gatings = current_components_grid_ping._get_gatings(self.dt, self.potentials)
        ex_gatings = gatings[self.params_ping.neur_slice[NeuronTypes.EX]]
        in_gatings = gatings[self.params_ping.neur_slice[NeuronTypes.IN]]
        in_conductance = 0.5 * sum(in_gatings) / self.params_ping.nr_neurons[NeuronTypes.IN]
        ex_conductance = 0.5 * sum(ex_gatings) / self.params_ping.nr_neurons[NeuronTypes.EX]
        synaptic_currents_exp = np.array([
            in_conductance * (-100 - 0) + ex_conductance * (-100 + 80),
            in_conductance * (30 - 0) + ex_conductance * (30 + 80),
            in_conductance * (-50 - 0) + ex_conductance * (-50 + 80),
            in_conductance * (0 - 0) + ex_conductance * (0 + 80),
            in_conductance * (0 - 0) + ex_conductance * (0 + 80),
            in_conductance * (-50 - 0) + ex_conductance * (-50 + 80),
            in_conductance * (30 - 0) + ex_conductance * (30 + 80),
            in_conductance * (-100 - 0) + ex_conductance * (-100 + 80),
            in_conductance * (-60 - 0) + ex_conductance * (-60 + 80),
            in_conductance * (0 - 0) + ex_conductance * (0 + 80),
            in_conductance * (30 - 0) + ex_conductance * (30 + 80),
            in_conductance * (0 - 0) + ex_conductance * (0 + 80),
        ])

        assert np.all(synaptic_currents == synaptic_currents_exp)





