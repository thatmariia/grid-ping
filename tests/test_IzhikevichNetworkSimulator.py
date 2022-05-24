from src.izhikevich_simulation.IzhikevichNetworkSimulator import *
from src.izhikevich_simulation.IzhikevichNetworkOutcome import *
from src.params.ParamsIzhikevich import *
from src.izhikevich_simulation.CurrentComponentsGridPING import *
from src.izhikevich_simulation.ConnectivityGridPINGFactory import *


class TestIzhikevichNetworkSimulator:

    potentials = np.array([-100, 30, -50, 0, 0, -50, 30, -100, -60, 0, 30, 0])
    recovery = np.array([5, -10, -15, -10, -10, -15, -10, 5, -10, -5, -10, -5])
    currents = np.array([1, 1, 2, 2, 3, 3, 4, 4, 0, 0, 0, 0])

    params_izhi = ParamsIzhikevich(
        peak_potential=30,
        alpha_E=0.5,
        beta_E=0.5,
        gamma_E=-10,
        zeta_E=10,
        alpha_I=0.5,
        beta_I=0.5,
        gamma_I=-10,
        zeta_I=10
    )

    izhi_alpha = np.ones(12) * 0.5
    izhi_beta = np.ones(12) * 0.5

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
    current_components_grid_ping = CurrentComponentsGridPING(
        connectivity=ConnectivityGridPINGFactory().create(params_ping, params_connectivity, cortical_distances),
        params_synaptic=params_synaptic,
        stimulus_currents=np.array([1, 2, 3, 4])
    )


    def test_get_change_in_potentials(self):
        network_simulator = IzhikevichNetworkSimulator(self.params_izhi, self.current_components_grid_ping)
        change_in_potentials = network_simulator._get_change_in_potentials(self.potentials, self.recovery, self.currents)
        change_in_potentials_exp = np.array([
            36, 337, 7, 152, 153, 8, 340, 39, -6, 145, 336, 145
        ])

        assert np.all(change_in_potentials == change_in_potentials_exp)

    def test_get_change_in_recovery(self):
        network_simulator = IzhikevichNetworkSimulator(self.params_izhi, self.current_components_grid_ping)
        change_in_recovery = network_simulator._get_change_in_recovery(self.potentials, self.recovery, self.izhi_alpha, self.izhi_beta)
        change_in_recovery_exp = np.array([
            -27.5, 12.5, -5, 5, 5, -5, 12.5, -27.5, -10, 2.5, 12.5, 2.5
        ])

        assert np.all(change_in_recovery == change_in_recovery_exp)