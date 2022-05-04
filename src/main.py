from src.StimulusFactory import *
from src.ConnectivityGridPINGFactory import *
from src.CurrentComponentsGridPING import *
from src.IzhikevichNetworkSimulator import *

if __name__ == "__main__":

    nr_ping_networks = 4

    stimulus = CurrentStimulusFactory().create(
        spatial_freq=5.7,
        vlum=0.5,
        diameter_dg=0.7,
        diameter=48,
        dist_scale=1,
        full_width_dg=33.87,
        full_height_dg=27.09,
        contrast_range=0.01,
        figure_width_dg=9,
        figure_height_dg=5,
        figure_ecc_dg=7,
        patch_size_dg=4.9,
        nr_ping_networks=nr_ping_networks,
        slope=0.172,
        intercept=-0.25,
        min_diam_rf=1
    )

    stimulus_locations = stimulus.extract_stimulus_location()

    connectivity = ConnectivityGridPINGFactory().create(
        nr_excitatory=4,
        nr_inhibitory=4,
        nr_ping_networks=nr_ping_networks,
        cortical_coords=stimulus_locations.cortical_coords
    )
    neural_model = CurrentComponentsGridPING(
        connectivity=connectivity,
        stimulus_currents=stimulus.stimulus_currents
    )
    simulation_outcome = IzhikevichNetworkSimulator(
        current_components=neural_model
    ).simulate(
        simulation_time=8,
        dt=1
    )

    # oscillatory_network = OscillatoryNetwork(
    #     stimulus=stimulus.stimulus_currents,
    #     nr_excitatory=4,
    #     nr_inhibitory=4,
    #     nr_ping_networks=nr_ping_networks
    # )
    # oscillatory_network.run_simulation(
    #     cortical_coords=stimulus_locations.cortical_coords,
    #     simulation_time=8,
    #     dt=1
    # )
