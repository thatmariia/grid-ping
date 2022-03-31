from src.OscillatoryNetwork import *
from src.InputStimulus import *


if __name__ == "__main__":

    nr_circuits = 4

    stimulus = InputStimulus(
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
        nr_circuits=nr_circuits,
        slope=0.172,
        intercept=-0.25,
        min_diam_rf=1
    )
    stimulus.plot_stimulus(stimulus.stimulus, filename="full-stimulus")
    stimulus.plot_stimulus(stimulus.stimulus_patch, filename="stimulus-patch")

    oscillatory_network = OscillatoryNetwork(
        stimulus=stimulus.current,
        nr_excitatory=4,
        nr_inhibitory=4,
        nr_ping_networks=nr_circuits
    )
    oscillatory_network.run_simulation(
        simulation_time=8,
        dt=1
    )
