from src.OscillatoryNetwork import *
from src.InputStimulus import *


if __name__ == "__main__":

    nr_circuits = 4

    stimulus = InputStimulus(
        dist_scale=2,
        contrast_range=0.5,
        spatial_freq=5.7,
        diameter=0.7,
        side_length=7,
        grating_res=50,
        patch_res=480,
        nr_circuits=nr_circuits,
        slope=0,
        intercept=0,
        min_diam_rf=1
    )
    # stimulus.plot_stimulus(stimulus.stimulus_patch, filename="stim-patch")

    oscillatory_network = OscillatoryNetwork(
        stimulus=stimulus.current,
        nr_excitatory=8,
        nr_inhibitory=4,
        nr_ping_networks=nr_circuits
    )
    oscillatory_network.run_simulation(
        simulation_time=8,
        dt=1
    )


