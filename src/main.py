from src.OscillatoryNetwork import *
from src.GaborTextureStimulus import *


if __name__ == "__main__":
    # oscillatory_network = OscillatoryNetwork(
    #     nr_excitatory=8,
    #     nr_inhibitory=4,
    #     nr_ping_networks=4
    # )
    # oscillatory_network.run_simulation(
    #     simulation_time=8,
    #     dt=1
    # )
    stimulus = GaborTextureStimulus(
        dist_scale=2,
        contrast_range=0.5,
        spatial_freq=5.7,
        diameter=0.7,
        side_length=7,
        grating_res=50,
        patch_res=480
    )
    #stimulus.plot_stimulus(stimulus.stimulus_patch, filename="stim-patch")

