from src.ParamsPING import *
from src.ParamsGaborStimulus import *
from src.ParamsReceptiveField import *

from src.StimulusFactory import *
from src.ConnectivityGridPINGFactory import *
from src.CurrentComponentsGridPING import *
from src.IzhikevichNetworkSimulator import *

# TODO:: plot lc

if __name__ == "__main__":

    params_ping = ParamsPING(
        nr_excitatory=200,
        nr_inhibitory=50,
        nr_ping_networks=25
    )
    params_gabor = ParamsGaborStimulus(
        spatial_freq=5.7,
        vlum=0.5,
        diameter_dg=1,
        diameter=50,
        dist_scale=1,
        full_width_dg=40,
        full_height_dg=27.09,
        contrast_range=0.01,
        figure_width_dg=10,
        figure_height_dg=5,
        figure_ecc_dg=7,
        patch_size_dg=5
    )
    params_rf = ParamsReceptiveField(
        slope=0.172,
        intercept=-0.25,
        min_diam_rf=1
    )

    stimulus = StimulusFactory().create(params_gabor, params_rf, params_ping)

    stimulus_locations = stimulus.extract_stimulus_location()

    connectivity = ConnectivityGridPINGFactory().create(
        params_ping=params_ping,
        cortical_coords=stimulus_locations.cortical_coords
    )
    neural_model = CurrentComponentsGridPING(
        connectivity=connectivity,
        stimulus_currents=stimulus.stimulus_currents
    )
    simulation_outcome = IzhikevichNetworkSimulator(
        current_components=neural_model,
        pb_off=False
    ).simulate(
        simulation_time=8,
        dt=1
    )