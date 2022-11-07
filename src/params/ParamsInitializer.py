from src.params.ParamsPING import *
from src.params.ParamsGaborStimulus import *
from src.params.ParamsReceptiveField import *
from src.params.ParamsConnectivity import *
from src.params.ParamsIzhikevich import *
from src.params.ParamsSynaptic import *
from src.params.ParamsFrequencies import *
from src.params.ParamsSync import *

class ParamsInitializer:
    """
    This class initializes user-defined parameters.
    """

    def initialize(self, dist_scale=1.5, contrast_range=0.01):
        params_ping = ParamsPING(
            nr_excitatory=400 * 40,
            nr_inhibitory=400 * 10,
            nr_ping_networks=400
        )
        # params_ping = ParamsPING(
        #     nr_excitatory=100 * 250,
        #     nr_inhibitory=100 * 50,
        #     nr_ping_networks=100
        # )

        params_gabor = ParamsGaborStimulus(
            spatial_freq=5.7,
            vlum=0.5,
            diameter_dg=0.7,
            diameter=50,
            dist_scale=dist_scale,
            full_width_dg=33.87,
            full_height_dg=27.09,
            contrast_range=contrast_range,
            figure_width_dg=5,
            figure_height_dg=9,
            figure_ecc_dg=7,
            patch_size_dg=2.23 #4.914 for 81 #4.2 for 100 #4.89 for 25 #4.928 for 16 #4.95 for 4 #4.2 for 400
        )

        params_rf = ParamsReceptiveField(
            slope=0.172,
            intercept=-0.25,
            min_diam_rf=1
        )
        params_connectivity = ParamsConnectivity(
            max_connect_strength_EE=0.04,
            max_connect_strength_EI=0.07,
            max_connect_strength_IE=-0.04,
            max_connect_strength_II=-0.015,
            spatial_const_EE=0.4,#0.4,
            spatial_const_EI=0.3,#0.3,
            spatial_const_IE=0.3,#0.3,
            spatial_const_II=0.3#0.3
        )
        params_izhi = ParamsIzhikevich(
            peak_potential=30,
            alpha_E=0.02,
            beta_E=0.2,
            gamma_E=-65,
            zeta_E=8,
            alpha_I=0.1,
            beta_I=0.2,
            gamma_I=-65,
            zeta_I=2
        )
        params_synaptic = ParamsSynaptic(
            rise_E=1,
            decay_E=2.4,
            rise_I=2,
            decay_I=20,
            conductance_EE=0.6,
            conductance_EI=0.06,
            conductance_IE=0.8,
            conductance_II=0.5,
            reversal_potential_E=-80,
            reversal_potential_I=0
        )
        params_freqs = ParamsFrequencies(
            frequency_range=range(20, 81),
            gaussian_width=0.5
        )

        return params_ping, params_gabor, params_rf, params_connectivity, params_izhi, params_synaptic, params_freqs