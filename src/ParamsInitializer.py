from src.ParamsPING import *
from src.ParamsGaborStimulus import *
from src.ParamsReceptiveField import *
from src.ParamsConnectivity import *
from src.ParamsIzhikevich import *
from src.ParamsSynaptic import *

class ParamsInitializer:
    """
    This class initializes user-defined parameters.
    """

    def initialize(self):
        params_ping = ParamsPING(
            nr_excitatory=200,#400,
            nr_inhibitory=50,#100,
            nr_ping_networks=25#100
        )
        params_gabor = ParamsGaborStimulus(
            spatial_freq=5.7,
            vlum=0.5,
            diameter_dg=1,
            diameter=50,
            dist_scale=2.5,
            full_width_dg=35,
            full_height_dg=27.09,
            contrast_range=0.01,
            figure_width_dg=9,
            figure_height_dg=5.1,
            figure_ecc_dg=8,
            patch_size_dg=5
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
            spatial_consts_EE=0.4,
            spatial_consts_EI=0.3,
            spatial_consts_IE=0.3,
            spatial_consts_II=0.3
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

        return params_ping, params_gabor, params_rf, params_connectivity, params_izhi, params_synaptic