from src.Application import Application

from src.params.ParamsPING import *
from src.params.ParamsGaborStimulus import *
from src.params.ParamsReceptiveField import *
from src.params.ParamsConnectivity import *
from src.params.ParamsIzhikevich import *
from src.params.ParamsSynaptic import *
from src.params.ParamsFrequencies import *
from src.params.ParamsSync import *
from src.params.ParamsContrastToCurrent import *

import os
import argparse
from math import exp

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='grid-ping')
    parser.add_argument('--root', action=argparse.BooleanOptionalAction)

    parser.add_argument('--single_simulation', action=argparse.BooleanOptionalAction)
    parser.add_argument('--single_dist_scale', type=float, default=1.0)
    parser.add_argument('--single_contrast_range', type=float, default=0.01)

    parser.add_argument('--simulation_time', type=int, default=1000)
    parser.add_argument('--nr_excitatory', type=int, default=100 * 200)
    parser.add_argument('--nr_inhibitory', type=int, default=100 * 50)
    parser.add_argument('--nr_ping_networks', type=int, default=100)
    parser.add_argument('--spatial_freq', type=float, default=5.7)
    parser.add_argument('--vlum', type=float, default=0.5)
    parser.add_argument('--diameter_dg', type=float, default=0.7)
    parser.add_argument('--diameter', type=int, default=50)
    parser.add_argument('--dist_scale', type=float, default=1.5)
    parser.add_argument('--full_width_dg', type=float, default=33.87)
    parser.add_argument('--full_height_dg', type=float, default=27.09)
    parser.add_argument('--contrast_range', type=float, default=0.01)
    parser.add_argument('--figure_width_dg', type=float, default=5)
    parser.add_argument('--figure_height_dg', type=float, default=9)
    parser.add_argument('--figure_ecc_dg', type=float, default=7)
    parser.add_argument('--patch_size_dg', type=float, default=2.23)
    parser.add_argument('--slope', type=float, default=0.172)
    parser.add_argument('--intercept', type=float, default=-0.25)
    parser.add_argument('--min_diam_rf', type=float, default=1)
    parser.add_argument('--max_connect_strength_EE', type=float, default=0.04)
    parser.add_argument('--max_connect_strength_EI', type=float, default=0.07)
    parser.add_argument('--max_connect_strength_IE', type=float, default=-0.04)
    parser.add_argument('--max_connect_strength_II', type=float, default=-0.015)
    parser.add_argument('--spatial_const_EE', type=float, default=0.4)
    parser.add_argument('--spatial_const_EI', type=float, default=0.3)
    parser.add_argument('--spatial_const_IE', type=float, default=0.3)
    parser.add_argument('--spatial_const_II', type=float, default=0.3)
    parser.add_argument('--peak_potential', type=float, default=30)
    parser.add_argument('--alpha_E', type=float, default=0.02)
    parser.add_argument('--beta_E', type=float, default=0.2)
    parser.add_argument('--gamma_E', type=float, default=-65)
    parser.add_argument('--zeta_E', type=float, default=8)
    parser.add_argument('--alpha_I', type=float, default=0.1)
    parser.add_argument('--beta_I', type=float, default=0.2)
    parser.add_argument('--gamma_I', type=float, default=-65)
    parser.add_argument('--zeta_I', type=float, default=2)
    parser.add_argument('--rise_E', type=float, default=1)
    parser.add_argument('--decay_E', type=float, default=2.4)
    parser.add_argument('--rise_I', type=float, default=2)
    parser.add_argument('--decay_I', type=float, default=20)
    parser.add_argument('--conductance_EE', type=float, default=0.6)
    parser.add_argument('--conductance_EI', type=float, default=0.06)
    parser.add_argument('--conductance_IE', type=float, default=0.8)
    parser.add_argument('--conductance_II', type=float, default=0.5)
    parser.add_argument('--reversal_potential_E', type=float, default=-80)
    parser.add_argument('--reversal_potential_I', type=float, default=0)
    parser.add_argument('--frequency_low', type=int, default=20)
    parser.add_argument('--frequency_high', type=int, default=81)
    parser.add_argument('--gaussian_width', type=float, default=0.5)
    parser.add_argument('--dist_scales', type=float, default=[1.0, 1.125, 1.25, 1.375, 1.5], nargs='+')
    parser.add_argument('--contrast_ranges', type=float, default=[0.01, 0.2575, 0.505, 0.7525, 1.0], nargs='+')
    parser.add_argument('--sync_step_size', type=int, default=10)
    parser.add_argument('--sync_sigma', type=float, default=1)
    parser.add_argument('--sync_nr_cores', type=int, default=1)
    parser.add_argument('--c2c_min_current', type=float, default=exp(0.5))
    parser.add_argument('--c2c_max_current', type=float, default=exp(2.3))

    args = parser.parse_args()

    if args.root is None:
        os.chdir("../")

    simulation_time = args.simulation_time

    params_gabor = ParamsGaborStimulus(
        spatial_freq=args.spatial_freq,
        vlum=args.vlum,
        diameter_dg=args.diameter_dg,
        diameter=args.diameter,
        dist_scale=args.single_dist_scale,
        full_width_dg=args.full_width_dg,
        full_height_dg=args.full_height_dg,
        contrast_range=args.single_contrast_range,
        figure_width_dg=args.figure_width_dg,
        figure_height_dg=args.figure_height_dg,
        figure_ecc_dg=args.figure_ecc_dg,
        patch_size_dg=args.patch_size_dg
        # 4.914 for 81 #4.2 for 100 #4.89 for 25 #4.928 for 16 #4.95 for 4 #4.2 for 400
    )

    params_ping = ParamsPING(
        nr_excitatory=args.nr_excitatory,
        nr_inhibitory=args.nr_inhibitory,
        nr_ping_networks=args.nr_ping_networks
    )

    params_rf = ParamsReceptiveField(
        slope=args.slope,
        intercept=args.intercept,
        min_diam_rf=args.min_diam_rf
    )
    params_connectivity = ParamsConnectivity(
        max_connect_strength_EE=args.max_connect_strength_EE,
        max_connect_strength_EI=args.max_connect_strength_EI,
        max_connect_strength_IE=args.max_connect_strength_IE,
        max_connect_strength_II=args.max_connect_strength_II,
        spatial_const_EE=args.spatial_const_EE,
        spatial_const_EI=args.spatial_const_EI,
        spatial_const_IE=args.spatial_const_IE,
        spatial_const_II=args.spatial_const_II
    )
    params_izhi = ParamsIzhikevich(
        peak_potential=args.peak_potential,
        alpha_E=args.alpha_E,
        beta_E=args.beta_E,
        gamma_E=args.gamma_E,
        zeta_E=args.zeta_E,
        alpha_I=args.alpha_I,
        beta_I=args.beta_I,
        gamma_I=args.gamma_I,
        zeta_I=args.zeta_I
    )
    params_synaptic = ParamsSynaptic(
        rise_E=args.rise_E,
        decay_E=args.decay_E,
        rise_I=args.rise_I,
        decay_I=args.decay_I,
        conductance_EE=args.conductance_EE,
        conductance_EI=args.conductance_EI,
        conductance_IE=args.conductance_IE,
        conductance_II=args.conductance_II,
        reversal_potential_E=args.reversal_potential_E,
        reversal_potential_I=args.reversal_potential_I
    )
    params_freqs = ParamsFrequencies(
        frequency_range=range(args.frequency_low, args.frequency_high),
        gaussian_width=args.gaussian_width
    )

    params_sync = ParamsSync(
        step_size=args.sync_step_size,
        sigma=args.sync_sigma,
        nr_cores=args.sync_nr_cores
    )

    params_c2c = ParamsContrastToCurrent(
        min_current=args.c2c_min_current,
        max_current=args.c2c_max_current
    )

    application = Application(
        simulation_time=simulation_time,
        params_gabor=params_gabor,
        params_rf=params_rf,
        params_ping=params_ping,
        params_izhi=params_izhi,
        params_freqs=params_freqs,
        params_connectivity=params_connectivity,
        params_synaptic=params_synaptic,
        params_sync=params_sync,
        params_c2c=params_c2c
    )

    if args.single_simulation:
        dist_scale = args.single_dist_scale
        contrast_range = args.single_contrast_range

        application.run_single_simulation((dist_scale, contrast_range))
    else:
        dist_scales = args.dist_scales
        contrast_ranges = args.contrast_ranges

        application.run_full_simulation(dist_scales=dist_scales, contrast_ranges=contrast_ranges)


