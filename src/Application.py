from src.params.ParamsInitializer import *
from src.izhikevich_simulation.ConnectivityGridPINGFactory import *
from src.izhikevich_simulation.CurrentComponentsGridPING import *
from src.spiking_frequencies.SpikingFrequencyFactory import *
from src.spiking_frequencies.CrossCorrelationFactory import CrossCorrelationFactory
from src.debug_funcs import *

from src.plotter.directory_management import *
from src.plotter.ping_frequencies import plot_ping_frequencies, plot_frequencies_std

from itertools import product
import pandas as pd

DEBUGMODE = False


class Application:

    def run(self):
        clear_simulation_directory()

        # [1.0, 1.125, 1.25, 1.375, 1.5]
        dist_scales = [1.0]

        # [0.01, 0.0257, 0.505, 0.7525, 1]
        contrast_ranges = [0.01]

        frequencies_std = pd.DataFrame(index=dist_scales, columns=contrast_ranges, dtype=float)

        for dist_scale, contrast_range in product(dist_scales, contrast_ranges):

            print("********************************************************")
            print(f"Starting simulation for dist_scale={dist_scale}, contrast_range={contrast_range}")
            print("********************************************************")

            params_initializer = ParamsInitializer()
            params_ping, params_gabor, params_rf, params_connectivity, params_izhi, params_synaptic, params_freqs = params_initializer.initialize(
                dist_scale=dist_scale,
                contrast_range=contrast_range
            )

            cd_or_create_partic_plotting_directory(params_gabor.dist_scale, params_gabor.contrast_range)

            if DEBUGMODE:
                stimulus_currents, cortical_distances = try_pulling_stimulus_data(params_gabor, params_rf, params_ping,
                                                                                  params_izhi, params_freqs)
            else:
                stimulus = StimulusFactory().create(params_gabor, params_rf, params_ping, params_izhi, params_freqs)

                stimulus_currents = stimulus.stimulus_currents
                cortical_distances = stimulus.extract_stimulus_location().cortical_distances

            connectivity = ConnectivityGridPINGFactory().create(
                params_ping=params_ping,
                params_connectivity=params_connectivity,
                cortical_distances=cortical_distances
            )
            neural_model = CurrentComponentsGridPING(
                connectivity=connectivity,
                params_synaptic=params_synaptic,
                stimulus_currents=stimulus_currents
            )
            simulation_time = 100000
            simulation_outcome = IzhikevichNetworkSimulator(
                params_izhi=params_izhi,
                current_components=neural_model,
                pb_off=False
            ).simulate(
                simulation_time=simulation_time,
                dt=0.001,
                params_freqs=params_freqs
            )

            cc = CrossCorrelationFactory().create(simulation_outcome, params_ping, simulation_time)

            spiking_frequencies = SpikingFrequencyFactory().create(
                simulation_outcome=simulation_outcome,
                params_freqs=params_freqs
            )
            plot_ping_frequencies(spiking_frequencies.ping_frequencies)

            frequencies_std.loc[dist_scale, contrast_range] = spiking_frequencies.std

            return_to_start_path_from_partic()

        cd_or_create_general_plotting_directory()

        plot_frequencies_std(frequencies_std)
        # print(frequencies_std)

        return_to_start_path_from_general()

