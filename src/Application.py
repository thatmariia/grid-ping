from src.params.ParamsInitializer import ParamsInitializer
from src.izhikevich_simulation.IzhikevichNetworkOutcome import IzhikevichNetworkOutcome
from src.izhikevich_simulation.GridGeometryFactory import GridGeometryFactory

from src.plotter.directory_management import *
from src.plotter.raw_data import fetch_spikes_data

from src.Simulator import Simulator
from src.Analyzer import Analyzer
from src.overview.Results import Results

from itertools import product, repeat
import numpy as np
import os

from multiprocessing import Manager, Pool, Lock


CREATING_DATA = True

class Application:

    def __init__(self):
        # [1.0, 1.125, 1.25, 1.375, 1.5]
        self.dist_scales = [1.0, 1.125, 1.25, 1.375, 1.5]
        # [0.01, 0.2575, 0.505, 0.7525, 1.0]
        self.contrast_ranges = [0.01, 0.2575, 0.505, 0.7525, 1.0]

        self.simulation_time = 1000

        self.results = Results(self.dist_scales, self.contrast_ranges)

    def _single_simulation(self, sim_input, lock):
        dist_scale, contrast_range = sim_input
        print("**********************************************************")
        print(f"Starting simulation for dist_scale={dist_scale}, contrast_range={contrast_range}")
        print("**********************************************************")

        params_initializer = ParamsInitializer()
        params_ping, params_gabor, params_rf, params_connectivity, params_izhi, params_synaptic, params_freqs = \
            params_initializer.initialize(
                dist_scale=dist_scale,
                contrast_range=contrast_range
            )

        print(os.getcwd())
        cd_or_create_partic_plotting_directory(params_gabor.dist_scale, params_gabor.contrast_range)

        """DO SIMULATOR CRAP"""
        if CREATING_DATA:
            print("\n ~~~~ SIMULATION ~~~~ \n")
            simulation_outcome = Simulator().run_simulation(
                simulation_time=self.simulation_time,
                params_gabor=params_gabor,
                params_rf=params_rf,
                params_ping=params_ping,
                params_izhi=params_izhi,
                params_freqs=params_freqs,
                params_connectivity=params_connectivity,
                params_synaptic=params_synaptic
            )
        else:
            simulation_outcome = IzhikevichNetworkOutcome(
                spikes=fetch_spikes_data(dist_scale=dist_scale, contrast_range=contrast_range),
                params_ping=params_ping,
                params_freqs=params_freqs,
                simulation_time=self.simulation_time,
                grid_geometry=GridGeometryFactory().create(
                    params_ping,
                    np.zeros((params_ping.nr_neurons["total"], params_ping.nr_neurons["total"]))
                )
            )

        """DO ANALYSIS CRAP"""
        print("\n ~~~~ ANALYSIS ~~~~ \n")
        analyser = Analyzer(simulation_outcome=simulation_outcome, step=250)
        analyser.make_plots()

        with lock:
            self.results.add_results(
                dist_scale=dist_scale,
                contrast_range=contrast_range,
                avg_phase_locking=analyser.analysis_data.sync_evaluation.avg_phase_locking,
                frequency_std=analyser.analysis_data.spiking_freq.std
            )

        return_to_start_path_from_partic()

        print("\n")

    def run(self):
        if CREATING_DATA:
            clear_simulation_directory()

        simulation_time = 1000

        # results = Results(self.dist_scales, self.contrast_ranges)

        with Manager() as manager:
            lock = manager.Lock()
            with Pool(5) as pool:
                pool.starmap(self._single_simulation, list(zip(product(self.dist_scales, self.contrast_ranges), repeat(lock))))
        # for dist_scale, contrast_range in product(self.dist_scales, self.contrast_ranges):
        #    self._single_simulation(dist_scale=dist_scale, contrast_range=contrast_range)

        cd_or_create_general_plotting_directory()

        """DO OVERVIEW CRAP"""
        print("\n ~~~~ CREATING OVERVIEW ~~~~ \n")
        self.results.make_plots()

        return_to_start_path_from_general()
