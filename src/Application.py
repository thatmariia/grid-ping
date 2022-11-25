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

    def __init__(
            self,
            simulation_time,
            params_gabor,
            params_rf,
            params_ping,
            params_izhi,
            params_freqs,
            params_connectivity,
            params_synaptic,
            params_sync,
            params_c2c
    ):
        # # [1.0, 1.125, 1.25, 1.375, 1.5]
        # self.dist_scales = dist_scales
        # # [0.01, 0.2575, 0.505, 0.7525, 1.0]
        # self.contrast_ranges = contrast_ranges

        self.simulation_time = 1000

        self.simulation_time = simulation_time
        self.params_gabor = params_gabor
        self.params_rf = params_rf
        self.params_ping = params_ping
        self.params_izhi = params_izhi
        self.params_freqs = params_freqs
        self.params_connectivity = params_connectivity
        self.params_synaptic = params_synaptic
        self.params_sync = params_sync
        self.params_c2c = params_c2c


    def run_single_simulation(self, sim_input):
        dist_scale, contrast_range = sim_input
        print("**********************************************************")
        print(f"Starting simulation for dist_scale={dist_scale}, contrast_range={contrast_range}")
        print("**********************************************************")

        self.params_gabor.dist_scale = dist_scale
        self.params_gabor.contrast_range = contrast_range

        cd_or_create_partic_plotting_directory(self.params_gabor.dist_scale, self.params_gabor.contrast_range)

        """DO SIMULATOR CRAP"""
        print("\n ~~~~ SIMULATION ~~~~ \n")
        simulation_outcome = Simulator().run_simulation(
            simulation_time=self.simulation_time,
            params_gabor=self.params_gabor,
            params_rf=self.params_rf,
            params_ping=self.params_ping,
            params_izhi=self.params_izhi,
            params_freqs=self.params_freqs,
            params_connectivity=self.params_connectivity,
            params_synaptic=self.params_synaptic,
            params_c2c=self.params_c2c
        )

        """DO ANALYSIS CRAP"""
        print("\n ~~~~ ANALYSIS ~~~~ \n")
        analyser = Analyzer(simulation_outcome=simulation_outcome, step=250, params_sync=self.params_sync)
        analyser.make_plots()

        return_to_start_path_from_partic()

        avg_phase_locking = analyser.analysis_data.sync_evaluation.avg_phase_locking
        # write avg_phase_locking to file
        with open("avg_phase_locking.txt", "w") as f:
            f.write(str(avg_phase_locking))

        # read avg_phase_locking from file
        # with open("avg_phase_locking.txt", "r") as f:
        #     avg_phase_locking = float(f.read())
        #



    def _single_simulation_from_full(self, sim_input, results, lock=None):
        dist_scale, contrast_range = sim_input
        print("**********************************************************")
        print(f"Starting simulation for dist_scale={dist_scale}, contrast_range={contrast_range}")
        print("**********************************************************")

        self.params_gabor.dist_scale = dist_scale
        self.params_gabor.contrast_range = contrast_range

        print(os.getcwd())
        cd_or_create_partic_plotting_directory(self.params_gabor.dist_scale, self.params_gabor.contrast_range)

        """DO SIMULATOR CRAP"""
        if CREATING_DATA:
            print("\n ~~~~ SIMULATION ~~~~ \n")
            simulation_outcome = Simulator().run_simulation(
                simulation_time=self.simulation_time,
                params_gabor=self.params_gabor,
                params_rf=self.params_rf,
                params_ping=self.params_ping,
                params_izhi=self.params_izhi,
                params_freqs=self.params_freqs,
                params_connectivity=self.params_connectivity,
                params_synaptic=self.params_synaptic,
                params_c2c=self.params_c2c
            )
        else:
            simulation_outcome = IzhikevichNetworkOutcome(
                spikes=fetch_spikes_data(dist_scale=dist_scale, contrast_range=contrast_range),
                params_ping=self.params_ping,
                params_freqs=self.params_freqs,
                simulation_time=self.simulation_time,
                grid_geometry=GridGeometryFactory().create(
                    self.params_ping,
                    np.zeros((self.params_ping.nr_neurons["total"], self.params_ping.nr_neurons["total"]))
                )
            )

        """DO ANALYSIS CRAP"""
        print("\n ~~~~ ANALYSIS ~~~~ \n")
        analyser = Analyzer(simulation_outcome=simulation_outcome, params_sync=self.params_sync, step=250)
        analyser.make_plots()

        # with lock:
        results.add_results(
            dist_scale=dist_scale,
            contrast_range=contrast_range,
            avg_phase_locking=analyser.analysis_data.sync_evaluation.avg_phase_locking,
            frequency_std=analyser.analysis_data.spiking_freq.std
        )
        print(results.avg_phase_lockings_df)

        return_to_start_path_from_partic()

        print("\n")

        return results

    def run_full_simulation(self, dist_scales, contrast_ranges):
        if CREATING_DATA:
            clear_simulation_directory()

        results = Results(dist_scales, contrast_ranges)

        # with Manager() as manager:
        #     lock = manager.Lock()
        #     with Pool(1) as pool:
        #         pool.starmap(self._single_simulation, list(zip(product(self.dist_scales, self.contrast_ranges), repeat(lock))))
        for dist_scale, contrast_range in product(dist_scales, contrast_ranges):
           results = self._single_simulation_from_full((dist_scale, contrast_range), results)

        cd_or_create_general_plotting_directory()

        """DO OVERVIEW CRAP"""
        print("\n ~~~~ CREATING OVERVIEW ~~~~ \n")
        results.make_plots()

        return_to_start_path_from_general()
