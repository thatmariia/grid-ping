from src.izhikevich_simulation.IzhikevichNetworkOutcome import IzhikevichNetworkOutcome
from src.params.NeuronTypes import NeuronTypes
from src.after_simulation_analysis.AfterSimulationAnalysisData import AfterSimulationAnalysisData
from src.after_simulation_analysis.SpikingFrequencyFactory import SpikingFrequencyFactory
from src.after_simulation_analysis.SpikingFrequency import SpikingFrequency
from src.after_simulation_analysis.SyncEvaluationFactory import SyncEvaluationFactory
from src.params.ParamsSync import *

import numpy as np
import pandas as pd
from tqdm import tqdm


class AfterSimulationAnalysisDataFactory:

    def create(self, simulation_outcome: IzhikevichNetworkOutcome, params_sync: ParamsSync, step: int = 100) -> AfterSimulationAnalysisData:
        # TODO:: assert sim time divides step

        spikes_stats = []
        windows = [(i, i + step) for i in range(0, 1000, step)]

        def compute_nr_spikes(spikes):
            return len(spikes)

        def compute_mean_nr_spikes_per_ts(spikes, window):
            return np.mean([np.count_nonzero(spikes.T[0] == t) for t in range(window[0], window[1])])

        def compute_std_nr_spikes_per_ts(spikes, window):
            return np.std([np.count_nonzero(spikes.T[0] == t) for t in range(window[0], window[1])])

        def compute_nr_neurons_spiked_count(spikes, step):
            id_counter = np.array([
                np.count_nonzero(spikes.T[1] == i) for i in range(simulation_outcome.params_ping.nr_neurons["total"])
            ])
            return np.array([np.count_nonzero(id_counter == i) for i in range(step)])

        def compute_stats(spikes, spikes_ex, spikes_in, window):
            nr_spikes = compute_nr_spikes(spikes)
            nr_spikes_ex = compute_nr_spikes(spikes_ex)
            nr_spikes_in = compute_nr_spikes(spikes_in)
            mean_nr_spikes_per_ts = compute_mean_nr_spikes_per_ts(spikes, window)
            mean_nr_spikes_ex_per_ts = compute_mean_nr_spikes_per_ts(spikes_ex, window)
            mean_nr_spikes_in_per_ts = compute_mean_nr_spikes_per_ts(spikes_in, window)
            std_nr_spikes_per_ts = compute_std_nr_spikes_per_ts(spikes, window)
            std_nr_spikes_ex_per_ts = compute_std_nr_spikes_per_ts(spikes_ex, window)
            std_nr_spikes_in_per_ts = compute_std_nr_spikes_per_ts(spikes_in, window)
            nr_neurons_spiked_count = compute_nr_neurons_spiked_count(spikes, step)
            nr_neurons_spiked_count_ex = compute_nr_neurons_spiked_count(spikes_ex, step)
            nr_neurons_spiked_count_in = compute_nr_neurons_spiked_count(spikes_in, step)
            return [
                nr_spikes, nr_spikes_ex, nr_spikes_in,
                mean_nr_spikes_per_ts, mean_nr_spikes_ex_per_ts, mean_nr_spikes_in_per_ts,
                std_nr_spikes_per_ts, std_nr_spikes_ex_per_ts, std_nr_spikes_in_per_ts,
                nr_neurons_spiked_count, nr_neurons_spiked_count_ex, nr_neurons_spiked_count_in
            ]

        def get_spikes(spikes, ind_start, ind_end):
            indices = np.argwhere(
                (spikes.T[1] >= ind_start) &
                (spikes.T[1] < ind_end)
            ).flatten()
            return spikes[indices]

        def get_ex_spikes(spikes):
            return get_spikes(spikes, simulation_outcome.params_ping.neur_slice[NeuronTypes.EX].start,
                              simulation_outcome.params_ping.neur_slice[NeuronTypes.EX].stop)

        def get_in_spikes(spikes):
            return get_spikes(spikes, simulation_outcome.params_ping.neur_slice[NeuronTypes.IN].start,
                              simulation_outcome.params_ping.neur_slice[NeuronTypes.IN].stop)

        all_spikes = np.array(simulation_outcome.spikes)
        all_spikes_ex = get_ex_spikes(all_spikes)
        all_spikes_in = get_in_spikes(all_spikes)
        #
        # ping_networks = simulation_outcome.grid_geometry.ping_networks
        #
        # def neuron_ids_of_network(ping_network):
        #     return ping_network.ids[NeuronTypes.EX] + ping_network.ids[NeuronTypes.IN]
        #
        # ping_spikes = {}
        # ping_spikes_ex = {}
        # ping_spikes_in = {}
        #
        # for ping_network in ping_networks:
        #     ping_spikes_mask = np.isin(
        #         all_spikes.T[1], neuron_ids_of_network(ping_network)
        #     )
        #     ping_spikes[ping_network.grid_location] = all_spikes[ping_spikes_mask]
        #     ping_spikes_ex[ping_network.grid_location] = get_ex_spikes(ping_spikes[ping_network.grid_location])
        #     ping_spikes_in[ping_network.grid_location] = get_in_spikes(ping_spikes[ping_network.grid_location])
        #
        # cols = []
        # for i in range(-1, len(ping_networks)):
        #     imod = "" if (i == -1) else i
        #     new_cols = [
        #         f"nr_spikes{imod}",
        #         f"nr_spikes_ex{imod}",
        #         f"nr_spikes_in{imod}",
        #         f"mean_nr_spikes_per_ts{imod}",
        #         f"mean_nr_spikes_ex_per_ts{imod}",
        #         f"mean_nr_spikes_in_per_ts{imod}",
        #         f"std_nr_spikes_per_ts{imod}",
        #         f"std_nr_spikes_ex_per_ts{imod}",
        #         f"std_nr_spikes_in_per_ts{imod}",
        #         f"nr_neurons_spiked_count{imod}",
        #         f"nr_neurons_spiked_count_ex{imod}",
        #         f"nr_neurons_spiked_count_in{imod}"
        #     ]
        #     cols = cols + new_cols
        #
        # def apply_window(arr, window_start, window_end):
        #     arr_window_indices = np.argwhere(
        #         (arr.T[0] >= window_start) &
        #         (arr.T[0] < window_end)
        #     ).flatten()
        #     return arr[arr_window_indices]
        #
        # def select_spikes_window(spikes, spikes_ex, spikes_in, window_start, window_end):
        #     spikes = apply_window(spikes, window_start, window_end)
        #     spikes_ex = apply_window(spikes_ex, window_start, window_end)
        #     spikes_in = apply_window(spikes_in, window_start, window_end)
        #     return spikes, spikes_ex, spikes_in
        #
        # for window in (pbar := tqdm(windows)):
        #     pbar.set_description("Getting data overview")
        #
        #     spikes, spikes_ex, spikes_in = select_spikes_window(
        #         all_spikes, all_spikes_ex, all_spikes_in,
        #         window[0], window[1]
        #     )
        #     gen_stats = compute_stats(spikes, spikes_ex, spikes_in, window)
        #
        #     ping_stats = []
        #     for ping_network in ping_networks:
        #         spikes, spikes_ex, spikes_in = select_spikes_window(
        #             ping_spikes[ping_network.grid_location],
        #             ping_spikes_ex[ping_network.grid_location],
        #             ping_spikes_in[ping_network.grid_location],
        #             window[0], window[1]
        #         )
        #         ping_stats = ping_stats + compute_stats(spikes, spikes_ex, spikes_in, window)
        #
        #     spikes_stats.append(gen_stats + ping_stats)
        #
        # spikes_df = pd.DataFrame(
        #     spikes_stats,
        #     columns=cols,
        #     index=windows
        # )

        spiking_freq = SpikingFrequencyFactory().create(simulation_outcome)
        sync_evaluation = SyncEvaluationFactory().create(
            spikes=simulation_outcome.spikes,
            params_ping=simulation_outcome.params_ping,
            simulation_time=simulation_outcome.simulation_time,
            params_sync=params_sync
        )

        after_simulation_analysis_data = AfterSimulationAnalysisData(
            step=step,
            windows=windows,
            all_spikes=all_spikes,
            all_spikes_ex=all_spikes_ex,
            all_spikes_in=all_spikes_in,
            # ping_networks=ping_networks,
            # ping_spikes=ping_spikes,
            # ping_spikes_ex=ping_spikes_ex,
            # ping_spikes_in=ping_spikes_in,
            # spikes_df=spikes_df,
            spiking_freq=spiking_freq,
            sync_evaluation=sync_evaluation
        )
        return after_simulation_analysis_data

