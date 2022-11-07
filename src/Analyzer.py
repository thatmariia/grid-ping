from src.izhikevich_simulation.IzhikevichNetworkOutcome import IzhikevichNetworkOutcome
from src.after_simulation_analysis.AfterSimulationAnalysisDataFactory import AfterSimulationAnalysisDataFactory
from src.after_simulation_analysis.AfterSimulationAnalysisData import AfterSimulationAnalysisData
from src.params.ParamsSync import *

from src.plotter.after_simulation_plots import *



class Analyzer:

    def __init__(self, simulation_outcome: IzhikevichNetworkOutcome, step: int, params_sync: ParamsSync):
        self.step = step
        self.analysis_data = AfterSimulationAnalysisDataFactory().create(
            simulation_outcome=simulation_outcome,
            params_sync=params_sync,
            step=step
        )

    def make_plots(self):

        plot_raster(all_spikes_ex=self.analysis_data.all_spikes_ex, all_spikes_in=self.analysis_data.all_spikes_in)
        # plot_number_neurons_spiked(windows=self.analysis_data.windows, spikes_df=self.analysis_data.spikes_df)
        # plot_stats_per_ping(
        #     ping_networks=self.analysis_data.ping_networks, spikes_df=self.analysis_data.spikes_df,
        #     windows=self.analysis_data.windows, step=self.step
        # )
        # plot_ping_frequencies(frequencies=self.analysis_data.spiking_freq.ping_frequencies)
        plot_phase_locking(phase_locking=self.analysis_data.sync_evaluation.phase_locking)




