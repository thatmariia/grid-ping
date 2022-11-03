from src.izhikevich_simulation.PINGNetworkNeurons import PINGNetworkNeurons
from src.after_simulation_analysis.SpikingFrequency import SpikingFrequency
from src.after_simulation_analysis.SyncEvaluation import SyncEvaluation

import numpy as np
import pandas as pd


class AfterSimulationAnalysisData:

    def __init__(
            self,
            step: int,
            windows: list[tuple[int, int]],
            all_spikes: np.ndarray,
            all_spikes_ex: np.ndarray,
            all_spikes_in: np.ndarray,
            # ping_networks: list[PINGNetworkNeurons],
            # ping_spikes: dict[tuple[int, int], np.ndarray],
            # ping_spikes_ex: dict[tuple[int, int], np.ndarray],
            # ping_spikes_in: dict[tuple[int, int], np.ndarray],
            # spikes_df: pd.DataFrame,
            spiking_freq: SpikingFrequency,
            sync_evaluation: SyncEvaluation
    ):
        self.step = step
        self.windows = windows
        self.all_spikes = all_spikes
        self.all_spikes_ex = all_spikes_ex
        self.all_spikes_in = all_spikes_in
        # self.ping_networks = ping_networks
        # self.ping_spikes = ping_spikes
        # self.ping_spikes_ex = ping_spikes_ex
        # self.ping_spikes_in = ping_spikes_in
        # self.spikes_df = spikes_df
        #
        self.spiking_freq = spiking_freq
        self.sync_evaluation = sync_evaluation
