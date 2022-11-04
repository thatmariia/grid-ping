from src.izhikevich_simulation.IzhikevichNetworkOutcome import IzhikevichNetworkOutcome
from src.params.ParamsPING import ParamsPING
from src.params.NeuronTypes import NeuronTypes
from src.after_simulation_analysis.SyncEvaluation import SyncEvaluation

import numpy as np
from math import sqrt, pi
from scipy.signal import correlate
import sys
from tqdm import tqdm
from scipy.ndimage import gaussian_filter


class SyncEvaluationFactory:

    def create(
            self, spikes, params_ping: ParamsPING, simulation_time: int
    ) -> SyncEvaluation:
        if spikes == []:
            print("No spikes")
            # return empty SyncEvaluation
            return SyncEvaluation(
                phase_values=[0],
                phase_locking=[0]
            )
        spikes_T = np.array(spikes).T

        # indices when neurons fired
        spikes_in_indices = np.argwhere(
            (spikes_T[1] >= params_ping.neur_slice[NeuronTypes.IN].start) &
            (spikes_T[1] < params_ping.neur_slice[NeuronTypes.IN].stop)
        ).flatten()
        # times when neurons fired
        spikes_in = np.array(list(zip(spikes_T[0][spikes_in_indices], spikes_T[1][spikes_in_indices])))

        # raster_ex = self._get_type_raster(
        #     spikes_type=spikes_ex,
        #     params_ping=params_ping,
        #     neur_type=NeuronTypes.EX,
        #     simulation_time=simulation_time
        # )
        raster_in = self._get_type_raster(
            spikes_type=spikes_in,
            params_ping=params_ping,
            neur_type=NeuronTypes.IN,
            simulation_time=simulation_time
        )

        phase_values, phase_locking = self._compute_cross_correlation(raster_in, simulation_time)

        sync_evaluation = SyncEvaluation(
            phase_values=phase_values,
            phase_locking=phase_locking,
        )
        return sync_evaluation

    def _compute_cross_correlation(self, raster, simulation_time):
        max_lag = self.get_max_lag(raster)# 12
        print("max_lag =", max_lag)
        step_size = 5

        phase_locking = np.zeros((raster.shape[0] // step_size, raster.shape[0] // step_size))
        alltim = np.zeros((raster.shape[0] // step_size, raster.shape[0] // step_size))
        phase_values = np.zeros((raster.shape[0] // step_size, raster.shape[0] // step_size))

        raster = self._apply_filter(raster)

        nn1 = 0
        time_window = list(range(199, simulation_time - 50))
        for id1 in (pbar := tqdm(range(0, raster.shape[0], step_size))):
            pbar.set_description("Computing cross-correlation & stuff")

            nn2 = 0
            for id2 in range(0, raster.shape[0], step_size):
                if id1 != id2:

                    sig1 = raster[id1, time_window].T
                    sig2 = raster[id2, time_window].T

                    correlation = correlate(sig1, sig2)

                    # cropping to account for maximum lag
                    if correlation.shape[0] > (2 * max_lag + 1):
                        mid = correlation.shape[0] // 2 + 1
                        correlation = correlation[mid - max_lag:mid + max_lag + 1]

                    # normalizing correlation
                    correlation = correlation / sqrt(
                        self._correlate_with_zero_lag(sig1, sig1) *
                        self._correlate_with_zero_lag(sig2, sig2)
                    )

                    peak_lag = np.argmax(correlation)
                    peak_height = correlation[peak_lag]

                else:
                    # autocorrelation
                    peak_lag = max_lag + 1
                    peak_height = 1

                phase_locking[nn1, nn2] = peak_height
                alltim[nn1, nn2] = peak_lag

                mean_spike_rate_1 = self.get_mean_spike_rate(raster[id1])
                mean_spike_rate_2 = self.get_mean_spike_rate(raster[id2])
                spike_timing_diff = abs(max_lag - abs(peak_lag))
                phase_values[nn1, nn2] = pi * (2 * spike_timing_diff) / (0.5 * (mean_spike_rate_1 + mean_spike_rate_2))

                nn2 += 1
            nn1 += 1

        return phase_values, phase_locking

    def _correlate_with_zero_lag(self, sig1, sig2):

        correlation = correlate(sig1, sig2)
        mid = correlation.shape[0] // 2 #+ 1

        if correlation[mid] == 0:
            return 1
        return correlation[mid]

    def get_mean_spike_rate(self, id_raster):
        if sum(id_raster) == 0:
            print("BOOM")
            #return sys.maxsize
            return 20
        return len(id_raster) / sum(id_raster)

    def get_max_lag(self, raster):

        return round(0.5 * np.mean([self.get_mean_spike_rate(raster[i]) for i in range(raster.shape[0])]))


        # inter_spike_times = []
        #
        # for t in id_raster:
        #     if t == 0:
        #         if len(inter_spike_times) == 0:
        #             continue
        #         inter_spike_times[-1] += 1
        #     else:
        #         inter_spike_times.append(1)
        #
        # if len(inter_spike_times) > 1:
        #     return np.mean(inter_spike_times[:-1])
        #
        # # TODO:: what do we return if the neuron didn't spike or only spiked once?
        # return 0


    def _get_type_raster(self, spikes_type, params_ping, neur_type, simulation_time):
        """
        TODO
        TODO:: assert see that there are spikes at all
        """

        raster = np.zeros((params_ping.nr_neurons[neur_type], simulation_time))

        for neur_id in range(params_ping.neur_slice[neur_type].start, params_ping.neur_slice[neur_type].stop):
            # find indices of spikes of a neuron with id neur_id
            spikes_indices = np.argwhere(
                (spikes_type.T[1] == neur_id)
            ).flatten()
            # assign 1 to the indices of the raster
            raster[neur_id - params_ping.neur_slice[neur_type].start, spikes_type.T[0].astype(int)[spikes_indices]] = 1

        return raster

    def _apply_filter(self, raster):
        # apply gaussian filter to raster
        return gaussian_filter(raster, sigma=6)


