from src.izhikevich_simulation.IzhikevichNetworkOutcome import IzhikevichNetworkOutcome
from src.params.ParamsPING import ParamsPING
from src.params.NeuronTypes import NeuronTypes

import numpy as np
from math import sqrt
from scipy.signal import correlate


class CrossCorrelationFactory:

    def create(
            self, simulation_outcome: IzhikevichNetworkOutcome, params_ping: ParamsPING, simulation_time: int
    ):
        spikes_T = np.array(simulation_outcome.spikes).T

        spikes_ex = self._get_type_spikes(
            spikes_T=spikes_T,
            id_start=params_ping.neur_slice[NeuronTypes.EX].start,
            id_stop=params_ping.neur_slice[NeuronTypes.EX].stop
        )
        spikes_in = self._get_type_spikes(
            spikes_T=spikes_T,
            id_start=params_ping.neur_slice[NeuronTypes.IN].start,
            id_stop=params_ping.neur_slice[NeuronTypes.IN].stop
        )

        raster_ex = self._get_type_raster(
            spikes_type=spikes_ex,
            params_ping=params_ping,
            neur_type=NeuronTypes.EX,
            simulation_time=simulation_time
        )
        raster_in = self._get_type_raster(
            spikes_type=spikes_in,
            params_ping=params_ping,
            neur_type=NeuronTypes.IN,
            simulation_time=simulation_time
        )

        self._compute_cross_correlation(raster_in, simulation_time)

    def _compute_cross_correlation(self, spike_dat, simulation_time):
        max_lag = 12
        step_size = 10

        allcoh = np.zeros((spike_dat.shape[0] // step_size, spike_dat.shape[0] // step_size))
        alltim = np.zeros((spike_dat.shape[0] // step_size, spike_dat.shape[0] // step_size))

        nn1 = 0
        time_window = list(range(199, simulation_time - 50))
        for id1 in range(0, spike_dat.shape[0], step_size):
            nn2 = 0
            for id2 in range(0, spike_dat.shape[0], step_size):
                if id1 != id2:

                    sig1 = spike_dat[id1, time_window].T
                    sig2 = spike_dat[id2, time_window].T

                    correlation = correlate(sig1, sig2)

                    # cropping to account for maximum lag
                    if correlation.shape[0] > (2 * max_lag + 1):
                        mid = correlation.shape[0] // 2 + 1
                        correlation = correlation[mid - max_lag:mid + max_lag + 1]

                    # normalizing correlation
                    correlation = correlation / sqrt(
                        self._correlate_with_zero_lag(sig1, sig1) +
                        self._correlate_with_zero_lag(sig2, sig2)
                    )

                    peak_lag = np.argmax(correlation)
                    peak_height = correlation[peak_lag]

                else:
                    # autocorrelation

                    peak_lag = max_lag + 1
                    peak_height = 1

                allcoh[nn1, nn2] = peak_height
                alltim[nn1, nn2] = peak_lag

                nn2 += 1
            nn1 += 1

    def _correlate_with_zero_lag(self, sig1, sig2):

        correlation = correlate(sig1, sig2)
        mid = correlation.shape[0] // 2 + 1
        # FIXME:: what to actually do when encounter 0?
        if correlation[mid] == 0:
            return 1
        return correlation[mid]

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
            raster[neur_id - params_ping.neur_slice[neur_type].start, spikes_type.T[0][spikes_indices]] = 1

        return raster

    def _get_type_spikes(
            self, spikes_T: np.ndarray, id_start: int, id_stop: int
    ):
        """
        TODO
        """

        # indices when neurons fired
        spikes_indices = np.argwhere(
            (spikes_T[1] >= id_start) &
            (spikes_T[1] < id_stop)
        ).flatten()
        # times when neurons fired
        spikes_type = np.array(list(zip(spikes_T[0][spikes_indices], spikes_T[1][spikes_indices])))

        return spikes_type
