from src.izhikevich_simulation.IzhikevichNetworkOutcome import *
from src.params.ParamsFrequencies import *
from src.spiking_frequencies.SpikingFrequency import SpikingFrequency

import numpy as np
from scipy import fft
from tqdm import tqdm

import warnings


class SpikingFrequencyFactory:
    """
    TODO:: docs
    TODO:: assrt that outcome isnt empty
    """

    def create(
            self, simulation_outcome: IzhikevichNetworkOutcome, params_freqs: ParamsFrequencies
    ) -> SpikingFrequency:

        frequencies = []

        for ping_network in (pbar := tqdm(simulation_outcome.grid_geometry.ping_networks, disable=True)):
            pbar.set_description("Frequency distribution per PING")

            # select ex neurons for a single ping network from spikes
            spikes_in_ping_mask = np.isin(
                np.array(simulation_outcome.spikes).T[1], ping_network.ids[NeuronTypes.EX]
            )

            # times when excitatory neurons fired
            spikes_times_in_ping = np.array(simulation_outcome.spikes)[spikes_in_ping_mask].T[0]
            spikes_ex_per_times = [
                np.count_nonzero(spikes_times_in_ping == t) for t in range(simulation_outcome.simulation_time)
            ]
            signal = np.array(spikes_ex_per_times[299:])

            frequency = self.tfr_single_ping(
                signal=signal,
                simulation_time=simulation_outcome.simulation_time,
                params_freqs=params_freqs
            )
            frequencies.append(frequency)

        spiking_frequency = SpikingFrequency(frequencies)

        return spiking_frequency

    def fft_single_ping(
            self, signal: np.ndarray[int, int], params_freqs: ParamsFrequencies
    ) -> int:
        """
        TODO
        :param signal:
        :param simulation_time:
        :param params_freqs:
        :return:
        """

        fft_data = fft.fft(signal)
        freqs = fft.fftfreq(len(signal), d=1 / 1000)

        gamma_indices = np.argwhere(
            (freqs >= params_freqs.frequencies[0]) &
            (freqs <= params_freqs.frequencies[-1])
        ).flatten()
        max_i = np.argmax(np.abs(fft_data[gamma_indices]))
        freq_max = freqs[gamma_indices][max_i]
        freq_max_abs = np.abs(freq_max)

        return np.abs(freq_max_abs)


    def tfr_single_ping(
            self, signal: np.ndarray[int, int], simulation_time: int, params_freqs: ParamsFrequencies
    ) -> int:
        """
        TODO:: Determines most prominent frequency??

        :param simulation_time: number of epochs to run the simulation.
        :type simulation_time: int

        :param signal: number of excitatory neurons fired at relevant epochs of the simulation.
        :type signal: list[int]

        :return: TODO:: most prominent frequency?
        :rtype: int
        """

        t = [i / 0.001 for i in range(1, simulation_time+1)]
        t = t[298:]
        # the size of the data + zero padding
        nr_points = len(params_freqs.wt) + len(signal) - 1
        fft_data = fft.fft(signal, nr_points)

        tfr = np.zeros((len(params_freqs.frequencies), len(t)), dtype="complex_") * np.nan

        for fi in range(len(params_freqs.frequencies)):

            fft_wavelet = fft.fft(params_freqs.complex_wavelets[fi], nr_points)
            fft_wavelet = fft_wavelet / max(fft_wavelet)

            tmp = fft.ifft(fft_wavelet * fft_data, nr_points)
            # trim the edges, these are the bits we included by zero padding
            tfr[
            np.argwhere(np.array(params_freqs.frequencies) == params_freqs.frequencies[fi]).flatten(), :
            ] = tmp[params_freqs.half_wave_size: -params_freqs.half_wave_size + 1]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mx_i = int(np.argmax(np.nanmean(np.abs(tfr), 1)))

        return params_freqs.frequencies[mx_i]
