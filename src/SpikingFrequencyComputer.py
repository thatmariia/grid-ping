from src.params.ParamsPING import *
from src.izhikevich_simulation.IzhikevichNetworkOutcome import *

import numpy as np
from math import floor, pi
from scipy import fft
import matplotlib.pyplot as plt
from collections import Counter

import warnings


class SpikingFrequencyComputer:
    """
    TODO:: docs
    """

    def compute_per_ping(self, simulation_outcome: IzhikevichNetworkOutcome):

        frequencies = []

        for ping_network in simulation_outcome.grid_geometry.ping_networks:
            # select ex neurons for a single ping network from spikes
            spikes_in_ping_mask = np.isin(
                np.array(simulation_outcome.spikes).T[1], ping_network.ids[NeuronTypes.EX]
            )

            # times when excitatory neurons fired
            spikes_times_in_ping = np.array(simulation_outcome.spikes)[spikes_in_ping_mask].T[0]

            frequency = self.compute_for_single_ping(
                spikes_times=spikes_times_in_ping,
                simulation_time=simulation_outcome.simulation_time
            )
            frequencies.append(frequency)

        return frequencies

    def plot_ping_frequencies(self, frequencies):
        # TODO:: make pretty

        print(frequencies)

        print("Plotting current-frequency.....", end="")
        path = "../plots/test-freq-in-pings.png"

        fig, ax = plt.subplots(figsize=(30, 30))
        ax.tick_params(axis='both', which='major', labelsize=50)

        plt.hist(frequencies, color="#ACDDE7")
        fig.savefig(path, bbox_inches='tight')

        print(end="\r", flush=True)
        print(f"Plotting ended, result: {path[3:]}")


    def compute_for_single_ping(self, spikes_times: np.ndarray[int, int], simulation_time: int) -> int:

        # number of excitatory neurons fired at each time
        spikes_ex_per_times = [np.count_nonzero(spikes_times == t) for t in range(simulation_time)]
        signal = spikes_ex_per_times[299:]
        # print("signal =", signal)

        # making TFR
        frequency = self._make_tfr(simulation_time, signal)
        return frequency

    def _make_tfr(self, simulation_time: int, signal: list[int]) -> int:
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
        wt = np.linspace(-1, 1, floor(2 / 0.001) + 1)
        # half the size of the wavelet
        half_wave_size = floor((len(wt) - 1) / 2)
        # the size of the data + zero padding
        nr_points = len(wt) + len(signal) - 1
        fft_data = fft.fft(signal, nr_points)

        gamma_frequencies = list(range(20, 81))
        tfr = np.zeros((len(gamma_frequencies), len(t)), dtype="complex_") * np.nan

        # set the width of the Gaussian
        sd = 0.05
        for frequency in gamma_frequencies:
            g = np.exp(-np.power(wt, 2) / (2 * sd ** 2))
            complex_sine = np.exp(1j * 2 * pi * frequency * wt)
            complex_wavelet = complex_sine * g
            fft_wavelet = fft.fft(complex_wavelet, nr_points)
            fft_wavelet = fft_wavelet / max(fft_wavelet)

            tmp = fft.ifft(fft_wavelet * fft_data, nr_points)
            # trim the edges, these are the bits we included by zero padding
            tfr[np.argwhere(np.array(gamma_frequencies) == frequency).flatten(), :] = tmp[half_wave_size: -half_wave_size + 1]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mx_i = int(np.argmax(np.nanmean(np.abs(tfr), 1)))

        return gamma_frequencies[mx_i]
