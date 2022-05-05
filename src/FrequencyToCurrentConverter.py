from src.ConnectivitySinglePINGFactory import *
from src.CurrentComponentsSinglePING import *
from src.IzhikevichNetworkSimulator import *
from src.IzhikevichNetworkOutcome import *
from src.NeuronTypes import *

import numpy as np
from math import floor, pi
from scipy import fft
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.linear_model import TheilSenRegressor
from sklearn.model_selection import RepeatedKFold, cross_val_score

from typing import Any
import warnings

class FrequencyToCurrentConverter:
    """
    This class converts the frequencies stimulus into the currents stimulus.
    """

    def convert(
            self, stimulus_frequencies: np.ndarray[int, float],
            nr_neurons: dict[Any, int], neur_slice: dict[NeuronTypes, slice]
    ) -> np.ndarray[int, float]:
        """
        Converts the frequencies stimulus into the currents stimulus.

        TODO:: how do I cite this?

        :param stimulus_frequencies: frequencies stimulus.
        :type stimulus_frequencies: numpy.ndarray[int, float]

        :param nr_neurons: dictionary of number of neurons of each type and the total number of neurons.
        :type nr_neurons: dict[Any, int]

        :param neur_slice: indices of each type of neurons.
        :type neur_slice: dict[NeuronTypes, slice]

        :return: the stimulus converted to currents.
        :rtype: numpy.ndarray[int, float]
        """

        simulation_time = 1000
        inputs = list(range(20, 51))
        g_power = np.zeros(len(inputs)) * np.nan

        for i in (pbar := tqdm(range(len(inputs)))):
            pbar.set_description("Stimulus conversion to current")
            simulation_outcome = self._simulate(simulation_time, nr_neurons, inputs[i])
            spikes_T = np.array(simulation_outcome.spikes).T

            # indices when excitatory neurons fired
            spikes_ex_indices = np.argwhere(
                (spikes_T[1] >= neur_slice[NeuronTypes.E].start) & (spikes_T[1] < neur_slice[NeuronTypes.E].stop)
            ).flatten()
            # times when excitatory neurons fired
            spikes_ex_times = spikes_T[0][spikes_ex_indices]
            # number of excitatory neurons fired at each time
            spikes_ex_per_times = [np.count_nonzero(spikes_ex_times == t) for t in range(simulation_time)]
            signal = spikes_ex_per_times[299:]

            # making TFR
            g_power[i] = self._make_tfr(simulation_time, signal)

        # fitting a line
        inputs = np.array(inputs)
        fitted_model = self._fit_line_robust(x=inputs, y=g_power.reshape((-1, 1)))

        # plot
        freqs_line = np.arange(g_power.min(), inputs.max(), 0.01)
        currents_line = fitted_model.predict(freqs_line.reshape((len(freqs_line), 1)))
        self._plot_relationship(g_power, inputs, freqs_line, currents_line)

        # finally, convert frequencies
        stimulus_currents = fitted_model.predict(stimulus_frequencies.reshape(len(stimulus_frequencies), 1))

        return stimulus_currents

    def _fit_line_robust(self, x: np.ndarray[int, float], y: np.ndarray[int, float]) -> TheilSenRegressor:
        """
        Creates a model that fits a line to the data.

        :param x: values to predict (target).
        :type x: numpy.ndarray[int, float]

        :param y: values used for prediction.
        :type y: numpy.ndarray[int, float]

        :return: a fitted regression model.
        :rtype: TheilSenRegressor
        """

        # define the model
        model = TheilSenRegressor()
        # define model evaluation method
        cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
        # evaluate model
        scores = cross_val_score(model, y, x, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
        # force scores to be positive
        scores = np.absolute(scores)
        # fut the model on all data
        model.fit(y, x)

        return model

    def _simulate(self, simulation_time: int, nr_neurons: dict[Any, int], mean_ex: float) -> IzhikevichNetworkOutcome:
        """
        Simulates an Izhikevich network with a single PING.

        :param simulation_time: number of epochs to run the simulation.
        :type simulation_time: int

        :param nr_neurons: dictionary of number of neurons of each type and the total number of neurons.
        :type nr_neurons: dict[Any, int]

        :param mean_ex: mean of input strength to excitatory neurons.
        :type mean_ex: float

        :return: collected information from the simulation.
        :rtype: IzhikevichNetworkOutcome
        """

        max_connect_strength = {
            (NeuronTypes.E, NeuronTypes.E): 0.05,
            (NeuronTypes.E, NeuronTypes.I): 0.4,
            (NeuronTypes.I, NeuronTypes.E): 0.3,
            (NeuronTypes.I, NeuronTypes.I): 0.2
        }
        connectivity = ConnectivitySinglePINGFactory().create(
            nr_excitatory=nr_neurons[NeuronTypes.E],
            nr_inhibitory=nr_neurons[NeuronTypes.I],
            max_connect_strength=max_connect_strength
        )
        synaptic_rise = {
            NeuronTypes.E: 0.15,
            NeuronTypes.I: 0.2
        }
        synaptic_decay = {
            NeuronTypes.E: 1,
            NeuronTypes.I: 7
        }
        current_components = CurrentComponentsSinglePING(
            connectivity=connectivity,
            synaptic_rise=synaptic_rise,
            synaptic_decay=synaptic_decay,
            mean_ex=mean_ex,
            var_ex=0,
            mean_in=4,
            var_in=0
        )
        simulation_outcome = IzhikevichNetworkSimulator(current_components).simulate(
            simulation_time=simulation_time,
            dt=1
        )

        return simulation_outcome

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

        freq_of_interest = list(range(20, 81))
        tfr = np.zeros((len(freq_of_interest), len(t)), dtype="complex_") * np.nan

        # set the width of the Gaussian
        sd = .05
        for freq in freq_of_interest:
            g = np.exp(-np.power(wt, 2) / (2 * sd ** 2))
            complex_sine = np.exp(1j * 2 * pi * freq * wt)
            complex_wavelet = complex_sine * g
            fft_wavelet = fft.fft(complex_wavelet, nr_points)
            fft_wavelet = fft_wavelet / max(fft_wavelet)

            tmp = fft.ifft(fft_wavelet * fft_data, nr_points)
            # trim the edges, these are the bits we included by zero padding
            tfr[np.argwhere(np.array(freq_of_interest) == freq).flatten(), :] = tmp[half_wave_size: -half_wave_size + 1]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mx_i = int(np.argmax(np.nanmean(np.abs(tfr), 1)))

        return mx_i

    def _plot_relationship(
            self, freqs: np.ndarray[int, float], currents: np.ndarray[int, float],
            freqs_line: np.ndarray[int, float], currents_line: np.ndarray[int, float]
    ) -> None:
        """
        Plots the relationship between frequency and current.

        :param freqs: frequencies from simulated data.
        :type freqs: numpy.ndarray[int, float]

        :param currents: currents from simulated data.
        :type currents: numpy.ndarray[int, float]

        :param freqs_line: frequencies from fitted line.
        :type freqs_line: numpy.ndarray[int, float]

        :param currents_line: currents from fitted line.
        :type currents_line: numpy.ndarray[int, float]

        :rtype: None
        """

        # TODO:: make pretty

        fig, ax = plt.subplots(figsize=(30, 30))
        ax.tick_params(axis='both', which='major', labelsize=20)

        # simulation data
        plt.scatter(currents, freqs, linewidths=20)
        # fitted line
        plt.plot(currents_line, freqs_line, color='r')

        fig.savefig("../plots/test-freq-current-relationship.png", bbox_inches='tight', pad_inches=0)