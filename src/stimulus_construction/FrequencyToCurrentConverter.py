from src.izhikevich_simulation.ConnectivitySinglePINGFactory import *
from src.izhikevich_simulation.CurrentComponentsSinglePING import *
from src.izhikevich_simulation.IzhikevichNetworkSimulator import *
from src.spiking_frequencies.SpikingFrequencyFactory import *
from src.debug_funcs import USE_CALCULATED_FC_RELATIONSHIP

from src.plotter.stimulus import plot_frequency_vs_current

import numpy as np
from tqdm import tqdm

from sklearn.linear_model import TheilSenRegressor
from sklearn.model_selection import RepeatedKFold, cross_val_score


class FrequencyToCurrentConverter:
    """
    This class converts the frequencies stimulus into the currents stimulus.
    """

    def convert(
            self,
            stimulus_frequencies: np.ndarray[int, float],
            params_ping: ParamsPING, params_izhi: ParamsIzhikevich, params_freqs: ParamsFrequencies
    ) -> np.ndarray[int, float]:
        """
        Converts the frequencies stimulus into the currents stimulus.

        TODO:: how do I cite this?

        :param stimulus_frequencies: frequencies stimulus.
        :type stimulus_frequencies: numpy.ndarray[int, float]

        :param params_ping: parameters describing PING networks and their composition.
        :type params_ping: ParamsPING

        :param params_izhi: contains Izhikevich parameters.
        :type params_izhi: ParamsIzhikevich

        :return: the stimulus converted to currents.
        :rtype: numpy.ndarray[int, float]
        """

        if USE_CALCULATED_FC_RELATIONSHIP:
            slope = 0.712
            intercept = -4.15
            return slope * stimulus_frequencies + intercept

        simulation_time = 1000
        inputs = list(range(20, 51))
        g_power = np.zeros(len(inputs)) * np.nan
        frequency_computer = SpikingFrequencyFactory()

        for i in (pbar := tqdm(range(len(inputs)))):
            pbar.set_description("Stimulus conversion to current")
            simulation_outcome = self._simulate(simulation_time, params_ping, params_izhi, inputs[i])
            spikes_T = np.array(simulation_outcome.spikes).T

            # indices when excitatory neurons fired
            spikes_ex_indices = np.argwhere(
                (spikes_T[1] >= params_ping.neur_slice[NeuronTypes.EX].start) &
                (spikes_T[1] < params_ping.neur_slice[NeuronTypes.EX].stop)
            ).flatten()
            # times when excitatory neurons fired
            spikes_ex_times = spikes_T[0][spikes_ex_indices]
            spikes_ex_per_times = [np.count_nonzero(spikes_ex_times == t) for t in range(simulation_time)]
            signal = np.array(spikes_ex_per_times[299:])

            # g_power[i] = frequency_computer.fft_single_ping(
            #     signal=signal,
            #     params_freqs=params_freqs
            # )

            g_power[i] = frequency_computer.tfr_single_ping(
                signal, simulation_time, params_freqs
            )

        # fitting a line
        inputs = np.array(inputs)
        fitted_model = self._fit_line_robust(x=inputs, y=g_power.reshape((-1, 1)))

        # plot
        freqs_line = np.arange(g_power.min(), g_power.max(), 0.01)
        currents_line = fitted_model.predict(freqs_line.reshape((len(freqs_line), 1)))
        plot_frequency_vs_current(g_power, inputs, freqs_line, currents_line)

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
        :rtype: sklearn.linear_model.TheilSenRegressor
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

    def _simulate(
            self, simulation_time: int, params_ping: ParamsPING, params_izhi: ParamsIzhikevich, mean_ex: float
    ) -> IzhikevichNetworkOutcome:
        """
        Simulates an Izhikevich network with a single PING.

        :param simulation_time: number of epochs to run the simulation.
        :type simulation_time: int

        :param params_ping: parameters describing PING networks and their composition.
        :type params_ping: ParamsPING

        :param params_izhi: contains Izhikevich parameters.
        :type params_izhi: ParamsIzhikevich

        :param mean_ex: mean of input strength to excitatory neurons.
        :type mean_ex: float

        :return: collected information from the simulation.
        :rtype: IzhikevichNetworkOutcome
        """

        params_connectivity = ParamsConnectivity(
            max_connect_strength_EE=0.05,
            max_connect_strength_EI=0.4,
            max_connect_strength_IE=0.3,
            max_connect_strength_II=0.2
        )
        params_single_ping = ParamsPING(
            nr_excitatory=params_ping.nr_neurons_per_ping[NeuronTypes.EX],
            nr_inhibitory=params_ping.nr_neurons_per_ping[NeuronTypes.IN]
        )
        connectivity = ConnectivitySinglePINGFactory().create(
            params_ping=params_single_ping,
            params_connectivity=params_connectivity
        )
        params_synaptic = ParamsSynaptic(
            rise_E=0.15,
            decay_E=1,
            rise_I=0.2,
            decay_I=7,
        )
        current_components = CurrentComponentsSinglePING(
            connectivity=connectivity,
            params_synaptic=params_synaptic,
            mean_ex=mean_ex,
            var_ex=0,
            mean_in=4,
            var_in=0
        )
        simulation_outcome = IzhikevichNetworkSimulator(params_izhi, current_components).simulate(
            simulation_time=simulation_time,
            dt=1
        )

        return simulation_outcome

