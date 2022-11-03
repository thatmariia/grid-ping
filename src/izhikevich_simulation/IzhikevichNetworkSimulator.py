from src.params.ParamsIzhikevich import *

from src.izhikevich_simulation.CurrentComponents import *
from src.after_simulation_analysis.SpikingFrequencyFactory import *
# from src.plotter.ping_frequencies import plot_ping_frequencies, plot_single_ping_frequency_evolution

from tqdm import tqdm
import numpy as np
from itertools import product

import matplotlib.pyplot as plt


class IzhikevichNetworkSimulator:
    """
    This class runs the simulation of the network of neurons.

    Every neuron in the system can be described with a 2D system of ODEs:

    :math:`dp_v / dt = 0.04 p_v^2 + 5 p_v + 140 - r_v + I_v`,

    :math:`dr_v / dt = \\alpha_{\mathsf{type}(v)} \cdot (\\beta_{\mathsf{type}(v)} p_v - r_v)`,
    if :math:`p_v \geq 30` mV, then :math:`\\begin{cases} p_v \leftarrow \gamma_{\mathsf{type}(v)} \\\ r_v \leftarrow r_v + \zeta_{\mathsf{type}(v)} \\end{cases}`,

    where

    * :math:`v` is a neuron,
    * :math:`\mathsf{type}(v)` maps a neuron to its type (see :obj:`NeuronTypes`),
    * :math:`p` represents the membrane potential of the neuron,
    * :math:`r` represents a membrane recovery variable; provides negative feedback to :math:`p`,
    * :math:`\\alpha, \\beta, \\gamma, \\zeta` are Izhikevich parameters (see :obj:`ParamsIzhikevich`),
    * :math:`I` describes the current (see :obj:`CurrentComponents`).

    This neural dynamics model is introduced in :cite:p:`Izhikevich2003`.

    :param params_izhi: contains Izhikevich parameters.
    :type params_izhi: ParamsIzhikevich

    :param current_components: contains methods of computing the neural network current components.
    :type current_components: CurrentComponents

    :param pb_off: indicates whether the progress bar should be off, default is True.
    :type pb_off: bool


    :ivar _params_izhi: contains Izhikevich parameters.
    :ivar _current_components: contains methods of computing the neural network current components.
    :ivar _pb_off: indicates whether the progress bar should be off.
    """

    def __init__(
            self,
            params_izhi: ParamsIzhikevich,
            current_components: CurrentComponents,
            pb_off: bool = True
    ):
        self._params_izhi: ParamsIzhikevich = params_izhi
        self._current_components: CurrentComponents = current_components
        self._pb_off = pb_off

    def simulate(
            self, simulation_time: int, dt: float, params_freqs: Union[None, ParamsFrequencies] = None
    ) -> IzhikevichNetworkOutcome:
        """
        Runs the simulation.

        Parts of the code in this function and its components are rewritten from MATLAB code listed in supplementary
        materials of :cite:p:`Lowet2015`.

        :param simulation_time: number of epochs to run the simulation.
        :type simulation_time: int

        :param dt: time interval
        :type dt: float

        :param params_freqs: TODO, default is None
        :type params_freqs: TODO

        :return: collected information from the simulation.
        :rtype: IzhikevichNetworkOutcome
        """

        izhi_alpha, izhi_beta, izhi_gamma, izhi_zeta = self._get_izhi_parameters()
        potentials, recovery = self._get_initial_values(izhi_beta)

        # spike timings
        spikes: list[tuple[int, int]] = []

        # # ping_freq_evol
        # ping_evol_id = self._current_components.connectivity.params_ping.nr_ping_networks // 2
        # ping_freq_evol = []

        for t in (pbar := tqdm(range(simulation_time), disable=self._pb_off)):
            pbar.set_description("Network simulation")

            # spiking
            fired_neurons_ids = np.argwhere(
                potentials >= self._params_izhi.peak_potential).flatten()  # indices of spikes
            for neur_id in fired_neurons_ids:
                spikes.append((t, neur_id))

                potentials[neur_id] = izhi_gamma[neur_id]
                recovery[neur_id] += izhi_zeta[neur_id]

            # current
            synaptic_currents = self._current_components.get_synaptic_currents(dt, potentials)

            def get_cost(i):
                sum_costs = 0

                nt_i = NeuronTypes.EX \
                    if i in self._current_components.connectivity.params_ping.neur_indices[NeuronTypes.EX] \
                    else NeuronTypes.IN
                ping_id_i = self._current_components.connectivity.grid_geometry.find_neurons_ping(i, nt_i)

                for nt_j in [NeuronTypes.EX, NeuronTypes.IN]:

                    coupling_weights_ij = np.repeat(
                        self._current_components.connectivity.coupling_weights[(nt_i, nt_j)][ping_id_i],
                        self._current_components.connectivity.params_ping.nr_neurons_per_ping[nt_j]
                    )
                    synaptic_currents_j = synaptic_currents[self._current_components.connectivity.params_ping.neur_indices[nt_j]]
                    costs_j = sum(coupling_weights_ij * synaptic_currents_j)
                    sum_costs += costs_j


                    # for j in self._current_components.connectivity.params_ping.neur_indices[nt_j]:
                    #     ping_id_j = self._current_components.connectivity.grid_geometry.find_neurons_ping(j, nt_j)
                    #     cost_i_j = self._current_components.connectivity.coupling_weights[(nt_i, nt_j)][
                    #                    ping_id_i, ping_id_j] * synaptic_currents[j]
                    #     sum_costs += cost_i_j

                return sum_costs

            cost = np.array([get_cost(i) for i in range(self._current_components.connectivity.params_ping.nr_neurons["total"])])

            current_input = self._current_components.get_current_input()
            currents = current_input + cost
            # currents = current_input + np.matmul(
            #     self._current_components.connectivity.coupling_weights,
            #     synaptic_currents
            # )

            # updating potential and recovery
            potentials = potentials + 0.5 * self._get_change_in_potentials(potentials, recovery, currents)
            potentials = potentials + 0.5 * self._get_change_in_potentials(potentials, recovery, currents)
            recovery = recovery + self._get_change_in_recovery(potentials, recovery, izhi_alpha, izhi_beta)

            # if (params_freqs is not None) and (len(spikes) > 0):
            #     simulation_outcome = IzhikevichNetworkOutcome(
            #         spikes=spikes,
            #         params_ping=self._current_components.connectivity.params_ping,
            #         params_freqs=params_freqs,
            #         simulation_time=simulation_time,
            #         grid_geometry=self._current_components.connectivity.grid_geometry
            #     )
            #     spiking_frequencies = SpikingFrequencyFactory().create(
            #         simulation_outcome=simulation_outcome
            #     )
            #     ping_freq_evol.append(spiking_frequencies.ping_frequencies[ping_evol_id])
            #
            #     # plotting frequency distribution every 1000 epochs
            #     if (t + 1) % 100 == 0:
            #         plot_ping_frequencies(spiking_frequencies.ping_frequencies, round(t * dt, 2))

        simulation_outcome = IzhikevichNetworkOutcome(
            spikes=spikes,
            params_ping=self._current_components.connectivity.params_ping,
            params_freqs=params_freqs,
            simulation_time=simulation_time,
            grid_geometry=self._current_components.connectivity.grid_geometry
        )

        return simulation_outcome

    # def _get_cost(self, i, synaptic_currents):
    #     sum_costs = 0
    #
    #     nt_i = NeuronTypes.EX if i in self._current_components.connectivity.params_ping.neur_indices[NeuronTypes.EX] \
    #         else NeuronTypes.IN
    #     # to which ping network i belongs
    #     ping_id_i = self._current_components.connectivity.grid_geometry.find_neurons_ping(i, nt_i)
    #
    #     for nt_j in [NeuronTypes.EX, NeuronTypes.IN]:
    #         for j in self._current_components.connectivity.params_ping.neur_indices[nt_j]:
    #             ping_id_j = self._current_components.connectivity.grid_geometry.find_neurons_ping(j, nt_j)
    #             cost_i_j = self._current_components.connectivity.coupling_weights[(nt_i, nt_j)][ping_id_i, ping_id_j] * synaptic_currents[j]
    #             sum_costs += cost_i_j
    #
    #     return sum_costs


    def _get_izhi_parameters(self) \
            -> tuple[np.ndarray[int, float], ...]:
        """
        Allocates Izhikevich parameters :math:`\\alpha, \\beta, \\gamma, \\zeta` to all neurons.

        :return: Izhikevich parameters.
        :rtype: tuple[np.ndarray[int, float], ...]
        """

        params_per_neuron = []
        for param in [self._params_izhi.alpha, self._params_izhi.beta, self._params_izhi.gamma, self._params_izhi.zeta]:
            param_per_neuron = np.array(
                [param[NeuronTypes.EX]
                 for _ in range(self._current_components.connectivity.params_ping.nr_neurons[NeuronTypes.EX])] +
                [param[NeuronTypes.IN]
                 for _ in range(self._current_components.connectivity.params_ping.nr_neurons[NeuronTypes.IN])]
            )
            params_per_neuron.append(param_per_neuron)

        return tuple(params_per_neuron)

    def _get_initial_values(
            self, izhi_beta: np.ndarray[int, float]
    ) -> tuple[np.ndarray[int, float], np.ndarray[int, float]]:
        """
        Creates initial values for the membrane potential and recovery variable.

        :param izhi_beta: Izhikevich parameter :math:`\\beta` for all neurons.
        :type izhi_beta: numpy.ndarray[int, float]

        :return: initial values for the membrane potential and recovery variable.
        :rtype: tuple[numpy.ndarray[int, float], numpy.ndarray[int, float]]
        """

        potentials = np.array([
            -65 for _ in range(self._current_components.connectivity.params_ping.nr_neurons["total"])
        ])
        recovery = np.multiply(izhi_beta, potentials)
        return potentials, recovery

    def _get_change_in_recovery(
            self, potentials: np.ndarray[int, float], recovery: np.ndarray[int, float],
            izhi_alpha: np.ndarray[int, float], izhi_beta: np.ndarray[int, float]
    ) -> np.ndarray[int, float]:
        """
        Computes the change in membrane recovery.

        Computes :math:`dr_v / dt = \\alpha_{\mathsf{type}(v)} \cdot (\\beta_{\mathsf{type}(v)} p_v - r_v)`.

        :param potentials: neurons' membrane potentials.
        :type potentials: numpy.ndarray[int, float]

        :param recovery: recovery variables.
        :type recovery: numpy.ndarray[int, float]

        :param izhi_alpha: Izhikevich parameter :math:`\\alpha` for all neurons.
        :type izhi_alpha: numpy.ndarray[int, float]

        :param izhi_beta: Izhikevich parameter :math:`\\beta` for all neurons.
        :type izhi_beta: numpy.ndarray[int, float]

        :return: change in membrane recovery.
        :rtype: numpy.ndarray[int, float]
        """

        return izhi_alpha * (izhi_beta * potentials - recovery)

    def _get_change_in_potentials(
            self, potentials: np.ndarray[int, float], recovery: np.ndarray[int, float], currents: np.ndarray[int, float]
    ) -> np.ndarray[int, float]:
        """
        Computes the change in membrane potentials.

        Computes :math:`dp_v / dt = 0.04 p_v^2 + 5 p_v + 140 - r_v + I_v`.

        :param potentials: neurons' membrane potentials.
        :type potentials: numpy.ndarray[int, float]

        :param recovery: recovery variables.
        :type recovery: numpy.ndarray[int, float]

        :param currents: currents.
        :type currents: numpy.ndarray[int, float]

        :return: change in membrane potentials.
        :rtype: numpy.ndarray[int, float]
        """

        # TODO:: why do we multiply the equation for dv/dt with 0.5 and then call this function twice in run_simulation?
        return 0.04 * np.power(potentials, 2) + 5 * potentials + 140 - recovery + currents
