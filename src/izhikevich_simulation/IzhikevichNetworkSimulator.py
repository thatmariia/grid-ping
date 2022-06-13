from src.params.ParamsIzhikevich import *

from src.izhikevich_simulation.CurrentComponents import *
from src.izhikevich_simulation.IzhikevichNetworkOutcome import *
from src.SpikingFrequencyComputer import *

from tqdm import tqdm
import numpy as np

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
    :type pb_off: Bool


    :ivar _params_izhi: contains Izhikevich parameters.
    :ivar _current_components: contains methods of computing the neural network current components.
    :ivar _pb_off: indicates whether the progress bar should be off.
    """

    def __init__(self, params_izhi: ParamsIzhikevich, params_freqs, current_components: CurrentComponents, pb_off=True):
        self._params_izhi: ParamsIzhikevich = params_izhi
        self.params_freqs = params_freqs
        self._current_components: CurrentComponents = current_components
        self._pb_off = pb_off

    def simulate(self, simulation_time: int, dt: float) -> IzhikevichNetworkOutcome:
        """
        Runs the simulation.

        Parts of the code in this function and its components are rewritten from MATLAB code listed in supplementary
        materials of :cite:p:`Lowet2015`.

        :param simulation_time: number of epochs to run the simulation.
        :type simulation_time: int

        :param dt: time interval
        :type dt: float

        :return: collected information from the simulation.
        :rtype: IzhikevichNetworkOutcome
        """

        izhi_alpha, izhi_beta, izhi_gamma, izhi_zeta = self._get_izhi_parameters()
        potentials, recovery = self._get_initial_values(izhi_beta)

        # spike timings
        spikes: list[tuple[int, int]] = []

        for t in (pbar := tqdm(range(simulation_time), disable=self._pb_off)):
            pbar.set_description("Network simulation")

            # spiking
            fired_neurons_ids = np.argwhere(potentials >= self._params_izhi.peak_potential).flatten()  # indices of spikes
            for id in fired_neurons_ids:
                spikes.append((t, id))

                potentials[id] = izhi_gamma[id]
                recovery[id] += izhi_zeta[id]

            # current
            synaptic_currents = self._current_components.get_synaptic_currents(dt, potentials)
            current_input = self._current_components.get_current_input()
            currents = current_input + np.matmul(
                self._current_components.connectivity.coupling_weights,
                synaptic_currents
            )

            # updating potential and recovery
            potentials = potentials + 0.5 * self._get_change_in_potentials(potentials, recovery, currents)
            potentials = potentials + 0.5 * self._get_change_in_potentials(potentials, recovery, currents)
            recovery = recovery + self._get_change_in_recovery(potentials, recovery, izhi_alpha, izhi_beta)

            if (t > 0) and (t % 1000 == 0):

                outcome = IzhikevichNetworkOutcome(
                    spikes=spikes,
                    params_ping=self._current_components.connectivity.params_ping,
                    simulation_time=simulation_time,
                    grid_geometry=self._current_components.connectivity.grid_geometry
                )

                ping_frequencies = SpikingFrequencyComputer().compute_for_all_pings(
                    simulation_outcome=outcome,
                    params_freqs=self.params_freqs
                )
                SpikingFrequencyComputer().plot_ping_frequencies(ping_frequencies, t=t)

        # outcome = IzhikevichNetworkOutcome(
        #     spikes=spikes,
        #     params_ping=self._current_components.connectivity.params_ping,
        #     simulation_time=simulation_time,
        #     grid_geometry=self._current_components.connectivity.grid_geometry
        # )

        return outcome

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





