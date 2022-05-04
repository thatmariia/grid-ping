from src.constants import *
from src.CurrentComponents import *
from src.NeuronTypes import *
from src.IzhikevichNetworkOutcome import *

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
    * :math:`\\alpha` describes the timescale of :math:`r` (see :obj:`constants.IZHI_ALPHA`),
    * :math:`\\beta` describes the sensitivity of :math:`r` to the subthreshold fluctuations of :math:`p` (see :obj:`constants.IZHI_BETA`),
    * :math:`\\gamma` describes the after-spike reset value of :math:`p` (see :obj:`constants.IZHI_GAMMA`),
    * :math:`\\zeta` describes the after-spike reset of :math:`r` (see :obj:`constants.IZHI_ZETA`),
    * :math:`I` describes the current (see :obj:`CurrentComponents`).

    This neural dynamics model is introduced in :cite:p:`Izhikevich2003`.

    :param current_components: contains methods of computing the neural network current components.
    :type current_components: CurrentComponents

    :ivar _current_components: contains methods of computing the neural network current components.
    """

    def __init__(self, current_components: CurrentComponents):
        self._current_components: CurrentComponents = current_components

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
        gatings = np.zeros(self._current_components.connectivity.nr_neurons["total"])

        # spike timings
        firing_times = []

        for t in (pbar := tqdm(range(simulation_time))):
            pbar.set_description("Network simulation")

            # spiking
            fired = np.argwhere(potentials > PEAK_POTENTIAL).flatten()  # indices of spikes
            # firing times will be used later
            # TODO:: wtf is going on here?
            firing_times = [firing_times, [t for _ in range(len(fired))] + fired]
            for f in fired:
                potentials[f] = izhi_gamma[f]
                recovery[f] += izhi_zeta[f]

            # synaptic current
            syn_currents, gatings = self._current_components.get_synaptic_currents(gatings, dt, potentials)
            # total current
            self._currents = self._current_components.get_current_input() + np.matmul(
                self._current_components.connectivity.coupling_weights, syn_currents
            )

            # updating potential and recovery
            potentials = potentials + self._get_change_in_potentials(potentials, recovery)
            potentials = potentials + self._get_change_in_potentials(potentials, recovery)
            recovery = recovery + self._get_change_in_recovery(potentials, recovery, izhi_alpha, izhi_beta)

        outcome = IzhikevichNetworkOutcome(
            firing_times=firing_times
        )

        return outcome

    def _get_izhi_parameters(self) \
            -> tuple[np.ndarray[int, float], np.ndarray[int, float], np.ndarray[int, float], np.ndarray[int, float]]:
        """
        Allocates Izhikevich parameters :math:`\\alpha, \\beta, \\gamma, \\zeta` to all neurons.

        :return: Izhikevich parameters.
        :rtype: tuple[numpy.ndarray[int, float], numpy.ndarray[int, float], numpy.ndarray[int, float], numpy.ndarray[int, float]]
        """

        izhi_alpha = np.array(
            [IZHI_ALPHA[NeuronTypes.E] for _ in range(self._current_components.connectivity.nr_neurons[NeuronTypes.E])] +
            [IZHI_ALPHA[NeuronTypes.I] for _ in range(self._current_components.connectivity.nr_neurons[NeuronTypes.I])]
        )
        izhi_beta = np.array(
            [IZHI_BETA[NeuronTypes.E] for _ in range(self._current_components.connectivity.nr_neurons[NeuronTypes.E])] +
            [IZHI_BETA[NeuronTypes.I] for _ in range(self._current_components.connectivity.nr_neurons[NeuronTypes.I])]
        )
        izhi_gamma = np.array(
            [IZHI_GAMMA[NeuronTypes.E] for _ in range(self._current_components.connectivity.nr_neurons[NeuronTypes.E])] +
            [IZHI_GAMMA[NeuronTypes.I] for _ in range(self._current_components.connectivity.nr_neurons[NeuronTypes.I])]
        )
        izhi_zeta = np.array(
            [IZHI_ZETA[NeuronTypes.E] for _ in range(self._current_components.connectivity.nr_neurons[NeuronTypes.E])] +
            [IZHI_ZETA[NeuronTypes.I] for _ in range(self._current_components.connectivity.nr_neurons[NeuronTypes.I])]
        )

        return izhi_alpha, izhi_beta, izhi_gamma, izhi_zeta

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

        potentials = np.array(
            [INIT_MEMBRANE_POTENTIAL for _ in range(self._current_components.connectivity.nr_neurons["total"])]
        )
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

        return np.multiply(
            izhi_alpha,
            np.multiply(izhi_beta, potentials) - recovery
        )

    def _get_change_in_potentials(
            self, potentials: np.ndarray[int, float], recovery: np.ndarray[int, float]
    ) -> np.ndarray[int, float]:
        """
        Computes the change in membrane potentials.

        Computes :math:`dp_v / dt = 0.04 p_v^2 + 5 p_v + 140 - r_v + I_v`.

        :param potentials: neurons' membrane potentials.
        :type potentials: numpy.ndarray[int, float]

        :param recovery: recovery variables.
        :type recovery: numpy.ndarray[int, float]

        :return: change in membrane potentials.
        :rtype: numpy.ndarray[int, float]
        """

        # TODO:: why do we multiply the equation for dv/dt with 0.5 and then call this function twice in run_simulation?
        return 0.5 * (0.04 * potentials ** 2 + 5 * potentials + 140 - recovery + self._currents)





