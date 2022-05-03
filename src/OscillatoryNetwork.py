from src.GridConnectivity import *
from src.misc import *
from src.constants import *

from tqdm import tqdm
import numpy as np
from itertools import product


class OscillatoryNetwork:
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
    * :math:`I` describes the _current.

    This neural dynamics model is introduced in :cite:p:`Izhikevich2003`.

    In the network, neurons are not isolated, and the current involves the accumulated effect of interactions
    with other neurons: TODO:: change the equation to match the report

    :math:`I_v = \\begin{cases} \\sum_{w \in V} K_{v, w} I_{syn, w} + I_{stim, v} &\\text{ if } \mathsf{type}(v) = ex \\\ \\sum_{w \in V} K_{v, w} I_{syn, w} &\\text{ if } \mathsf{type}(v) = in \\end{cases}`,

    where

    * :math:`K` is the coupling weights (see :obj:`GridConnectivity`),
    * :math:`I_{syn}` represents the effect of synaptic potentials,
    * :math:`I_{stim}` is the current caused by external stimuli.

    :param stimulus: TODO
    :type stimulus: TODO

    :param nr_excitatory: number of excitatory neurons in the network.
    :type nr_excitatory: int

    :param nr_inhibitory: number of inhibitory neurons in the network.
    :type nr_inhibitory: int

    :param nr_ping_networks: number of ping_networks in the network.
    :type nr_ping_networks: int


    :raises:
        AssertionError: if the number of excitatory neurons is smaller than 2.
    :raises:
        AssertionError: if the number of inhibitory neurons is smaller than 2.


    :ivar _nr_neurons: number of neurons of each type in the network.
    :type _nr_neurons: dict[Any, int]

    :ivar _nr_ping_networks: number of ping_networks in the network.
    :type _nr_ping_networks: int

    :ivar _stimulus: TODO
    :type _stimulus: TODO

    :ivar _currents: current (from input and interaction).
    :type _currents: numpy.ndarray[int, float]

    :ivar _potentials: voltage (membrane potential).
    :type _potentials: numpy.ndarray[int, float]

    :ivar _recovery: membrane recovery variable.
    :type _recovery: numpy.ndarray[int, float]

    :ivar _izhi_alpha: timescale of recovery variable `recovery`.
    :type _izhi_alpha: numpy.ndarray[int, float]

    :ivar _izhi_beta: sensitivity of `recovery` to sub-threshold oscillations of `potential`.
    :type _izhi_beta: numpy.ndarray[int, float]

    :ivar _izhi_gamma: membrane voltage after spike (after-spike reset of `potential`).
    :type _izhi_gamma: numpy.ndarray[int, float]

    :ivar _izhi_zeta: after-spike reset of recovery variable `recovery`.
    :type _izhi_zeta: numpy.ndarray[int, float]
    """

    def __init__(self, stimulus, nr_excitatory, nr_inhibitory, nr_ping_networks=1):

        # FIXME:: this assertions are only there because of the stim_input
        assert nr_excitatory >= 2, "Number of excitatory neurons cannot be smaller than 2."
        assert nr_inhibitory >= 2, "Number of inhibitory neurons cannot be smaller than 2."

        self._nr_neurons = {
            NeuronTypes.E: nr_excitatory,
            NeuronTypes.I: nr_inhibitory,
            "total": nr_excitatory + nr_inhibitory
        }
        self._nr_ping_networks = nr_ping_networks

        self._izhi_alpha = np.array(
            [IZHI_ALPHA[NeuronTypes.E] for _ in range(self._nr_neurons[NeuronTypes.E])] +
            [IZHI_ALPHA[NeuronTypes.I] for _ in range(self._nr_neurons[NeuronTypes.I])]
        )
        self._izhi_beta = np.array(
            [IZHI_BETA[NeuronTypes.E] for _ in range(self._nr_neurons[NeuronTypes.E])] +
            [IZHI_BETA[NeuronTypes.I] for _ in range(self._nr_neurons[NeuronTypes.I])]
        )
        self._izhi_gamma = np.array(
            [IZHI_GAMMA[NeuronTypes.E] for _ in range(self._nr_neurons[NeuronTypes.E])] +
            [IZHI_GAMMA[NeuronTypes.I] for _ in range(self._nr_neurons[NeuronTypes.I])]
        )
        self._izhi_zeta = np.array(
            [IZHI_ZETA[NeuronTypes.E] for _ in range(self._nr_neurons[NeuronTypes.E])] +
            [IZHI_ZETA[NeuronTypes.I] for _ in range(self._nr_neurons[NeuronTypes.I])]
        )

        self._stimulus = stimulus

        # TODO:: change init
        self._currents = np.zeros((self._nr_neurons["total"]))

        self._potentials = np.array(
            [INIT_MEMBRANE_POTENTIAL for _ in range(self._nr_neurons["total"])]
        )
        self._recovery = np.multiply(self._izhi_beta, self._potentials)

    def run_simulation(self, cortical_coords: list[list[tuple[float, float]]], simulation_time: int, dt: float) -> None:
        """
        Runs the simulation.

        Parts of the code in this function and its components are rewritten from MATLAB code listed in supplementary
        materials of :cite:p:`Lowet2015`.

        :param cortical_coords: locations of PING networks in the visual cortex.
        :type cortical_coords: list[list[tuple[float, float]]]

        :param simulation_time: number of epochs to run the simulation.
        :type simulation_time: int

        :param dt: time interval
        :type dt: float

        :rtype: None
        """

        # spike timings
        firing_times = []

        gatings = np.zeros(self._nr_neurons["total"])

        stim_input = self._create_main_input_stimulus()

        coupling_weights = GridConnectivity(
            nr_neurons=self._nr_neurons,
            nr_ping_networks=self._nr_ping_networks,
            cortical_coords=cortical_coords
        ).coupling_weights

        for t in (pbar := tqdm(range(simulation_time))):
            pbar.set_description("Network simulation")

            # spiking
            fired = np.argwhere(self._potentials > THRESHOLD_POTENTIAL).flatten()  # indices of spikes
            # firing times will be used later
            firing_times = [firing_times, [t for _ in range(len(fired))] + fired]
            for f in fired:
                self._potentials[f] = self._izhi_gamma[f]
                self._recovery[f] += self._izhi_zeta[f]

            # synaptic current
            syn_currents, gatings = self._get_synaptic_currents(gatings=gatings, dt=dt)
            # total current
            self._currents = stim_input + self._get_thalamic_input() + np.matmul(coupling_weights, syn_currents)
            
            # updating potential and recovery
            self._potentials = self._potentials + self._get_change_in_potentials()
            self._potentials = self._potentials + self._get_change_in_potentials()
            self._recovery = self._recovery + self._get_change_in_recovery()

    def _get_change_in_recovery(self) -> np.ndarray[int, float]:
        """
        Computes the change in membrane recovery.

        Computes :math:`dr_v / dt = \\alpha_{\mathsf{type}(v)} \cdot (\\beta_{\mathsf{type}(v)} p_v - r_v)`.

        :return: change in membrane recovery.
        :rtype: numpy.ndarray[int, float]
        """

        return np.multiply(
            self._izhi_alpha,
            np.multiply(self._izhi_beta, self._potentials) - self._recovery
        )

    def _get_change_in_potentials(self) -> np.ndarray[int, float]:
        """
        Computes the change in membrane potentials.

        Computes :math:`dp_v / dt = 0.04 p_v^2 + 5 p_v + 140 - r_v + I_v`.

        :return: change in membrane potentials.
        :rtype: numpy.ndarray[int, float]
        """

        # TODO:: why do we multiply the equation for dv/dt with 0.5 and then call this function twice in run_simulation?
        return 0.5 * (0.04 * self._potentials ** 2 + 5 * self._potentials + 140 - self._recovery + self._currents)

    def _get_gatings(
            self, gatings: np.ndarray[int, float], dt: float,
    ) -> np.ndarray[int, float]:
        """
        Computes the gating values for synapses of given types.

        :param gatings: current gating values.
        :type gatings: numpy.ndarray[int, float]

        :param dt: time interval.
        :type dt: float

        :return: the change in synaptic values for a unit of time.
        :rtype: numpy.ndarray[int, float]
        """

        new_gatings = np.zeros(self._nr_neurons["total"])

        for nt in [NeuronTypes.E, NeuronTypes.I]:
            nt_slice = neur_slice(nt, self._nr_neurons[NeuronTypes.E], self._nr_neurons[NeuronTypes.I])

            transmission_concs = 1 + np.tanh(self._potentials[nt_slice] / 4)
            change_gatings = (
                SYNAPTIC_RISE[nt] * transmission_concs * (1 - gatings[nt_slice]) -
                gatings[nt_slice] / SYNAPTIC_DECAY[nt]
            )
            new_gatings[nt_slice] = gatings[nt_slice] + dt * change_gatings

        return new_gatings

    def _get_synaptic_currents(
            self, gatings: np.ndarray[(int, int), float], dt: float
    ) -> tuple[np.ndarray[int, float], np.ndarray[int, float]]:
        """
        Computes the new synaptic currents for postsynaptic neurons.

        Computes the :math:`I_{syn}`.

        :param gatings: synaptic gating values
        :type gatings: numpy.ndarray[int, float]

        :param dt: time interval
        :type dt: float

        :return: change in synaptic gates for excitatory postsynaptic neurons.
        :rtype: numpy.ndarray[(int, int), float]
        """

        new_gatings = self._get_gatings(gatings, dt)
        new_currents = np.zeros(self._nr_neurons["total"])

        for postsyn_type, presyn_type in list(product([NeuronTypes.E, NeuronTypes.I], repeat=2)):
            presyn_slice = neur_slice(presyn_type, self._nr_neurons[NeuronTypes.E], self._nr_neurons[NeuronTypes.I])
            postsyn_slice = neur_slice(postsyn_type, self._nr_neurons[NeuronTypes.E], self._nr_neurons[NeuronTypes.I])

            # conductance calculation between neurons (synapse)
            conductances = SYNAPTIC_CONDUCTANCE[(presyn_type, postsyn_type)] * new_gatings[presyn_slice]

            # synaptic current calculation of a postsynaptic neuron
            new_currents[postsyn_slice] += \
                sum(conductances) * (self._potentials[postsyn_slice] - REVERSAL_POTENTIAL[presyn_type])

        return new_currents, new_gatings

    def _get_thalamic_input(self) -> np.ndarray[int, float]:
        """
        Generates the thalamic input.

        :return: change in thalamic input.
        :rtype: numpy.ndarray[int, float]
        """

        # TODO:: what is this exactly?
        return np.append(
            GAUSSIAN_INPUT[NeuronTypes.E] * np.random.randn(self._nr_neurons[NeuronTypes.E]),
            GAUSSIAN_INPUT[NeuronTypes.I] * np.random.randn(self._nr_neurons[NeuronTypes.I])
        )

    def _create_main_input_stimulus(self) -> list[float]:
        """
        Parses external input stimulus. ARTIFICIAL FUNCTION - REAL NOT IMPLEMENTED YET.

        Creates initial :math:`I_{stim}`.

        :return: input stimulus.
        :rtype: list[float]
        """

        # TODO:: implement the real strategy

        stim_input = []
        for i in self._stimulus:
            stim_input += [i] * \
                          ((self._nr_neurons[NeuronTypes.E] + self._nr_neurons[
                              NeuronTypes.I]) // self._nr_ping_networks)

        return stim_input

        # old code:
        # # sinusoidal spatial modulation of input strength
        # amplitude = 1
        # # mean input level to RS cells
        # mean_input_lvl_RS = 7
        # step = 2 * pi / (self._nr_neurons[NeuronTypes.E] - 1)
        # stim_input = mean_input_lvl_RS + amplitude * np.sin(
        #     crange(-pi, pi, step)
        # )
        # # additional mean input to FS cells
        # stim_input = np.append(stim_input, 3.5 * np.ones(self._nr_neurons[NeuronTypes.I]))
        #
        # print("stim input\n", stim_input)
        #
        # return stim_input
