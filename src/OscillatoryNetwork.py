from src.GridConnectivity import *
from src.misc import *
from src.constants import *

from tqdm import tqdm
import numpy as np


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
    with other neurons:

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
    :type _nr_neurons: dict[NeuronTypes: int]

    :ivar _nr_ping_networks: number of ping_networks in the network.
    :type _nr_ping_networks: int

    :ivar _stimulus: TODO
    :type _stimulus: TODO

    :ivar _synaptic_potentials: synaptic potentials.
    :type _synaptic_potentials: ndarray[float]

    :ivar _current: current (from input and interaction).
    :type _current: ndarray[float]

    :ivar _potentials: voltage (membrane potential).
    :type _potentials: ndarray[float]

    :ivar _recovery: membrane recovery variable.
    :type _recovery: ndarray[float]

    :ivar _izhi_alpha: timescale of recovery variable `recovery`.
    :type _izhi_alpha: ndarray[float]

    :ivar _izhi_beta: sensitivity of `recovery` to sub-threshold oscillations of `potential`.
    :type _izhi_beta: ndarray[float]

    :ivar _izhi_gamma: membrane voltage after spike (after-spike reset of `potential`).
    :type _izhi_gamma: ndarray[float]

    :ivar _izhi_zeta: after-spike reset of recovery variable `recovery`.
    :type _izhi_zeta: ndarray[float]
    """

    def __init__(self, stimulus, nr_excitatory, nr_inhibitory, nr_ping_networks=1):

        # FIXME:: this assertions are only there because of the stim_input
        assert nr_excitatory >= 2, "Number of excitatory neurons cannot be smaller than 2."
        assert nr_inhibitory >= 2, "Number of inhibitory neurons cannot be smaller than 2."

        self._nr_neurons = {
            NeuronTypes.E: nr_excitatory,
            NeuronTypes.I: nr_inhibitory
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

        self._synaptic_potentials = {
            NeuronTypes.E: np.zeros(self._nr_neurons[NeuronTypes.E]),
            NeuronTypes.I: np.zeros(self._nr_neurons[NeuronTypes.I])
        }

        self._current = np.array((0, 0))

        self._potentials = np.array(
            [INIT_MEMBRANE_POTENTIAL for _ in range(self._nr_neurons[NeuronTypes.E] + self._nr_neurons[NeuronTypes.I])]
        )
        self._recovery = np.multiply(self._izhi_beta, self._potentials)

    def run_simulation(self, simulation_time, dt):
        """
        Runs the simulation.

        Parts of the code in this function and its components are rewritten from MATLAB code listed in supplementary
        materials of :cite:p:`Lowet2015`.

        :param simulation_time: number of epochs to run the simulation.
        :param dt: TODO

        :rtype: None
        """

        # spike timings
        firing_times = []

        stim_input = self._create_main_input_stimulus()

        connectivity = GridConnectivity(
            nr_neurons=self._nr_neurons,
            nr_ping_networks=self._nr_ping_networks
        )

        print("Simulation started")

        for t in tqdm(range(simulation_time)):

            fired = np.argwhere(self._potentials > 30).flatten()  # indices of spikes
            # TODO:: why do we need firing_times?
            firing_times = [firing_times, [t for _ in range(len(fired))] + fired]
            for f in fired:
                self._potentials[f] = self._izhi_gamma[f]
                self._recovery[f] += self._izhi_zeta[f]

            # thalamic input
            self._current = np.add(stim_input, self._change_thalamic_input())

            # synaptic _potentials
            # TODO:: change directly into 1 array later when I know what's going on
            self._synaptic_potentials[NeuronTypes.E] = np.add(
                self._synaptic_potentials[NeuronTypes.E],
                self._change_synaptic_potentials(neuron_type=NeuronTypes.E, dt=dt)
            )
            self._synaptic_potentials[NeuronTypes.I] = np.add(
                self._synaptic_potentials[NeuronTypes.I],
                self._change_synaptic_potentials(neuron_type=NeuronTypes.I, dt=dt)
            )
            gsyn = np.append(self._synaptic_potentials[NeuronTypes.E], self._synaptic_potentials[NeuronTypes.I])

            # defining input to eah neuron as the summation of all synaptic input
            # form all connected neurons
            self._current = np.add(self._current, np.matmul(connectivity.coupling_weights, gsyn))

            self._potentials = np.add(self._potentials, self._change_potential())
            self._potentials = np.add(self._potentials, self._change_potential())
            self._recovery = np.add(self._recovery, self._change_recovery())

        print("Simulation ended")

    def _change_recovery(self):
        """
        Computes the change in membrane recovery.

        Computes :math:`dr_v / dt = \\alpha_{\mathsf{type}(v)} \cdot (\\beta_{\mathsf{type}(v)} p_v - r_v)`.

        :return: change in membrane recovery.
        :rtype: ndarray[float]
        """

        return np.multiply(
            self._izhi_alpha,
            np.multiply(self._izhi_beta, self._potentials) - self._recovery
        )

    def _change_potential(self):
        """
        Computes the change in membrane potentials.

        Computes :math:`dp_v / dt = 0.04 p_v^2 + 5 p_v + 140 - r_v + I_v`.

        :return: change in membrane potentials.
        :rtype: ndarray[float]
        """

        # TODO:: why do we multiply the equation for dv/dt with 0.5 and then call this function twice in run_simulation?
        return 0.5 * (0.04 * self._potentials ** 2 + 5 * self._potentials + 140 - self._recovery + self._current)

    def _change_synaptic_potentials(self, neuron_type, dt):
        """
        Computes the change in synaptic gates for postsynaptic neurons.

        Computes the change in :math:`I_{syn}`.
        TODO:: maybe make a separate class for this component later

        :param neuron_type: neuron type
        :type neuron_type: NeuronTypes

        :param dt: TODO

        :return: change in synaptic gates for excitatory postsynaptic neurons.
        :rtype: ndarray[float]
        """

        potentials = self._potentials[
            neur_slice(neuron_type, self._nr_neurons[NeuronTypes.E], self._nr_neurons[NeuronTypes.I])
        ]
        # TODO:: in the paper, this computation is different
        alpha = potentials / 10.0 + 2
        z = np.tanh(alpha)
        comp1 = (z + 1) / 2.0
        comp2 = (1 - self._synaptic_potentials[neuron_type]) / SYNAPTIC_CONST_RISE[neuron_type]
        comp3 = self._synaptic_potentials[neuron_type] / SYNAPTIC_CONST_DECAY[neuron_type]
        return dt * 0.3 * (np.multiply(comp1, comp2) - comp3)

    def _change_thalamic_input(self):
        """
        Computes the change in thalamic input.

        :return: change in thalamic input.
        :rtype: ndarray[float]
        """

        # TODO:: what is this exactly?
        return np.append(
            GAUSSIAN_INPUT[NeuronTypes.E] * np.random.randn(self._nr_neurons[NeuronTypes.E]),
            GAUSSIAN_INPUT[NeuronTypes.I] * np.random.randn(self._nr_neurons[NeuronTypes.I])
        )

    def _create_main_input_stimulus(self):
        """
        Parses external input stimulus. ARTIFICIAL FUNCTION - REAL NOT IMPLEMENTED YET.

        Creates initial :math:`I_{stim}`.

        :return: input stimulus.
        :rtype: ndarray[float]
        """

        # TODO:: implement the real strategy

        stim_input = []
        for i in self._stimulus:
            stim_input += [i] * \
                          ((self._nr_neurons[NeuronTypes.E] + self._nr_neurons[NeuronTypes.I]) // self._nr_ping_networks)

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
