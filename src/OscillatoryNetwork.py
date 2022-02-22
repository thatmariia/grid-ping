from src.GridConnectivity import *
from src.misc import *
from src.constants import *

from tqdm import tqdm
import numpy as np
from math import pi


class OscillatoryNetwork:
    """
    This class runs the simulation of the network of OscillatoryNetwork oscillators.

    :param nr_excit: number of excitatory neurons in the network.
    :type nr_excit: int

    :param nr_inhibit: number of inhibitory neurons in the network.
    :type nr_inhibit: int

    :param nr_oscillators: number of oscillators in the network.
    :type nr_oscillators: int


    :raises:
        AssertionError: if the number of excitatory neurons is smaller than 2.
    :raises:
        AssertionError: if the number of inhibitory neurons is smaller than 2.


    :ivar nr_excit: number of inhibitory neurons in the network.
    :type nr_excit: int

    :ivar nr_inhibit: number of inhibitory neurons in the network.
    :type nr_inhibit: int

    :ivar nr_neurons: total number of neurons in the network.
    :type nr_neurons: int

    :ivar nr_oscillators: number of oscillators in the network.
    :type nr_oscillators: int

    :ivar potential: voltage (membrane potential).
    :type potential: ndarray[float]

    :ivar recovery: membrane recovery variable.
    :type recovery: ndarray[float]

    :ivar izhi_alpha: timescale of recovery variable u.
    :type izhi_alpha: ndarray[float]

    :ivar izhi_beta: sensitivity of u to sub-threshold oscillations of v.
    :type izhi_beta: ndarray[float]

    :ivar izhi_gamma: membrane voltage after spike (after-spike reset of v).
    :type izhi_gamma: ndarray[float]

    :ivar izhi_zeta: after-spike reset of recovery variable u.
    :type izhi_zeta: ndarray[float]
    """

    def __init__(self, nr_excit, nr_inhibit, nr_oscillators=1):

        # FIXME:: this assertions are only there because of the stim_input
        assert nr_excit >= 2, "Number of excitatory neurons cannot be smaller than 2."
        assert nr_inhibit >= 2, "Number of inhibitory neurons cannot be smaller than 2."

        self.nr_excit = nr_excit
        self.nr_inhibit = nr_inhibit
        self.nr_neurons = self.nr_excit + self.nr_inhibit
        self.nr_oscillators = nr_oscillators

        self.izhi_alpha = np.array(
            [IZHI_ALPHA[NeuronTypes.E] for _ in range(nr_excit)] +
            [IZHI_ALPHA[NeuronTypes.I] for _ in range(nr_inhibit)]
        )
        self.izhi_beta = np.array(
            [IZHI_BETA[NeuronTypes.E] for _ in range(nr_excit)] +
            [IZHI_BETA[NeuronTypes.I] for _ in range(nr_inhibit)]
        )
        self.izhi_gamma = np.array(
            [IZHI_GAMMA[NeuronTypes.E] for _ in range(nr_excit)] +
            [IZHI_GAMMA[NeuronTypes.I] for _ in range(nr_inhibit)]
        )
        self.izhi_zeta = np.array(
            [IZHI_ZETA[NeuronTypes.E] for _ in range(nr_excit)] +
            [IZHI_ZETA[NeuronTypes.I] for _ in range(nr_inhibit)]
        )

        self.potential = np.array([INIT_MEMBRANE_POTENTIAL for _ in range(nr_excit + nr_inhibit)])
        self.recovery = np.multiply(self.izhi_beta, self.potential)

    def run_simulation(self, simulation_time, dt):
        """
        Runs the simulation.

        :param simulation_time: number of epochs to run the simulation.
        :param dt: TODO

        :rtype: None
        """

        # spike timings
        firing_times = []

        ampa = np.zeros(self.nr_excit)
        gaba = np.zeros(self.nr_inhibit)

        stim_input = self._create_main_input_stimulus()

        connectivity = GridConnectivity(
            nr_excit=self.nr_excit,
            nr_inhibit=self.nr_inhibit,
            nr_oscillators=self.nr_oscillators
        )

        print("Simulation started")

        for t in tqdm(range(simulation_time)):

            fired = np.argwhere(self.potential > 30).flatten()  # indices of spikes
            # TODO:: why do we need firing_times?
            firing_times = [firing_times, [t for _ in range(len(fired))] + fired]
            for f in fired:
                self.potential[f] = self.izhi_gamma[f]
                self.recovery[f] += self.izhi_zeta[f]

            # thalamic input
            I = np.add(stim_input, self._change_thalamic_input())

            # synaptic potentials
            ampa = np.add(ampa, self._change_ampa(ampa=ampa, dt=dt))
            gaba = np.add(gaba, self._change_gaba(gaba=gaba, dt=dt))
            gsyn = np.append(ampa, gaba)

            # defining input to eah neuron as the summation of all synaptic input
            # form all connected neurons
            I = np.add(I, np.matmul(connectivity.K, gsyn))

            self.potential = np.add(self.potential, self._change_v(I=I))
            self.potential = np.add(self.potential, self._change_v(I=I))
            self.recovery = np.add(self.recovery, self._change_u())

        print("Simulation ended")

    def _change_u(self):
        """
        Computes the change in membrane recovery.

        :return: change in membrane recovery.
        :rtype: ndarray[float]
        """

        return np.multiply(
            self.izhi_alpha,
            np.multiply(self.izhi_beta, self.potential) - self.recovery
        )

    def _change_v(self, I):
        """
        Computes the change in membrane potential.

        :param I: current.

        :return: change in membrane potential.
        :rtype: ndarray[float]
        """

        # TODO:: why do we multiply the equation for dv/dt with 0.5 and then call this function twice in run_simulation?
        return 0.5 * (0.04 * self.potential ** 2 + 5 * self.potential + 140 - self.recovery + I)

    def _change_gaba(self, gaba, dt):
        """
        Computes the change in synaptic gates for excitatory postsynaptic neurons.

        :param ampa:
        :param dt: TODO

        :return: change in synaptic gates for excitatory postsynaptic neurons.
        :rtype: ndarray[float]
        """

        # TODO:: in the paper, this computation is different
        alpha = self.potential[self.nr_excit:] / 10.0 + 2
        z = np.tanh(alpha)
        comp1 = (z + 1) / 2.0
        comp2 = (1 - gaba) / SYNAPTIC_CONST_RISE[NeuronTypes.I]
        comp3 = gaba / SYNAPTIC_CONST_DECAY[NeuronTypes.I]
        return dt * 0.3 * (np.multiply(comp1, comp2) - comp3)

    def _change_ampa(self, ampa, dt):
        """
        Computes the change in synaptic gates for inhibitory postsynaptic neurons.

        :param gaba: current state of inhibitory synaptic gates.
        :param dt: TODO

        :return: change in synaptic gates for inhibitory postsynaptic neurons.
        :rtype: ndarray[float]
        """

        alpha = self.potential[:self.nr_excit] / 10.0 + 2
        z = np.tanh(alpha)
        comp1 = (z + 1) / 2.0
        comp2 = (1 - ampa) / SYNAPTIC_CONST_RISE[NeuronTypes.E]
        comp3 = ampa / SYNAPTIC_CONST_DECAY[NeuronTypes.E]
        return dt * 0.3 * (np.multiply(comp1, comp2) - comp3)

    def _change_thalamic_input(self):
        """
        Computes the change in thalamic input.

        :return: change in thalamic input.
        :rtype: ndarray[float]
        """

        return np.append(
            GAUSSIAN_INPUT[NeuronTypes.E] * np.random.randn(self.nr_excit),
            GAUSSIAN_INPUT[NeuronTypes.I] * np.random.randn(self.nr_inhibit)
        )

    def _create_main_input_stimulus(self):
        """
        Creates main (external) input stimulus.

        :return: input stimulus.
        :rtype: ndarray[float]
        """

        # sinusoidal spatial modulation of input strength
        amplitude = 1
        # mean input level to RS cells
        mean_input_lvl_RS = 7
        step = 2 * pi / (self.nr_excit - 1)
        stim_input = mean_input_lvl_RS + amplitude * np.sin(
            crange(-pi, pi, step)
        )
        # additional mean input to FS cells
        stim_input = np.append(stim_input, 3.5 * np.ones(self.nr_inhibit))

        return stim_input
