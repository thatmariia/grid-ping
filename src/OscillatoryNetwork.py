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


    :ivar nr_excit: number of inhibitory neurons in the network.
    :type nr_excit: int

    :ivar nr_inhibit: number of inhibitory neurons in the network.
    :type nr_inhibit: int

    :ivar nr_neurons: total number of neurons in the network.
    :type nr_neurons: int

    :ivar nr_oscillators: number of oscillators in the network.
    :type nr_oscillators: int

    :ivar v: voltage (membrane potential).
    :type v: ndarray[float]

    :ivar u: membrane recovery variable.
    :type u: ndarray[float]

    :ivar a: timescale of recovery variable u.
    :type a: ndarray[float]

    :ivar b: sensitivity of u to sub-threshold oscillations of v.
    :type b: ndarray[float]

    :ivar c: membrane voltage after spike (after-spike reset of v).
    :type c: ndarray[float]

    :ivar d: after-spike reset of recovery variable u.
    :type d: ndarray[float]
    """

    def __init__(self, nr_excit, nr_inhibit, nr_oscillators=1):

        self.nr_excit = nr_excit
        self.nr_inhibit = nr_inhibit
        self.nr_neurons = self.nr_excit + self.nr_inhibit
        self.nr_oscillators = nr_oscillators

        self.a = np.array([a_EXCIT for _ in range(self.nr_excit)] + [a_INHIBIT for _ in range(self.nr_inhibit)])
        self.b = np.array([b_EXCIT for _ in range(self.nr_excit)] + [b_INHIBIT for _ in range(self.nr_inhibit)])
        self.c = np.array([c_EXCIT for _ in range(self.nr_excit)] + [c_INHIBIT for _ in range(self.nr_inhibit)])
        self.d = np.array([d_EXCIT for _ in range(self.nr_excit)] + [d_INHIBIT for _ in range(self.nr_inhibit)])

        self.v = np.array([v_INIT for _ in range(self.nr_excit + self.nr_inhibit)])
        self.u = np.multiply(self.b, self.v)

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

            fired = np.argwhere(self.v > 30).flatten()  # indices of spikes
            # TODO:: why do we need firing_times?
            firing_times = [firing_times, [t for _ in range(len(fired))] + fired]
            for f in fired:
                self.v[f] = self.c[f]
                self.u[f] += self.d[f]

            # thalamic input
            I = np.add(stim_input, self._change_thalamic_input())

            # synaptic potentials
            ampa = np.add(ampa, self._change_ampa(ampa=ampa, dt=dt))
            gaba = np.add(gaba, self._change_gaba(gaba=gaba, dt=dt))
            gsyn = np.append(ampa, gaba)

            # defining input to eah neuron as the summation of all synaptic input
            # form all connected neurons
            I = np.add(I, np.matmul(connectivity.K, gsyn))

            self.v = np.add(self.v, self._change_v(I=I))
            self.v = np.add(self.v, self._change_v(I=I))
            self.u = np.add(self.u, self._change_u())

        print("Simulation ended")

    def _change_u(self):
        """
        Computes the change in membrane recovery.

        :return: change in membrane recovery.
        :rtype: ndarray[float]
        """

        return np.multiply(
            self.a,
            np.multiply(self.b, self.v) - self.u
        )

    def _change_v(self, I):
        """
        Computes the change in membrane potential.

        :param I: current.

        :return: change in membrane potential.
        :rtype: ndarray[float]
        """

        # TODO:: why do we multiply the equation for dv/dt with 0.5 and then call this function twice in run_simulation?
        return 0.5 * (0.04 * self.v ** 2 + 5 * self.v + 140 - self.u + I)

    def _change_gaba(self, gaba, dt):
        """
        Computes the change in synaptic gates for excitatory postsynaptic neurons.

        :param ampa:
        :param dt: TODO

        :return: change in synaptic gates for excitatory postsynaptic neurons.
        :rtype: ndarray[float]
        """

        # TODO:: in the paper, this computation is different
        alpha = self.v[self.nr_excit:] / 10.0 + 2
        z = np.tanh(alpha)
        comp1 = (z + 1) / 2.0
        comp2 = (1 - gaba) / RISE_GABA
        comp3 = gaba / DECAY_GABA
        return dt * 0.3 * (np.multiply(comp1, comp2) - comp3)

    def _change_ampa(self, ampa, dt):
        """
        Computes the change in synaptic gates for inhibitory postsynaptic neurons.

        :param gaba: current state of inhibitory synaptic gates.
        :param dt: TODO

        :return: change in synaptic gates for inhibitory postsynaptic neurons.
        :rtype: ndarray[float]
        """

        alpha = self.v[:self.nr_excit] / 10.0 + 2
        z = np.tanh(alpha)
        comp1 = (z + 1) / 2.0
        comp2 = (1 - ampa) / RISE_AMPA
        comp3 = ampa / DECAY_AMPA
        return dt * 0.3 * (np.multiply(comp1, comp2) - comp3)

    def _change_thalamic_input(self):
        """
        Computes the change in thalamic input.

        :return: change in thalamic input.
        :rtype: ndarray[float]
        """

        return np.append(
            GAUSSIAN_EXCIT_INPUT * np.random.randn(self.nr_excit),
            GAUSSIAN_INHIBIT_INPUT * np.random.randn(self.nr_inhibit)
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
