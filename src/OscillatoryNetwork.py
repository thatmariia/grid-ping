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

        # timescale of recovery variable u
        self.a = np.array([a_EXCIT for _ in range(self.nr_excit)] + [a_INHIBIT for _ in range(self.nr_inhibit)])
        # sensitivity of u to sub-threshold oscillations of v
        self.b = np.array([b_EXCIT for _ in range(self.nr_excit)] + [b_INHIBIT for _ in range(self.nr_inhibit)])
        # membrane voltage after spike (after-spike reset of v)
        self.c = np.array([c_EXCIT for _ in range(self.nr_excit)] + [c_INHIBIT for _ in range(self.nr_inhibit)])
        # after-spike reset of recovery variable u
        self.d = np.array([d_EXCIT for _ in range(self.nr_excit)] + [d_INHIBIT for _ in range(self.nr_inhibit)])

        # initial values of v = voltage (membrane potential)
        self.v = np.array([v_INIT for _ in range(self.nr_excit + self.nr_inhibit)])
        # initial values of u = membrane recovery variable
        self.u = np.multiply(self.b, self.v)

    def run_simulation(self, simulation_time, dt):
        print("Simulation started")

        # spike times
        # self.simulation_time = simulation_time
        # self.dt = dt

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

        for t in tqdm(range(simulation_time)):

            # indices of spikes
            fired = np.argwhere(self.v > 30).flatten()
            firing_times = [firing_times, [t for _ in range(len(fired))] + fired]
            for f in fired:
                self.v[f] = self.c[f]
                self.u[f] += self.d[f]

            # thalamic input
            I = self._get_new_thalamic_input(stim_input=stim_input)

            # synaptic potentials
            ampa = self._get_new_ampa(ampa=ampa, dt=dt)
            gaba = self._get_new_gaba(gaba=gaba, dt=dt)
            gsyn = np.append(ampa, gaba)

            # defining input to eah neuron as the summation of all synaptic input
            # form all connected neurons
            I = np.add(I, np.matmul(connectivity.S, gsyn))

            self.v = np.add(self.v, self._addon_to_v(I=I))
            self.v = np.add(self.v, self._addon_to_v(I=I))
            self.u = np.add(self.u, self._addon_to_u())

        print("Simulation ended")

    def _addon_to_u(self):
        return np.multiply(
            self.a,
            np.multiply(self.b, self.v) - self.u
        )

    def _addon_to_v(self, I):
        return 0.5 * (0.04 * self.v ** 2 + 5 * self.v + 140 - self.u + I)

    def _get_new_gaba(self, gaba, dt):
        alpha = self.v[self.nr_excit:] / 10.0 + 2
        z = np.tanh(alpha)
        comp1 = (z + 1) / 2.0
        comp2 = (1 - gaba) / RISE_GABA
        comp3 = gaba / DECAY_GABA
        new_comp = dt * 0.3 * (np.multiply(comp1, comp2) - comp3)
        return np.add(
            gaba,
            new_comp
        )

    def _get_new_ampa(self, ampa, dt):
        alpha = self.v[:self.nr_excit] / 10.0 + 2
        z = np.tanh(alpha)
        comp1 = (z + 1) / 2.0
        comp2 = (1 - ampa) / RISE_AMPA
        comp3 = ampa / DECAY_AMPA
        new_comp = dt * 0.3 * (np.multiply(comp1, comp2) - comp3)
        return np.add(
            ampa,
            new_comp
        )

    def _get_new_thalamic_input(self, stim_input):
        return np.add(
            stim_input,
            np.append(
                GAUSSIAN_EXCIT_INPUT * np.random.randn(self.nr_excit),
                GAUSSIAN_INHIBIT_INPUT * np.random.randn(self.nr_inhibit)
            )
        )

    def _create_main_input_stimulus(self):
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
