from src.GridConnectivity import *
from src.misc import *
from src.constants import *

from tqdm import tqdm
import numpy as np
from math import pi


class OscillatoryNetwork:
    """
    This class runs the simulation of the network of oscillators.

    Parts of the code in this class are rewritten from MATLAB code listed in supplementary
    materials of :cite:p:`Lowet2015`. The neural dynamic model is is introduced in :cite:p:`Izhikevich2003`.

    :param nr_excitatory: number of excitatory neurons in the network.
    :type nr_excitatory: int

    :param nr_inhibitory: number of inhibitory neurons in the network.
    :type nr_inhibitory: int

    :param nr_oscillators: number of oscillators in the network.
    :type nr_oscillators: int


    :raises:
        AssertionError: if the number of excitatory neurons is smaller than 2.
    :raises:
        AssertionError: if the number of inhibitory neurons is smaller than 2.


    :ivar nr_neurons: number of neurons of each type in the network.
    :type nr_neurons: dict[NeuronTypes: int]

    :ivar nr_oscillators: number of oscillators in the network.
    :type nr_oscillators: int

    :ivar synaptic_potentials: synaptic potentials.
    :type synaptic_potentials: ndarray[float]

    :ivar potentials: voltage (membrane potential).
    :type potentials: ndarray[float]

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

    def __init__(self, nr_excitatory, nr_inhibitory, nr_oscillators=1):

        # FIXME:: this assertions are only there because of the stim_input
        assert nr_excitatory >= 2, "Number of excitatory neurons cannot be smaller than 2."
        assert nr_inhibitory >= 2, "Number of inhibitory neurons cannot be smaller than 2."

        self.nr_neurons = {
            NeuronTypes.E: nr_excitatory,
            NeuronTypes.I: nr_inhibitory
        }
        self.nr_oscillators = nr_oscillators

        self.izhi_alpha = np.array(
            [IZHI_ALPHA[NeuronTypes.E] for _ in range(self.nr_neurons[NeuronTypes.E])] +
            [IZHI_ALPHA[NeuronTypes.I] for _ in range(self.nr_neurons[NeuronTypes.I])]
        )
        self.izhi_beta = np.array(
            [IZHI_BETA[NeuronTypes.E] for _ in range(self.nr_neurons[NeuronTypes.E])] +
            [IZHI_BETA[NeuronTypes.I] for _ in range(self.nr_neurons[NeuronTypes.I])]
        )
        self.izhi_gamma = np.array(
            [IZHI_GAMMA[NeuronTypes.E] for _ in range(self.nr_neurons[NeuronTypes.E])] +
            [IZHI_GAMMA[NeuronTypes.I] for _ in range(self.nr_neurons[NeuronTypes.I])]
        )
        self.izhi_zeta = np.array(
            [IZHI_ZETA[NeuronTypes.E] for _ in range(self.nr_neurons[NeuronTypes.E])] +
            [IZHI_ZETA[NeuronTypes.I] for _ in range(self.nr_neurons[NeuronTypes.I])]
        )

        self.synaptic_potentials = {
            NeuronTypes.E: np.zeros(self.nr_neurons[NeuronTypes.E]),
            NeuronTypes.I: np.zeros(self.nr_neurons[NeuronTypes.I])
        }

        self.potentials = np.array(
            [INIT_MEMBRANE_POTENTIAL for _ in range(self.nr_neurons[NeuronTypes.E] + self.nr_neurons[NeuronTypes.I])]
        )
        self.recovery = np.multiply(self.izhi_beta, self.potentials)

    def run_simulation(self, simulation_time, dt):
        """
        Runs the simulation.

        :param simulation_time: number of epochs to run the simulation.
        :param dt: TODO

        :rtype: None
        """

        # spike timings
        firing_times = []

        stim_input = self._create_main_input_stimulus()

        connectivity = GridConnectivity(
            nr_neurons=self.nr_neurons,
            nr_oscillators=self.nr_oscillators
        )

        print("Simulation started")

        for t in tqdm(range(simulation_time)):

            fired = np.argwhere(self.potentials > 30).flatten()  # indices of spikes
            # TODO:: why do we need firing_times?
            firing_times = [firing_times, [t for _ in range(len(fired))] + fired]
            for f in fired:
                self.potentials[f] = self.izhi_gamma[f]
                self.recovery[f] += self.izhi_zeta[f]

            # thalamic input
            current = np.add(stim_input, self._change_thalamic_input())

            # synaptic potentials
            self.synaptic_potentials[NeuronTypes.E] = np.add(
                self.synaptic_potentials[NeuronTypes.E],
                self._change_synaptic_potentials(neuron_type=NeuronTypes.E, dt=dt)
            )
            self.synaptic_potentials[NeuronTypes.I] = np.add(
                self.synaptic_potentials[NeuronTypes.I],
                self._change_synaptic_potentials(neuron_type=NeuronTypes.I, dt=dt)
            )
            gsyn = np.append(self.synaptic_potentials[NeuronTypes.E], self.synaptic_potentials[NeuronTypes.I])

            # defining input to eah neuron as the summation of all synaptic input
            # form all connected neurons
            current = np.add(current, np.matmul(connectivity.coupling_weights, gsyn))

            self.potentials = np.add(self.potentials, self._change_potential(current=current))
            self.potentials = np.add(self.potentials, self._change_potential(current=current))
            self.recovery = np.add(self.recovery, self._change_recovery())

        print("Simulation ended")

    def _change_recovery(self):
        """
        Computes the change in membrane recovery.

        :return: change in membrane recovery.
        :rtype: ndarray[float]
        """

        return np.multiply(
            self.izhi_alpha,
            np.multiply(self.izhi_beta, self.potentials) - self.recovery
        )

    def _change_potential(self, current):
        """
        Computes the change in membrane potentials.

        :param current: current.

        :return: change in membrane potentials.
        :rtype: ndarray[float]
        """

        # TODO:: why do we multiply the equation for dv/dt with 0.5 and then call this function twice in run_simulation?
        return 0.5 * (0.04 * self.potentials ** 2 + 5 * self.potentials + 140 - self.recovery + current)

    def _change_synaptic_potentials(self, neuron_type, dt):
        """
        Computes the change in synaptic gates for postsynaptic neurons.

        :param neuron_type: neuron type
        :type neuron_type: NeuronTypes

        :param dt: TODO

        :return: change in synaptic gates for excitatory postsynaptic neurons.
        :rtype: ndarray[float]
        """

        potentials = self.potentials[
            neur_slice(neuron_type, self.nr_neurons[NeuronTypes.E], self.nr_neurons[NeuronTypes.I])
        ]
        # TODO:: in the paper, this computation is different
        alpha = potentials / 10.0 + 2
        z = np.tanh(alpha)
        comp1 = (z + 1) / 2.0
        comp2 = (1 - self.synaptic_potentials[neuron_type]) / SYNAPTIC_CONST_RISE[neuron_type]
        comp3 = self.synaptic_potentials[neuron_type] / SYNAPTIC_CONST_DECAY[neuron_type]
        return dt * 0.3 * (np.multiply(comp1, comp2) - comp3)

    def _change_thalamic_input(self):
        """
        Computes the change in thalamic input.

        :return: change in thalamic input.
        :rtype: ndarray[float]
        """

        return np.append(
            GAUSSIAN_INPUT[NeuronTypes.E] * np.random.randn(self.nr_neurons[NeuronTypes.E]),
            GAUSSIAN_INPUT[NeuronTypes.I] * np.random.randn(self.nr_neurons[NeuronTypes.I])
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
        step = 2 * pi / (self.nr_neurons[NeuronTypes.E] - 1)
        stim_input = mean_input_lvl_RS + amplitude * np.sin(
            crange(-pi, pi, step)
        )
        # additional mean input to FS cells
        stim_input = np.append(stim_input, 3.5 * np.ones(self.nr_neurons[NeuronTypes.I]))

        return stim_input
