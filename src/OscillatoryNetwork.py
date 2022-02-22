from src.GridConnectivity import *
from src.misc import *
from src.constants import *

from tqdm import tqdm
import numpy as np
from math import pi


class OscillatoryNetwork:
    """
    This class runs the simulation of the network of OscillatoryNetwork oscillators.

    :raises:
        AssertionError: if the number of excitatory neurons is smaller than 2.
    :raises:
        AssertionError: if the number of inhibitory neurons is smaller than 2.


    :ivar potential: voltage (membrane potential).
    :neuron_type potential: ndarray[float]

    :ivar recovery: membrane recovery variable.
    :neuron_type recovery: ndarray[float]

    :ivar izhi_alpha: timescale of recovery variable u.
    :neuron_type izhi_alpha: ndarray[float]

    :ivar izhi_beta: sensitivity of u to sub-threshold oscillations of v.
    :neuron_type izhi_beta: ndarray[float]

    :ivar izhi_gamma: membrane voltage after spike (after-spike reset of v).
    :neuron_type izhi_gamma: ndarray[float]

    :ivar izhi_zeta: after-spike reset of recovery variable u.
    :neuron_type izhi_zeta: ndarray[float]
    """

    def __init__(self):

        # FIXME:: this assertions are only there because of the stim_input
        assert NR_NEURONS[NeuronTypes.E] >= 2, "Number of excitatory neurons cannot be smaller than 2."
        assert NR_NEURONS[NeuronTypes.I] >= 2, "Number of inhibitory neurons cannot be smaller than 2."

        self.izhi_alpha = np.array(
            [IZHI_ALPHA[NeuronTypes.E] for _ in range(NR_NEURONS[NeuronTypes.E])] +
            [IZHI_ALPHA[NeuronTypes.I] for _ in range(NR_NEURONS[NeuronTypes.I])]
        )
        self.izhi_beta = np.array(
            [IZHI_BETA[NeuronTypes.E] for _ in range(NR_NEURONS[NeuronTypes.E])] +
            [IZHI_BETA[NeuronTypes.I] for _ in range(NR_NEURONS[NeuronTypes.I])]
        )
        self.izhi_gamma = np.array(
            [IZHI_GAMMA[NeuronTypes.E] for _ in range(NR_NEURONS[NeuronTypes.E])] +
            [IZHI_GAMMA[NeuronTypes.I] for _ in range(NR_NEURONS[NeuronTypes.I])]
        )
        self.izhi_zeta = np.array(
            [IZHI_ZETA[NeuronTypes.E] for _ in range(NR_NEURONS[NeuronTypes.E])] +
            [IZHI_ZETA[NeuronTypes.I] for _ in range(NR_NEURONS[NeuronTypes.I])]
        )

        self.potential = np.array(
            [INIT_MEMBRANE_POTENTIAL for _ in range(NR_NEURONS[NeuronTypes.E] + NR_NEURONS[NeuronTypes.I])]
        )
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

        ampa = np.zeros(NR_NEURONS[NeuronTypes.E])
        gaba = np.zeros(NR_NEURONS[NeuronTypes.I])

        stim_input = self._create_main_input_stimulus()

        connectivity = GridConnectivity()

        print("Simulation started")

        for t in tqdm(range(simulation_time)):

            fired = np.argwhere(self.potential > 30).flatten()  # indices of spikes
            # TODO:: why do we need firing_times?
            firing_times = [firing_times, [t for _ in range(len(fired))] + fired]
            for f in fired:
                self.potential[f] = self.izhi_gamma[f]
                self.recovery[f] += self.izhi_zeta[f]

            # thalamic input
            current = np.add(stim_input, self._change_thalamic_input())

            # synaptic potentials
            ampa = np.add(ampa, self._change_synaptic_potentials(
                neuron_type=NeuronTypes.E,
                synaptic_potential=ampa,
                dt=dt
            ))
            gaba = np.add(gaba, self._change_synaptic_potentials(
                neuron_type=NeuronTypes.I,
                synaptic_potential=gaba,
                dt=dt
            ))
            gsyn = np.append(ampa, gaba)

            # defining input to eah neuron as the summation of all synaptic input
            # form all connected neurons
            current = np.add(current, np.matmul(connectivity.K, gsyn))

            self.potential = np.add(self.potential, self._change_potential(current=current))
            self.potential = np.add(self.potential, self._change_potential(current=current))
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
            np.multiply(self.izhi_beta, self.potential) - self.recovery
        )

    def _change_potential(self, current):
        """
        Computes the change in membrane potential.

        :param current: current.

        :return: change in membrane potential.
        :rtype: ndarray[float]
        """

        # TODO:: why do we multiply the equation for dv/dt with 0.5 and then call this function twice in run_simulation?
        return 0.5 * (0.04 * self.potential ** 2 + 5 * self.potential + 140 - self.recovery + current)

    def _change_synaptic_potentials(self, neuron_type, synaptic_potential, dt):
        """
        Computes the change in synaptic gates for postsynaptic neurons.

        :param neuron_type: neuron type
        :type neuron_type: NeuronTypes

        :param synaptic_potential: current synaptic potential
        :type synaptic_potential: ndarray[float]

        :param dt: TODO

        :return: change in synaptic gates for excitatory postsynaptic neurons.
        :rtype: ndarray[float]
        """

        potentials = self.potential[neur_slice(neuron_type)]
        # TODO:: in the paper, this computation is different
        alpha = potentials / 10.0 + 2
        z = np.tanh(alpha)
        comp1 = (z + 1) / 2.0
        comp2 = (1 - synaptic_potential) / SYNAPTIC_CONST_RISE[neuron_type]
        comp3 = synaptic_potential / SYNAPTIC_CONST_DECAY[neuron_type]
        return dt * 0.3 * (np.multiply(comp1, comp2) - comp3)

    def _change_thalamic_input(self):
        """
        Computes the change in thalamic input.

        :return: change in thalamic input.
        :rtype: ndarray[float]
        """

        return np.append(
            GAUSSIAN_INPUT[NeuronTypes.E] * np.random.randn(NR_NEURONS[NeuronTypes.E]),
            GAUSSIAN_INPUT[NeuronTypes.I] * np.random.randn(NR_NEURONS[NeuronTypes.I])
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
        step = 2 * pi / (NR_NEURONS[NeuronTypes.E] - 1)
        stim_input = mean_input_lvl_RS + amplitude * np.sin(
            crange(-pi, pi, step)
        )
        # additional mean input to FS cells
        stim_input = np.append(stim_input, 3.5 * np.ones(NR_NEURONS[NeuronTypes.I]))

        return stim_input
