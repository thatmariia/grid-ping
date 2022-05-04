from src.constants import *
from src.CurrentComponents import *
from src.NeuronTypes import *
from src.Connectivity import *

import numpy as np
from itertools import product


class CurrentComponentsGridPING(CurrentComponents):
    """
    This lass contains methods of computing the current components for the network of PING networks arranged in a grid.

    :param connectivity: information about connectivity between neurons in the oscillatory network.
    :type connectivity: Connectivity

    :param stimulus_currents: currents stimulus.
    :type stimulus_currents: numpy.ndarray[int, float]


    :ivar _stimulus_currents: currents stimulus.
    """

    def __init__(self, connectivity: Connectivity, stimulus_currents: np.ndarray[int, float]):
        super().__init__(connectivity)
        self._stimulus_currents: np.ndarray[int, float] = stimulus_currents

    def get_synaptic_currents(
            self, gatings: np.ndarray[(int, int), float], dt: float, potentials: np.ndarray[int, float]
    ) -> tuple[np.ndarray[int, float], np.ndarray[int, float]]:
        """
        Computes the new synaptic currents for postsynaptic neurons.

        Computes the :math:`I_{syn}`. The approach is derived from :cite:p:`Jensen2005`.

        :param gatings: synaptic gating values
        :type gatings: numpy.ndarray[int, float]

        :param dt: time interval
        :type dt: float

        :param potentials: neurons' membrane potentials.
        :type potentials: numpy.ndarray[int, float]

        :return: change in synaptic gates for excitatory postsynaptic neurons.
        :rtype: numpy.ndarray[(int, int), float]
        """

        new_gatings = self._get_gatings(gatings, dt, potentials)
        new_currents = np.zeros(self.connectivity.nr_neurons["total"])

        for postsyn_nt, presyn_nt in list(product([NeuronTypes.E, NeuronTypes.I], repeat=2)):

            # conductance calculation between neurons (synapse)
            conductances = SYNAPTIC_CONDUCTANCE[
                               (presyn_nt, postsyn_nt)] * new_gatings[self.connectivity.neur_slice[presyn_nt]
            ]

            # synaptic current calculation of a postsynaptic neuron
            new_currents[self.connectivity.neur_slice[postsyn_nt]] += \
                sum(conductances) * (
                        potentials[self.connectivity.neur_slice[postsyn_nt]] - REVERSAL_POTENTIAL[presyn_nt]
                )

        return new_currents, new_gatings

    def _get_gatings(
            self, gatings: np.ndarray[int, float], dt: float, potentials: np.ndarray[int, float]
    ) -> np.ndarray[int, float]:
        """
        Computes the gating values for synapses of given types.

        :param gatings: current gating values.
        :type gatings: numpy.ndarray[int, float]

        :param dt: time interval.
        :type dt: float

        :param potentials: neurons' membrane potentials.
        :type potentials: numpy.ndarray[int, float]

        :return: the change in synaptic values for a unit of time.
        :rtype: numpy.ndarray[int, float]
        """

        new_gatings = np.zeros(self.connectivity.nr_neurons["total"])

        for nt in [NeuronTypes.E, NeuronTypes.I]:
            transmission_concs = 1 + np.tanh(potentials[self.connectivity.neur_slice[nt]] / 4)
            change_gatings = (
                    SYNAPTIC_RISE[nt] * transmission_concs * (1 - gatings[self.connectivity.neur_slice[nt]]) -
                    gatings[self.connectivity.neur_slice[nt]] / SYNAPTIC_DECAY[nt]
            )
            new_gatings[self.connectivity.neur_slice[nt]] = \
                gatings[self.connectivity.neur_slice[nt]] + dt * change_gatings

        return new_gatings

    def get_current_input(self) -> np.ndarray[int, float]:
        """
        Computes the input current to each neuron.

        :return: input current to each neuron.
        :rtype: numpy.ndarray[int, float]
        """

        current = self._get_thalamic_input() + self._create_main_input_stimulus(self._stimulus_currents)

        return current

    def _get_thalamic_input(self) -> np.ndarray[int, float]:
        """
        Generates the thalamic input.

        :return: thalamic input.
        :rtype: numpy.ndarray[int, float]
        """

        # TODO:: what is this exactly?
        return np.append(
            GAUSSIAN_INPUT[NeuronTypes.E] * np.random.randn(self.connectivity.nr_neurons[NeuronTypes.E]),
            GAUSSIAN_INPUT[NeuronTypes.I] * np.random.randn(self.connectivity.nr_neurons[NeuronTypes.I])
        )

    def _create_main_input_stimulus(self, stimulus_currents) -> list[float]:
        """
        Parses external input stimulus. ARTIFICIAL FUNCTION - REAL NOT IMPLEMENTED YET.

        Creates initial :math:`I_{stim}`.

        :return: input stimulus.
        :rtype: list[float]
        """

        # TODO:: implement the real strategy
        # TODO:: change 4 to nr ping networks

        stim_input = []
        for i in stimulus_currents:
            stim_input += [i] * \
                          ((self.connectivity.nr_neurons[NeuronTypes.E] +
                            self.connectivity.nr_neurons[NeuronTypes.I]) // self.connectivity.nr_ping_networks)

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