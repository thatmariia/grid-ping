from src.params.ParamsSynaptic import *

from src.izhikevich_simulation.CurrentComponents import *
from src.izhikevich_simulation.Connectivity import *

import numpy as np
from itertools import product


class CurrentComponentsGridPING(CurrentComponents):
    """
    This lass contains methods of computing the current components for the network of PING networks arranged in a grid.

    :param connectivity: information about connectivity between neurons in the oscillatory network.
    :type connectivity: Connectivity

    :param params_synaptic: contains synaptic parameters.
    :type params_synaptic: ParamsSynaptic

    :param stimulus_currents: currents stimulus.
    :type stimulus_currents: numpy.ndarray[int, float]

    :ivar _params_synaptic: contains synaptic parameters.
    :ivar _stimulus_currents: currents from stimulus.
    :ivar _gatings: keeps track of gating values.
    """

    def __init__(
            self, connectivity: Connectivity, params_synaptic: ParamsSynaptic, stimulus_currents: np.ndarray[int, float]
    ):
        super().__init__(connectivity)
        self._params_synaptic = params_synaptic
        self._stimulus_currents: np.ndarray[int, float] = stimulus_currents
        self._gatings: np.ndarray[int, float] = np.zeros(self.connectivity.params_ping.nr_neurons["total"])

    def get_synaptic_currents(
            self, dt: float, potentials: np.ndarray[int, float]
    ) -> np.ndarray[int, float]:
        """
        Computes the new synaptic currents for postsynaptic neurons.

        Computes :math:`I_{\mathrm{syn}}`. The approach is derived from :cite:p:`Jensen2005`.

        :param dt: time interval
        :type dt: float

        :param potentials: neurons' membrane potentials.
        :type potentials: numpy.ndarray[int, float]

        :return: the change in synaptic currents for a unit of time.
        :rtype: numpy.ndarray[int, float]
        """

        new_gatings = self._get_gatings(dt, potentials)
        new_currents = np.zeros(self.connectivity.params_ping.nr_neurons["total"])

        for postsyn_nt, presyn_nt in list(product([NeuronTypes.EX, NeuronTypes.IN], repeat=2)):

            # conductance calculation between neurons (synapse)
            conductances = self._params_synaptic.conductance[(presyn_nt, postsyn_nt)] * \
                           new_gatings[self.connectivity.params_ping.neur_slice[presyn_nt]]

            # synaptic current calculation of a postsynaptic neuron
            new_currents[self.connectivity.params_ping.neur_slice[postsyn_nt]] += \
                sum(conductances) * (
                        potentials[self.connectivity.params_ping.neur_slice[postsyn_nt]] -
                        self._params_synaptic.reversal_potential[presyn_nt]
                )

        self._gatings = new_gatings

        return new_currents

    def _get_gatings(
            self, dt: float, potentials: np.ndarray[int, float]
    ) -> np.ndarray[int, float]:
        """
        Computes the gating values for synapses of given types.

        :param dt: time interval.
        :type dt: float

        :param potentials: neurons' membrane potentials.
        :type potentials: numpy.ndarray[int, float]

        :return: change in synaptic gates for excitatory postsynaptic neurons.
        :rtype: numpy.ndarray[int, float]
        """

        new_gatings = np.zeros(self.connectivity.params_ping.nr_neurons["total"])

        for nt in [NeuronTypes.EX, NeuronTypes.IN]:
            transmission_concs = 1 + np.tanh(potentials[self.connectivity.params_ping.neur_slice[nt]] / 4)
            change_gatings = (
                    self._params_synaptic.rise[nt] * transmission_concs *
                    (1 - self._gatings[self.connectivity.params_ping.neur_slice[nt]]) -
                    self._gatings[self.connectivity.params_ping.neur_slice[nt]] / self._params_synaptic.decay[nt]
            )
            new_gatings[self.connectivity.params_ping.neur_slice[nt]] = \
                self._gatings[self.connectivity.params_ping.neur_slice[nt]] + dt * change_gatings

        return new_gatings

    def get_current_input(self) -> np.ndarray[int, float]:
        """
        Computes the input current to each neuron.

        Computes :math:`I_{\mathrm{stim}}`.

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
            1.5 * np.random.randn(self.connectivity.params_ping.nr_neurons[NeuronTypes.EX]),
            1.5 * np.random.randn(self.connectivity.params_ping.nr_neurons[NeuronTypes.IN])
        )

    def _create_main_input_stimulus(self, stimulus_currents) -> list[float]:
        """
        Parses external input stimulus. ARTIFICIAL FUNCTION - REAL NOT IMPLEMENTED YET.

        Creates initial :math:`I_{stim}`.

        :return: input stimulus.
        :rtype: list[float]
        """

        # TODO:: implement the real strategy

        stim_input = []
        for i in stimulus_currents:
            stim_input += [i] * \
                          (self.connectivity.params_ping.nr_neurons["total"]
                           // self.connectivity.params_ping.nr_ping_networks)

        return stim_input

        # old code:
        # # sinusoidal spatial modulation of input strength
        # amplitude = 1
        # # mean input level to RS cells
        # mean_input_lvl_RS = 7
        # step = 2 * pi / (self._nr_neurons[NeuronTypes.EX] - 1)
        # stim_input = mean_input_lvl_RS + amplitude * np.sin(
        #     crange(-pi, pi, step)
        # )
        # # additional mean input to FS cells
        # stim_input = np.append(stim_input, 3.5 * np.ones(self._nr_neurons[NeuronTypes.IN]))
        #
        # print("stim input\n", stim_input)
        #
        # return stim_input