from src.params.ParamsSynaptic import *

from src.izhikevich_simulation.CurrentComponents import *
from src.izhikevich_simulation.Connectivity import *

import numpy as np
from itertools import product
from math import pi


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
        self._params_synaptic: ParamsSynaptic = params_synaptic
        self._stimulus_currents: np.ndarray[int, float] = stimulus_currents
        self._synaptic_currents: np.ndarray[int, float] = np.zeros(self.connectivity.params_ping.nr_neurons["total"])
    #     self._gatings: np.ndarray[int, float] = np.zeros(self.connectivity.params_ping.nr_neurons["total"])
    #
    # def get_synaptic_currents(
    #         self, dt: float, potentials: np.ndarray[int, float]
    # ) -> np.ndarray[int, float]:
    #     """
    #     Computes the new synaptic currents for postsynaptic neurons.
    #
    #     Computes :math:`I_{\mathrm{syn}}`. The approach is derived from :cite:p:`Jensen2005`.
    #
    #     :param dt: time interval
    #     :type dt: float
    #
    #     :param potentials: neurons' membrane potentials.
    #     :type potentials: numpy.ndarray[int, float]
    #
    #     :return: the change in synaptic currents for a unit of time.
    #     :rtype: numpy.ndarray[int, float]
    #     """
    #
    #     new_gatings = self._get_gatings(dt, potentials)
    #     new_currents = np.zeros(self.connectivity.params_ping.nr_neurons["total"])
    #
    #     for postsyn_nt, presyn_nt in list(product([NeuronTypes.EX, NeuronTypes.IN], repeat=2)):
    #
    #         # conductance calculation between neurons (synapse)
    #         conductances = self._params_synaptic.conductance[(presyn_nt, postsyn_nt)] * \
    #                        new_gatings[self.connectivity.params_ping.neur_slice[presyn_nt]]
    #         total_conductance = (1 / self.connectivity.params_ping.nr_neurons[presyn_nt]) * sum(conductances)
    #
    #         # synaptic current calculation of a postsynaptic neuron
    #         new_currents[self.connectivity.params_ping.neur_slice[postsyn_nt]] += \
    #              np.array(total_conductance) * (
    #                     potentials[self.connectivity.params_ping.neur_slice[postsyn_nt]] -
    #                     self._params_synaptic.reversal_potential[presyn_nt]
    #             )
    #
    #     self._gatings = new_gatings
    #
    #     return new_currents
    #
    # def _get_gatings(
    #         self, dt: float, potentials: np.ndarray[int, float]
    # ) -> np.ndarray[int, float]:
    #     """
    #     Computes the gating values for synapses of given types.
    #
    #     :param dt: time interval.
    #     :type dt: float
    #
    #     :param potentials: neurons' membrane potentials.
    #     :type potentials: numpy.ndarray[int, float]
    #
    #     :return: change in synaptic gates for excitatory postsynaptic neurons.
    #     :rtype: numpy.ndarray[int, float]
    #     """
    #
    #     new_gatings = np.zeros(self.connectivity.params_ping.nr_neurons["total"])
    #
    #     for nt in [NeuronTypes.EX, NeuronTypes.IN]:
    #         transmission_concs = 1 + np.tanh(potentials[self.connectivity.params_ping.neur_slice[nt]] / 4)
    #         change_gatings = (
    #                 self._params_synaptic.rise[nt] * transmission_concs *
    #                 (1 - self._gatings[self.connectivity.params_ping.neur_slice[nt]]) -
    #                 (self._gatings[self.connectivity.params_ping.neur_slice[nt]] / self._params_synaptic.decay[nt])
    #         )
    #         new_gatings[self.connectivity.params_ping.neur_slice[nt]] = \
    #             self._gatings[self.connectivity.params_ping.neur_slice[nt]] + dt * change_gatings
    #
    #     return new_gatings

    def get_synaptic_currents(self, dt, potentials) -> np.ndarray[int, float]:
        """
        Computes the new synaptic currents for postsynaptic neurons.

        Computes :math:`I_{\mathrm{syn}}`.

        :param dt: time interval
        :type dt: float

        :param potentials: neurons' membrane potentials.
        :type potentials: numpy.ndarray[int, float]

        :return: the change in synaptic currents for a unit of time.
        :rtype: numpy.ndarray[int, float]
        """

        new_currents = np.zeros(self.connectivity.params_ping.nr_neurons["total"])

        for nt in [NeuronTypes.EX, NeuronTypes.IN]:
            transmission_concs = 1 + np.tanh(potentials[self.connectivity.params_ping.neur_slice[nt]] / 4)
            # ampa or gaba
            curr_gatings = self._synaptic_currents[self.connectivity.params_ping.neur_slice[nt]]
            change_gatings = (
                    self._params_synaptic.rise[nt] * transmission_concs *
                    (1 - curr_gatings) -
                    (curr_gatings / self._params_synaptic.decay[nt])
            )
            new_currents[self.connectivity.params_ping.neur_slice[nt]] = curr_gatings + dt * change_gatings
        self._synaptic_currents = new_currents

        return new_currents

    def get_current_input(self) -> np.ndarray[int, float]:
        """
        Computes the input current to each neuron.

        Computes :math:`I_{\mathrm{stim}}`.

        :return: input current to each neuron.
        :rtype: numpy.ndarray[int, float]
        """

        current = self._get_thalamic_input() + self._get_stimulus_input()

        return current

    def _get_thalamic_input(self) -> np.ndarray[int, float]:
        """
        Generates the thalamic input.

        :return: thalamic input.
        :rtype: numpy.ndarray[int, float]
        """

        #1.5
        return np.append(
            1.0 * np.random.randn(self.connectivity.params_ping.nr_neurons[NeuronTypes.EX]),
            1.0 * np.random.randn(self.connectivity.params_ping.nr_neurons[NeuronTypes.IN])
        )

    def _get_stimulus_input(self) -> np.ndarray[int, float]:
        """
        Distributes the currents from stimulus to corresponding neurons.

        Creates initial :math:`I_{stim}`.

        :return: input from stimulus.
        :rtype: numpy.ndarray[int, float]
        """

        ####

        # # sinusoidal spatial modulation of input strength
        # amplitude = 1
        # # mean input level to RS cells
        # mean_input_lvl_RS = 7
        # step = 2 * pi / (self.connectivity.params_ping.nr_neurons[NeuronTypes.EX] - 1)
        # stim_input = mean_input_lvl_RS + amplitude * np.sin(
        #     crange(-pi, pi, step)
        # )
        # # additional mean input to FS cells
        # stim_input = np.append(stim_input, 3.5 * np.ones(self.connectivity.params_ping.nr_neurons[NeuronTypes.IN]))
        # return stim_input

        ####

        stimulus_input = np.zeros(self.connectivity.params_ping.nr_neurons["total"])
        currents_grid = self._stimulus_currents.reshape(
            self.connectivity.params_ping.grid_size, self.connectivity.params_ping.grid_size
        )

        for ping_network in self.connectivity.grid_geometry.ping_networks:
            i = ping_network.grid_location[0]
            j = ping_network.grid_location[1]
            stimulus_input[ping_network.ids[NeuronTypes.EX]] = currents_grid[i, j]

            # random number from gaussian distribution with mean 10 and standard deviation 5
            #stimulus_input[ping_network.ids[NeuronTypes.EX]] = np.random.normal(10, 0.5)

        return stimulus_input


def cust_range(*args, rtol=1e-05, atol=1e-08, include=[True, False]):
    """
    Combines numpy.arange and numpy.isclose to mimic
    open, half-open and closed intervals.
    Avoids also floating point rounding errors as with
    >>> numpy.arange(1, 1.3, 0.1)
    array([1. , 1.1, 1.2, 1.3])
    args: [start, ]stop, [step, ]
        as in numpy.arange
    rtol, atol: floats
        floating point tolerance as in numpy.isclose
    include: boolean list-like, length 2
        if start and end point are included
    """
    # process arguments
    if len(args) == 1:
        start = 0
        stop = args[0]
        step = 1
    elif len(args) == 2:
        start, stop = args
        step = 1
    else:
        assert len(args) == 3
        start, stop, step = tuple(args)

    # determine number of segments
    n = (stop-start)/step + 1

    # do rounding for n
    if np.isclose(n, np.round(n), rtol=rtol, atol=atol):
        n = np.round(n)

    # correct for start/end is exluded
    if not include[0]:
        n -= 1
        start += step
    if not include[1]:
        n -= 1
        stop -= step

    return np.linspace(start, stop, int(n))

def crange(*args, **kwargs):
    return cust_range(*args, **kwargs, include=[True, True])