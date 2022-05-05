from src.ParamsSynaptic import *

from src.CurrentComponents import *
from src.Connectivity import *

import numpy as np


class CurrentComponentsSinglePING(CurrentComponents):
    """
    This lass contains methods of computing the current components for the network of a single PING network.

    :param connectivity: information about connectivity between neurons in the oscillatory network.
    :type connectivity: Connectivity

    :param params_synaptic: contains synaptic parameters.
    :type params_synaptic: ParamsSynaptic

    :param mean_ex: mean of input strength to excitatory neurons.
    :type mean_ex: float

    :param var_ex: variance of input strength to excitatory neurons.
    :type var_ex: float

    :param mean_in: mean of input strength to inhibitory neurons.
    :type mean_in: float

    :param var_in: variance of input strength to inhibitory neurons.
    :type var_in: float


    :ivar _params_synaptic: contains synaptic parameters.
    :ivar _synaptic_currents: keeps track of intermediate synaptic currents.
    """

    def __init__(
            self, connectivity: Connectivity,
            params_synaptic: ParamsSynaptic,
            mean_ex: float = 20, var_ex: float = 0, mean_in: float = 4, var_in: float = 0
    ):
        super().__init__(connectivity)

        self._params_synaptic = params_synaptic
        self._synaptic_currents: np.ndarray[int, float] = np.zeros(self.connectivity.params_ping.nr_neurons["total"])

        self._mean_ex = mean_ex
        self._var_ex = var_ex
        self._mean_in = mean_in
        self._var_in = var_in

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

        for nt in [NeuronTypes.E, NeuronTypes.I]:
            transmission_concs = 1 + np.tanh(potentials[self.connectivity.params_ping.neur_slice[nt]] / 10 + 2)
            # ampa or gaba
            curr = self._synaptic_currents[self.connectivity.params_ping.neur_slice[nt]]
            new_currents[self.connectivity.params_ping.neur_slice[nt]] = curr + dt * 0.3 * (
                (transmission_concs / 2) * (1 - curr) / self._params_synaptic.rise[nt] -
                curr / self._params_synaptic.decay[nt]
            )
        self._synaptic_currents = new_currents

        return new_currents

    def get_current_input(
            self
    ) -> np.ndarray[int, float]:
        """
        Computes the input current to each neuron.

        Computes :math:`I_{\mathrm{stim}}`.

        :return: input strength to each neuron.
        :rtype: numpy.ndarray[int, float]
        """

        input_excitatory = [
            self._var_ex * np.random.randn() + self._mean_ex
            for _ in range(self.connectivity.params_ping.nr_neurons[NeuronTypes.E])
        ]
        input_inhibitory = [
            self._var_in * np.random.randn() + self._mean_in
            for _ in range(self.connectivity.params_ping.nr_neurons[NeuronTypes.I])
        ]

        return np.array(input_excitatory + input_inhibitory)
