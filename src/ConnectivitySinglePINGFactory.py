from src.NeuronTypes import *
from src.Connectivity import *

from itertools import product
import numpy as np


class ConnectivitySinglePINGFactory:

    def create(
            self, nr_excitatory: int, nr_inhibitory: int,
            max_connect_strength: dict[tuple[NeuronTypes, NeuronTypes], float]
    ) -> Connectivity:
        """
        Determines the connectivity between neurons in the oscillatory network.

        :param nr_excitatory: number of excitatory neurons in the network.
        :type nr_excitatory: int

        :param nr_inhibitory: number of inhibitory neurons in the network.
        :type nr_inhibitory: int

        :param max_connect_strength: maximum connection strength between types of neurons.
        :type max_connect_strength: dict[tuple[NeuronTypes, NeuronTypes], float]

        :return: connectivity between neurons in the oscillatory network.
        :rtype: Connectivity
        """

        nr_neurons = {
            NeuronTypes.E: nr_excitatory,
            NeuronTypes.I: nr_inhibitory,
            "total": nr_excitatory + nr_inhibitory
        }
        neur_slice = {
            NeuronTypes.E: slice(nr_excitatory),
            NeuronTypes.I: slice(nr_excitatory, nr_excitatory + nr_inhibitory),
        }

        coupling_weights = np.zeros((nr_neurons["total"], nr_neurons["total"]))

        for nts in list(product([NeuronTypes.E, NeuronTypes.I], repeat=2)):
            types_coupling_weights = max_connect_strength[(nts[1], nts[0])] * \
                                     np.random.rand(nr_neurons[nts[0]], nr_neurons[nts[1]])
            if nts[1] == NeuronTypes.I:
                types_coupling_weights *= -1
            coupling_weights[neur_slice[nts[0]], neur_slice[nts[1]]] = types_coupling_weights

        connectivity = Connectivity(
            nr_neurons=nr_neurons,
            neur_slice=neur_slice,
            nr_ping_networks=1,
            coupling_weights=np.nan_to_num(coupling_weights)
        )

        return connectivity