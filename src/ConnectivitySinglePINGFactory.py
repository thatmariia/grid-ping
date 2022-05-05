from src.ParamsPING import *

from src.NeuronTypes import *
from src.Connectivity import *

from itertools import product
import numpy as np


class ConnectivitySinglePINGFactory:

    def create(
            self, params_ping: ParamsPING,
            max_connect_strength: dict[tuple[NeuronTypes, NeuronTypes], float]
    ) -> Connectivity:
        """
        Determines the connectivity between neurons in the oscillatory network.

        :param params_ping: parameters describing PING networks and their composition.
        :type params_ping: ParamsPING

        :param max_connect_strength: maximum connection strength between types of neurons.
        :type max_connect_strength: dict[tuple[NeuronTypes, NeuronTypes], float]

        :return: connectivity between neurons in the oscillatory network.
        :rtype: Connectivity
        """

        coupling_weights = np.zeros((params_ping.nr_neurons["total"], params_ping.nr_neurons["total"]))

        for nts in list(product([NeuronTypes.E, NeuronTypes.I], repeat=2)):
            types_coupling_weights = max_connect_strength[(nts[1], nts[0])] * \
                                     np.random.rand(params_ping.nr_neurons[nts[0]], params_ping.nr_neurons[nts[1]])
            if nts[1] == NeuronTypes.I:
                types_coupling_weights *= -1
            coupling_weights[params_ping.neur_slice[nts[0]], params_ping.neur_slice[nts[1]]] = types_coupling_weights

        connectivity = Connectivity(
            params_ping=params_ping,
            coupling_weights=np.nan_to_num(coupling_weights)
        )

        return connectivity