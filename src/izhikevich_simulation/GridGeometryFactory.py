import numpy as np
from itertools import product

from src.izhikevich_simulation.GridGeometry import *
from src.izhikevich_simulation.PINGNetworkNeurons import *
from src.params.ParamsPING import *


class GridGeometryFactory:
    """
    This class constructs a grid of PING networks and distributes neurons among them.
    """

    def create(self, params_ping: ParamsPING) -> GridGeometry:
        """
        Goes through the steps to construct a grid of PING networks and distribute neurons among them.

        :param params_ping: parameters describing PING networks and their composition.
        :type params_ping: ParamsPING

        :return: information about grid locations of PING networks and neurons located in them.
        :rtype: GridGeometry
        """

        ping_networks, neuron_locations = self._assign_ping_networks(params_ping)

        grid_geometry = GridGeometry(
            ping_networks=ping_networks,
            neuron_locations=neuron_locations
        )

        return grid_geometry

    def _assign_ping_networks(
            self, params_ping: ParamsPING,
    ) -> tuple[list[PINGNetworkNeurons], np.ndarray[int, int]]:
        """
        Creates PING networks, assigns grid locations to them, and adds the same number of neurons of each neuron type
        to them.

        In other words, this function creates a map that can be used as function :math:`\mathsf{loc}`.

        :param params_ping: parameters describing PING networks and their composition.
        :type params_ping: ParamsPING

        TODO: update ping map or delete it for good

        :return: list of PING networks in the network and a dictionary mapping a neuron to the PING network it belongs to.
        :rtype: tuple[list[PINGNetworkNeurons], dict[NeuronTypes, dict[int, int]]]
        """

        ping_networks = []
        neuron_locations = np.zeros(params_ping.nr_neurons["total"], dtype=int)

        for i in range(params_ping.nr_ping_networks):
            x = i // params_ping.grid_size
            y = i % params_ping.grid_size

            ex_ids = []
            for neuron_id in range(
                    i * params_ping.nr_neurons_per_ping[NeuronTypes.EX],
                    (i + 1) * params_ping.nr_neurons_per_ping[NeuronTypes.EX]
            ):
                ex_ids.append(neuron_id)
                neuron_locations[neuron_id] = i

            in_ids = []
            for neuron_id in range(
                    i * params_ping.nr_neurons_per_ping[NeuronTypes.IN],
                    (i + 1) * params_ping.nr_neurons_per_ping[NeuronTypes.IN]
            ):
                in_ids.append(params_ping.nr_neurons[NeuronTypes.EX] + neuron_id)
                neuron_locations[params_ping.nr_neurons[NeuronTypes.EX] + neuron_id] = i


            ping_network = PINGNetworkNeurons(
                grid_location=(x, y),
                ids_ex=ex_ids,
                ids_in=in_ids
            )

            ping_networks.append(ping_network)

        return ping_networks, neuron_locations