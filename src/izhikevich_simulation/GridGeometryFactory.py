import numpy as np
from itertools import product
from tqdm import tqdm
from src.misc import *

from src.izhikevich_simulation.GridGeometry import *
from src.izhikevich_simulation.PINGNetworkNeurons import *
from src.params.ParamsPING import *


class GridGeometryFactory:
    """
    This class constructs a grid of PING networks and distributes neurons among them.
    """

    def create(self, params_ping: ParamsPING, cortical_distances: np.ndarray[(int, int), float]) -> GridGeometry:
        """
        Goes through the steps to construct a grid of PING networks and distribute neurons among them.

        :param params_ping: parameters describing PING networks and their composition.
        :type params_ping: ParamsPING

        :param cortical_distances: distances between PING networks in the visual cortex.
        :type cortical_distances: numpy.ndarray[(int, int), float]

        :return: information about grid locations of PING networks and neurons located in them.
        :rtype: GridGeometry
        """

        ping_networks, neuron_distances = self._assign_ping_distances(params_ping, cortical_distances)

        grid_geometry = GridGeometry(
            ping_networks=ping_networks,
            neuron_distances=neuron_distances
        )
        return grid_geometry

    def _refresh_neuron_ping_ids(
            self, ping_id: int,
            ping_ids: dict[int, dict[NeuronTypes, list[int]]],
            params_ping: ParamsPING
    ) -> dict[int, dict[NeuronTypes, list[int]]]:
        """
        Adds the neurons to the relevant PING network if they are not already there.

        :param ping_id: ID of the PING network.
        :type ping_id: int

        :param ping_ids: dictionary containing the IDs of the neurons in each PING network.
        :type ping_ids: dict[int, dict[NeuronTypes, list[int]]]

        :param params_ping: parameters describing PING networks and their composition.
        :type params_ping: ParamsPING

        :return: dictionary containing the IDs of the neurons in each PING network.
        :rtype: dict[int, dict[NeuronTypes, list[int]]]
        """

        if ping_id in ping_ids:
            return ping_ids

        ex_ids = list(range(
            ping_id * params_ping.nr_neurons_per_ping[NeuronTypes.EX],
            (ping_id + 1) * params_ping.nr_neurons_per_ping[NeuronTypes.EX]
        ))
        in_ids = list(range(
            params_ping.nr_neurons[NeuronTypes.EX] + ping_id * params_ping.nr_neurons_per_ping[NeuronTypes.IN],
            params_ping.nr_neurons[NeuronTypes.EX] + (ping_id + 1) * params_ping.nr_neurons_per_ping[NeuronTypes.IN]
        ))
        ping_ids[ping_id] = {
            NeuronTypes.EX: ex_ids,
            NeuronTypes.IN: in_ids
        }
        return ping_ids

    def _assign_ping_distances(
            self, params_ping: ParamsPING, cortical_distances: np.ndarray[(int, int), float]
    ) -> tuple[list[PINGNetworkNeurons], np.ndarray[(int, int), float]]:
        """
        Assigns neurons to PING networks and the distances between them in the visual cortex.

        :param params_ping: parameters describing PING networks and their composition.
        :type params_ping: ParamsPING

        :param cortical_distances: distances between PING networks in the visual cortex.
        :type cortical_distances: numpy.ndarray[(int, int), float]

        :return: information about grid locations of PING networks and distances between neurons located in them.
        :rtype: tuple[list[PINGNetworkNeurons], numpy.ndarray[(int, int), float]]
        """

        ping_networks = []
        # neuron_distances = np.zeros((params_ping.nr_neurons["total"], params_ping.nr_neurons["total"]), dtype=float)
        neuron_distances = cortical_distances
        #neuron_distances = np.zeros((params_ping.nr_neurons["total"], params_ping.nr_neurons["total"]), dtype=int)

        ping_ids = {}

        for ping1 in (pbar := tqdm(range(params_ping.nr_ping_networks))):
            pbar.set_description("Computing distances between PING networks")

            loc_ping1 = (ping1 // params_ping.grid_size, ping1 % params_ping.grid_size)

            ping_ids = self._refresh_neuron_ping_ids(ping1, ping_ids, params_ping)
            ex_ids1 = ping_ids[ping1][NeuronTypes.EX]
            in_ids1 = ping_ids[ping1][NeuronTypes.IN]

            ping_network = PINGNetworkNeurons(
                grid_id=ping1,
                grid_location=loc_ping1,
                ids_ex=ex_ids1,
                ids_in=in_ids1
            )
            ping_networks.append(ping_network)

            for ping2 in range(params_ping.nr_ping_networks):

                ping_ids = self._refresh_neuron_ping_ids(ping2, ping_ids, params_ping)
                ex_ids2 = ping_ids[ping2][NeuronTypes.EX]
                in_ids2 = ping_ids[ping2][NeuronTypes.IN]

                # all_id_pairs = cartesian_product(np.array(ex_ids1 + in_ids1), np.array(ex_ids2 + in_ids2))
                # neuron_distances.ravel()[
                #     np.ravel_multi_index(all_id_pairs.T, neuron_distances.shape)
                # ] = cortical_distances[ping1, ping2]

        return ping_networks, neuron_distances
