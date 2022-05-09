from src.NeuronTypes import *

from math import sqrt
from typing import Union

class ParamsPING:
    """
    This class contains parameters describing PING networks and their composition.

    :param nr_excitatory: number of excitatory neurons in the network.
    :type nr_excitatory: int

    :param nr_inhibitory: number of inhibitory neurons in the network.
    :type nr_inhibitory: int

    :param nr_ping_networks: number of PING networks in the network.
    :type nr_ping_networks: int

    :raises:
        AssertionError: If the number of PING networks is smaller than 1.
    :raises:
        AssertionError: if number of excitatory neurons doesn't divide the number of PING networks as there should be
        an equal number of excitatory neurons in each PING network.
    :raises:
        AssertionError: if number of inhibitory neurons doesn't divide the number of PING networks as there should be
        an equal number of inhibitory neurons in each PING network.
    :raises:
        AssertionError: if the number of PING networks is not a square as PING networks should be arranged in a square
        grid.


    :ivar nr_neurons: dictionary of number of neurons of each type and the total number of neurons.
    :ivar neur_slice: indices of each type of neurons.
    :ivar nr_ping_networks: number of PING networks in the network.
    :ivar nr_neurons_per_ping: number of neurons of each type in a single PING network.
    :ivar grid_size: number of PING networks in each row and column.
    """

    def __init__(self, nr_excitatory: int, nr_inhibitory: int, nr_ping_networks: int=1):

        assert nr_ping_networks >= 1, "Number of PING networks cannot be smaller than 1."
        assert nr_excitatory % nr_ping_networks == 0, \
            "Cannot allocated equal number of excitatory neurons to each PING network. Make sure the number of " \
            "PING networks divides the number of excitatory neurons."
        assert nr_inhibitory % nr_ping_networks == 0, \
            "Cannot allocated equal number of inhibitory neurons to each PING network. Make sure the number of " \
            "PING networks divides the number of inhibitory neurons."
        assert int(sqrt(nr_ping_networks)) == sqrt(nr_ping_networks), \
            "The PING networks should be arranged in a square grid. Make sure the number of PING networks is a " \
            "perfect square."

        self.nr_neurons: dict[Union[NeuronTypes, str], int] = {
            NeuronTypes.EX: nr_excitatory,
            NeuronTypes.IN: nr_inhibitory,
            "total": nr_excitatory + nr_inhibitory
        }
        self.neur_slice: dict[NeuronTypes, slice] = {
            NeuronTypes.EX: slice(0, nr_excitatory),
            NeuronTypes.IN: slice(nr_excitatory, nr_excitatory + nr_inhibitory),
        }
        self.nr_ping_networks: int = nr_ping_networks
        self.nr_neurons_per_ping: dict[NeuronTypes, int] = {
            NeuronTypes.EX: nr_excitatory // nr_ping_networks,
            NeuronTypes.IN: nr_inhibitory // nr_ping_networks
        }
        self.grid_size: int = int(sqrt(nr_ping_networks))