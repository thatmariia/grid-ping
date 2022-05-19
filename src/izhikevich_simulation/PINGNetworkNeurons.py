from src.params.NeuronTypes import *


class PINGNetworkNeurons:
    """
    Class containing neural locational information about a single PING network oscillator.

    :param grid_location: coordinates of this oscillator within the grid.
    :type grid_location: tuple[int, int]

    :param ids_ex: id's of excitatory neurons located in this oscillator.
    :type ids_ex: list[int]

    :param ids_in: id's of inhibitory neurons located in this oscillator.
    :type ids_in: list[int]


    :ivar grid_location: coordinates of this oscillator within the grid.
    :ivar ids: A dictionary with id's of neurons of both types in this oscillator.
    """

    def __init__(self, grid_location: tuple[int, int], ids_ex: list[int], ids_in: list[int]):
        self.grid_location: tuple[int, int] = grid_location
        self.ids: dict[NeuronTypes, list[int]] = {
            NeuronTypes.EX: ids_ex,
            NeuronTypes.IN: ids_in
        }

    def __eq__(self, other):
        return self.__dict__ == other.__dict__
