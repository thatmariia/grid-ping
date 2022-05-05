from src.NeuronTypes import *


class PINGNetworkNeurons:
    """
    Class containing neural locational information about a single PING network oscillator.

    :param grid_location: coordinates of this oscillator within the grid.
    :type grid_location: tuple[int, int]

    :param excit_ids: id's of excitatory neurons located in this oscillator.
    :type excit_ids: list[int]

    :param inhibit_ids: id's of inhibitory neurons located in this oscillator.
    :type inhibit_ids: list[int]


    :ivar grid_location: coordinates of this oscillator within the grid.
    :ivar ids: A dictionary with id's of neurons of both types in this oscillator.
    """

    def __init__(self, grid_location: tuple[int, int], excit_ids: list[int], inhibit_ids: list[int]):
        self.grid_location: tuple[int, int] = grid_location
        self.ids: dict[NeuronTypes, list[int]] = {
            NeuronTypes.EX: excit_ids,
            NeuronTypes.IN: inhibit_ids
        }

    def __eq__(self, other):
        return self.__dict__ == other.__dict__
