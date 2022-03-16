from src.NeuronTypes import *


class PINGNetwork:
    """
    Class containing locational information about a single PING network oscillator.

    :param location: 2D coordinates of this oscillator within the grid.
    :type location: tuple[int]

    :param excit_ids: id's of excitatory neurons located in this oscillator.
    :type excit_ids: list[int]

    :param inhibit_ids: id's of inhibitory neurons located in this oscillator.
    :type inhibit_ids: list[int]

    :ivar location: 2D coordinates of this oscillator within the grid.
    :type location: tuple[int]

    :ivar ids: A dictionary with id's of neurons of both types in this oscillator.
    :type ids: dict[NeuronTypes, list[int]]
    """

    def __init__(self, location, excit_ids, inhibit_ids):
        self.location = location
        self.ids = {
            NeuronTypes.E: excit_ids,
            NeuronTypes.I: inhibit_ids
        }

    def __eq__(self, other):
        return self.__dict__ == other.__dict__