from misc import *

class StimulusCircuit:
    """
    Class containing information about a single _stimulus patch circuit.

    :param center: coordinates of the center of the circuit within the _stimulus patch.
    :type center: tuple[float, float]

    :param pixels: list of pixel coordinates that belong to the circuit.
    :type pixels: list[tuple[int, int]]

    :param atopix: conversion coefficient between pixels and visual degrees.
    :type atopix: float

    :param grid_index: index of the circuit in the grid.
    :type grid_index: tuple[int, int]

    :ivar center: coordinates of the center of the circuit within the stimulus patch (in pixels).
    :type center: tuple[float, float]

    :ivar pixels: list of coordinates that belong to the circuit (in pixels).
    :type pixels: list[tuple[int, int]]

    :ivar center: coordinates of the center of the circuit within the stimulus patch (in degrees).
    :type center: tuple[float, float]

    :ivar pixels: list of pixel coordinates that belong to the circuit (in degrees).
    :type pixels: list[tuple[float, float]]

    :ivar grid_index: index of the circuit in the grid.
    :type grid_index: tuple[int, int]
    """

    def __init__(
            self,
            center: tuple[float, float], pixels: list[tuple[int, int]], atopix: float, grid_index: tuple[int, int]
    ):
        self.center = center
        self.pixels = pixels

        self.center_dg = multiply_point(center, 1 / atopix)
        self.pixels_dg = [multiply_point(p, 1 / atopix) for p in pixels]

        self.grid_index = grid_index
