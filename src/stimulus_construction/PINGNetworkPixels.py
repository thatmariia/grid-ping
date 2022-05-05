from misc import *

class PINGNetworkPixels:
    """
    This class contains information about a single stimulus patch-related information about a PING network.

    :param center: coordinates of the center of the PING network within the stimulus patch.
    :type center: tuple[float, float]

    :param pixels: list of pixel coordinates that belong to the PING network.
    :type pixels: list[tuple[int, int]]

    :param atopix: conversion coefficient between pixels and visual degrees.
    :type atopix: float

    :param grid_index: index of the PING network in the grid.
    :type grid_index: tuple[int, int]

    :ivar center: coordinates of the center of the PING network within the stimulus patch (in pixels).
    :type center: tuple[float, float]

    :ivar pixels: list of coordinates that belong to the PING network (in pixels).
    :type pixels: list[tuple[int, int]]

    :ivar center: coordinates of the center of the PING network within the stimulus patch (in degrees).
    :type center: tuple[float, float]

    :ivar pixels: list of pixel coordinates that belong to the PING network (in degrees).
    :type pixels: list[tuple[float, float]]

    :ivar grid_index: index of the PING network in the grid.
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
