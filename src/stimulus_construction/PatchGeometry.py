from src.misc import *
from src.stimulus_construction.PINGNetworkPixels import *


class PatchGeometry:
    """
    This class contains information about the grid layout of the stimulus patch in correspondence with PING networks.

    :param ping_networks_pixels: list of all PING networks of the stimulus patch created by applying a lattice.
    :type ping_networks_pixels: list[PINGNetworkPixels]

    :param atopix: conversion coefficient between pixels and visual degrees.
    :type atopix: float

    :param patch_start: top left coordinate of the patch.
    :type patch_start: tuple[int, int]

    :param stimulus_center: the center of the full stimulus.
    :type stimulus_center: tuple[float, float]


    :ivar ping_networks_pixels: list of all PING networks of the stimulus patch created by applying a lattice.
    :ivar _atopix: conversion coefficient between pixels and visual degrees.
    :ivar _patch_start: top left coordinate of the patch.
    :ivar _stimulus_center: the center of the full stimulus.
    """

    def __init__(
            self,
            ping_networks_pixels: list[PINGNetworkPixels],
            atopix: float,
            patch_start: tuple[int, int],
            stimulus_center: tuple[float, float]
    ):
        self.ping_networks_pixels: list[PINGNetworkPixels] = ping_networks_pixels

        self._atopix: float = atopix
        self._patch_start: tuple[int, int] = patch_start
        self._stimulus_center: tuple[float, float] = stimulus_center

    def angle_in_patch(
            self, point: tuple[float, float]
    ) -> float:
        """
        Calculates the angle between the horizontal axis and the line passing through the center of the stimulus and a
        given point within the patch.

        :param point: coordinates of the point within the patch in degrees.
        :type point: tuple[float, float]

        :return: angle of the point.
        :rtype: float
        """

        point_in_stimulus = self.point_in_stimulus(point)
        stimulus_center_dg = (self._stimulus_center[0] / self._atopix, 0.5 * self._stimulus_center[1] / self._atopix)

        new_point = add_points([
            point_in_stimulus, stimulus_center_dg
        ], [1, -1])

        angle = np.arctan(
            new_point[1] / new_point[0]
        )
        return angle

    def eccentricity_in_patch(
            self, point: tuple[float, float]
    ) -> float:
        """
        Calculates eccentricity at the given point within the patch.

        :param point: coordinates of the point within the patch in degrees.
        :type point: tuple[float, float]

        :return: eccentricity in degrees.
        :rtype: float
        """

        point_in_stimulus = self.point_in_stimulus(point)
        stimulus_center_dg = (self._stimulus_center[0] / self._atopix, self._stimulus_center[1] / self._atopix)

        ecc = euclidian_dist(
            stimulus_center_dg,
            point_in_stimulus
        )
        return ecc

    def point_in_stimulus(
            self, point: tuple[float, float]
    ) -> tuple[float, ...]:
        """
        Calculates the coordinate of a given point in the patch within the stimulus.

        :param point: coordinates of the point within the patch in degrees.
        :type point: tuple[float, float]

        :return: coordinates of the point within the stimulus in degrees.
        :rtype: tuple[float, float]
        """

        return add_points([
            point,
            (self._patch_start[0] / self._atopix, self._patch_start[1] / self._atopix)
        ])