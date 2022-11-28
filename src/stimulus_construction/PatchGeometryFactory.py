from src.stimulus_construction.PINGNetworkPixels import *
from src.stimulus_construction.PatchGeometry import *

import numpy as np
from math import sqrt
from itertools import product

class PatchGeometryFactory:
    """
    This class constructs the grid layout of the stimulus patch in correspondence with PING networks by applying the
    lattice.
    """

    def create(
            self,
            nr_ping_networks: int,
            stimulus_patch: np.ndarray[(int, int), float],
            patch_start: tuple[int, int],
            stimulus_center: tuple[float, float],
            atopix: float
    ) -> PatchGeometry:
        """
        Goes through the steps to construct a PING grid out of the stimulus.

        :param nr_ping_networks: number of PING networks.
        :type nr_ping_networks: int

        :param stimulus_patch: the luminance matrix of a patch of the stimulus.
        :type stimulus_patch: numpy.ndarray[(int, int), float]

        :param patch_start: top left coordinate of the patch.
        :type patch_start: tuple[int, int]

        :param stimulus_center: the center of the full stimulus.
        :type stimulus_center: tuple[float, float]

        :param atopix: conversion coefficient between pixels and visual degrees.
        :type atopix: float

        :return: the layout (geometry) of the patch.
        :rtype: PatchGeometry
        """

        r = np.linspace(
            0.0,
            stimulus_patch.shape[0] / atopix,
            num=stimulus_patch.shape[0],
            endpoint=True,
            dtype=float
        )
        all_pixels_x_dg, all_pixels_y_dg = np.meshgrid(r, r)
        all_pixels_x_dg = all_pixels_x_dg.flatten()
        all_pixels_y_dg = all_pixels_y_dg.flatten()

        r = np.linspace(
            0.0,
            stimulus_patch.shape[0] / atopix,
            num=int(sqrt(nr_ping_networks)),
            endpoint=True,
            dtype=float
        )
        centers_x_dg, centers_y_dg = np.meshgrid(r, r)
        centers_x_dg = centers_x_dg.flatten()
        centers_y_dg = centers_y_dg.flatten()

        ping_networks_pixels = []

        for i in range(int(sqrt(nr_ping_networks))):
            for j in range(int(sqrt(nr_ping_networks))):
                center_dg = (centers_x_dg[i * int(sqrt(nr_ping_networks)) + j], centers_y_dg[i * int(sqrt(nr_ping_networks)) + j])

                circuit = PINGNetworkPixels(
                    center_dg=center_dg,
                    grid_location=(i, j)
                )
                ping_networks_pixels.append(circuit)

        patch_geometry = PatchGeometry(
            ping_networks_pixels=ping_networks_pixels,
            nr_ping_networks=nr_ping_networks,
            all_pixels_x_dg=all_pixels_x_dg,
            all_pixels_y_dg=all_pixels_y_dg,
            centers_x_dg=centers_x_dg,
            centers_y_dg=centers_y_dg,
            atopix=atopix,
            patch_start=patch_start,
            stimulus_center=stimulus_center
        )

        return patch_geometry
