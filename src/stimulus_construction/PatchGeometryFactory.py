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

        ping_networks_pixels = self._assign_circuits(nr_ping_networks, stimulus_patch, atopix)

        patch_geometry = PatchGeometry(
            ping_networks_pixels=ping_networks_pixels,
            atopix=atopix,
            patch_start=patch_start,
            stimulus_center=stimulus_center
        )

        return patch_geometry

    def _assign_circuits(
            self, nr_ping_networks: int, stimulus_patch: np.ndarray[(int, int), float], atopix: float
    ) -> list[PINGNetworkPixels]:
        """
        Creates circuits and assigns centers and pixels of the stimulus patch to them.

        :param nr_ping_networks: number of PING networks.
        :type nr_ping_networks: int

        :param stimulus_patch: the luminance matrix of a patch of the stimulus.
        :type stimulus_patch: numpy.ndarray[(int, int), float]

        :param atopix: conversion coefficient between pixels and visual degrees.
        :type atopix: float

        :return: list of all PING networks of the stimulus patch created by applying a lattice.
        :rtype: list[PINGNetworkPixels]
        """

        lattice_edges = np.linspace(
            0,
            np.shape(stimulus_patch)[0],
            num=int(sqrt(nr_ping_networks)) + 1,
            endpoint=True,
            dtype=int
        )

        ping_networks_pixels = []

        for i in range(len(lattice_edges) - 1):
            for j in range(len(lattice_edges) - 1):

                center = add_points([
                    (lattice_edges[i], lattice_edges[j]),
                    ((lattice_edges[i + 1] - lattice_edges[i]) / 2, (lattice_edges[j + 1] - lattice_edges[j]) / 2)
                ])
                pixels = list(product(
                    np.arange(lattice_edges[i], lattice_edges[i + 1]),
                    np.arange(lattice_edges[j], lattice_edges[j + 1])
                ))

                circuit = PINGNetworkPixels(
                    center=center,
                    pixels=pixels,
                    atopix=atopix,
                    grid_location=(i, j)
                )
                ping_networks_pixels.append(circuit)

        return ping_networks_pixels