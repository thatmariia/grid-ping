from src.stimulus_construction.StimulusLocations import *
from src.stimulus_construction.PatchGeometry import *

import numpy as np
from math import sqrt
from tqdm import tqdm


class Stimulus:
    """
    This class contains information about the external stimulus for the usage in a neural network.

    :param stimulus_currents: currents stimulus.
    :type stimulus_currents: numpy.ndarray[int, float]

    :param patch_geometry: information about the grid layout of the stimulus patch in correspondence with PING networks.
    :type patch_geometry: PatchGeometry

    :ivar stimulus_currents: currents stimulus.
    :ivar _patch_geometry: information about the grid layout of the stimulus patch in correspondence with PING networks.
    """

    def __init__(self, stimulus_currents: np.ndarray[int, float], patch_geometry: PatchGeometry):
        self.stimulus_currents: np.ndarray[int, float] = stimulus_currents
        self._patch_geometry: PatchGeometry = patch_geometry

    def extract_stimulus_location(self) -> StimulusLocations:
        """
        Computes the grid_location info of the stimulus patch and, thus, the PING networks, namely eccentricity and
        angle of each PING network.

        :return: grid_location info of the network.
        :rtype: StimulusLocations
        """

        grid_side = int(sqrt(self._patch_geometry.nr_ping_networks))
        eccentricities = np.zeros((grid_side, grid_side))
        angles = np.zeros((grid_side, grid_side))

        for circuit in (pbar := tqdm(self._patch_geometry.ping_networks_pixels)):
            pbar.set_description("Coordinates conversion")

            i = circuit.grid_location[0]
            j = circuit.grid_location[1]

            eccentricities[i, j] = self._patch_geometry.eccentricity_in_patch(point=circuit.center_dg)
            angles[i, j] = self._patch_geometry.angle_in_patch(point=circuit.center_dg)

        stim_locations = StimulusLocations(
            eccentricities=eccentricities,
            angles=angles
        )

        return stim_locations