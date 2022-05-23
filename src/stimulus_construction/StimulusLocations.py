from src.misc import *

import numpy as np


class StimulusLocations:
    """
    This class contains information about PING locations in the visual cortex.

    :param eccentricities: eccentricities of the points in visual degrees.
    :type eccentricities: numpy.ndarray[int, float]

    :param angles: angles of the points relative to horizontal axis.
    :type angles: numpy.ndarray[int, float]

    :ivar cortical_distances: distances between PING networks in the visual cortex.
    :type cortical_distances: numpy.ndarray[(int, int), float]
    """

    def __init__(
            self, eccentricities: np.ndarray[int, float],
            angles: np.ndarray[int, float]
    ):
        self.cortical_distances: np.ndarray[(int, int), float] = self._compute_distances(eccentricities, angles)

    def _compute_distances(
            self, eccentricities: np.ndarray[int, float], angles: np.ndarray[int, float]
    ) -> np.ndarray[(int, int), float]:
        """
        Computes the cortical coordinates given eccentricities and angles in the visual field.
        TODO:: add refs

        :param eccentricities: eccentricities of the points in visual degrees.
        :type eccentricities: numpy.ndarray[int, float]

        :param angles: angles of the points relative to horizontal axis.
        :type angles: numpy.ndarray[int, float]

        :return: distances between PING networks in the visual cortex.
        :rtype: numpy.ndarray[(int, int), float]
        """

        k = 15.0
        a = 0.7
        b = 80
        alpha = 0.9

        z = eccentricities * np.exp(1j * alpha * angles)
        w = k * np.log((z + a) / (z + b)) - k * np.log(a / b)

        x = np.real(w)
        y = np.imag(w)

        coordinates = [list(zip(x[i], y[i])) for i in range(len(x))]
        coordinates = [i for row_coords in coordinates for i in row_coords]

        distances = [
            [euclidian_dist(tuple(c1), tuple(c2)) for c1 in coordinates]
            for c2 in coordinates
        ]

        return np.array(distances)
