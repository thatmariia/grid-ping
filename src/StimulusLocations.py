import numpy as np


class StimulusLocations:
    """
    Contains information about PING locations in the visual cortex.

    :param eccentricities: eccentricities of the points in visual degrees.
    :type eccentricities: numpy.ndarray[int, float]

    :param angles: angles of the points relative to horizontal axis.
    :type angles: numpy.ndarray[int, float]

    :ivar cortical_coords: coordinates of the points in the visual cortex.
    :type cortical_coords: list[list[tuple[float, float]]]
    """

    def __init__(self, eccentricities: np.ndarray[int, float], angles: np.ndarray[int, float]):
        self.cortical_coords = self._compute_coordinates(eccentricities, angles)

    def _compute_coordinates(
            self, eccentricities: np.ndarray[int, float], angles: np.ndarray[int, float]
    ) -> list[list[tuple[float, float]]]:
        """
        Computes the cortical coordinates given eccentricities and angles in the visual field.
        TODO:: add refs

        :param eccentricities: eccentricities of the points in visual degrees.
        :type eccentricities: numpy.ndarray[int, float]

        :param angles: angles of the points relative to horizontal axis.
        :type angles: numpy.ndarray[int, float]

        :return: coordinates of the points in the visual cortex.
        :rtype: list[list[tuple[float, float]]]
        """

        k = 15.0
        a = 0.7
        b = 80
        alpha = 0.9

        z = eccentricities * np.exp(1j * alpha * angles)
        w = k * np.log((z + a) / (z + b)) - k * np.log(a / b)

        x = np.real(w)
        y = np.imag(w)

        return [list(zip(x[i], y[i])) for i in range(len(x))]
