from src.misc import *

import numpy as np
from math import pi, ceil


class GaborLuminanceStimulus:
    """
    This class constructs the Gabor texture stimulus and selects a patch from it.

    TODO:: more elaborate explanation + ref.

    :param dist_scale: how far the circles are from each other.
    :type dist_scale: float

    :param contrast_range: contrast range for the figure.
    :type contrast_range: float

    :param spatial_freq: spatial frequency of the grating (cycles / degree).
    :type spatial_freq: float

    :param diameter: annulus diameter (degree).
    :type diameter: float

    :param side_length: side length (degree) of square stimulus region.
    :type side_length: TODO:: float or int?

    :param grating_res: resolution (number of pixels in a single row) of single grating.
    :type grating_res: int

    :param patch_res: resolution (number of pixels in a single row) of the stimulus patch.
    :type patch_res: int


    :raises:
        AssertionError: if distance between neighbouring annuli is smaller than the diameter of an annulus.
    :raises:
        AssertionError: if the contrast range falls outside of the range :math:`(0, 1]`.


    :ivar _side_length: side length (degree) of square stimulus region.
    :type _side_length: TODO:: float or int?

    :ivar _patch_res: resolution (number of pixels in a single row) of the stimulus patch.
    :type _patch_res: int

    :ivar _stim_res: TODO
    :type _stim_res: TODO

    :ivar stimulus: the luminance matrix of the full stimulus.
    :type stimulus: ndarray[ndarray[float]]

    :ivar stimulus_patch: the luminance matrix of a patch of the stimulus.
    :type stimulus_patch: ndarray[ndarray[float]]
    """

    def __init__(self, dist_scale, contrast_range, spatial_freq, diameter, side_length, grating_res, patch_res):

        assert dist_scale >= 1, "The distance between neighbouring annuli should be at least 1 diameter."
        assert (contrast_range > 0) and (contrast_range <= 1), "The contrast range should fall in range :math:`(0, 1]`."

        self._side_length = side_length
        self._patch_res = patch_res

        # TODO:: figure out what it is and remove it as instance
        self._stim_res = None

        grating = self._get_grating(spatial_freq, diameter, grating_res)
        self.stimulus = self._get_full_stimulus(grating, dist_scale, contrast_range, diameter, grating_res)
        self.stimulus_patch = self._get_stimulus_patch(self.stimulus)

    def plot_stimulus(self, stimulus, filename):
        """
        Plots the binary heatmap of a given stimulus.

        :param stimulus: a luminance matrix to plot.
        :type stimulus: list[list[float]]

        :rtype: None
        """

        path = f"../plots/{filename}.png"
        print("Plotting the _stimulus...")
        plot_binary_heatmap(im=stimulus, path=path)
        print(f"Plotting done, result: {path[3:]} \n")

    def _eccentricity_in_patch(self, point):
        """
        Calculates eccentricity at the given point, that is the distance between that point and the center of stimulus.

        :param point: coordinates of the point (within the stimulus patch).
        :type point: (float, float)

        :return: eccentricity at the point.
        :rtype: float
        """

        patch_start = self._get_start_of_patch()
        point_in_stimulus = (patch_start[0] + point[0], patch_start[1] + point[1])
        stimulus_center = (self.stimulus.shape[0] / 2.0, self.stimulus.shape[1] / 2.0)

        return euclidian_dist_R2(point_in_stimulus, stimulus_center)

    def _get_grating(self, spatial_freq, diameter, grating_res):
        """
        Generates a grating (single annulus) of the maximum contrast.

        :param spatial_freq: spatial frequency of the grating (cycles / degree).
        :type spatial_freq: float

        :param diameter: annulus diameter (degree).
        :type diameter: float

        :param grating_res: resolution (number of pixels in a single row) of single grating.
        :type grating_res: int

        :return: the luminance matrix of the single annulus.
        :rtype: ndarray[ndarray[float]]
        """

        r = np.linspace(
            -diameter / 2,
            diameter / 2,
            num=grating_res,
            endpoint=True
        )
        x, y = np.meshgrid(r, r)
        radius = np.power(np.power(x, 2) + np.power(y, 2), 1 / 2)
        mask = radius <= (diameter / 2)
        grating = np.cos(2 * pi * radius * spatial_freq + pi)
        grating = 0.5 * (np.multiply(grating, mask) + 1)
        return grating

    def _get_full_stimulus(self, grating, dist_scale, contrast_range, diameter, grating_res):
        """
        Generates the whole stimulus.

        :param grating: the luminance matrix of the annulus.
        :type grating: ndarray[ndarray[float]]

        :param dist_scale: how far the circles are from each other.
        :type dist_scale: float

        :param contrast_range: contrast range for the figure.
        :type contrast_range: float

        :param diameter: annulus diameter (degree).
        :type diameter: float

        :param grating_res: resolution (number of pixels in a single row) of single grating.
        :type grating_res: int

        :return: the luminance matrix of a full stimulus.
        :rtype: ndarray[ndarray[float]]
        """

        new_diameter = dist_scale * diameter
        reps = ceil(self._side_length / new_diameter)
        new_res = ceil(grating_res * dist_scale)

        self._stim_res = reps * new_res

        total = grating_res**2

        stim_size = (total // grating_res) * new_res + grating_res
        stimulus = np.zeros((stim_size, stim_size))
        # TODO:: color the whole uncovered area with 0.5
        stimulus[:self._stim_res, :self._stim_res] = 0.5
        # _stimulus = np.ones((self._stim_res, self._stim_res)) * 0.5

        for t in range(total):
            i = t // grating_res
            j = t % grating_res

            # TODO:: use contrast contrast_range for a figure only
            contrast = np.random.uniform() * contrast_range + 0.5 - contrast_range / 2
            lower = 0.5 - contrast / 2   # to keep the mean equals to 0.5
            element = grating * contrast + lower

            low = lambda x: x * new_res
            high = lambda x: low(x) + grating_res
            stimulus[low(i):high(i), low(j):high(j)] = element

        return stimulus

    def _get_start_of_patch(self):
        """
        Determines the starting point (left top) of the stimulus patch.

        :return: left top coordinate of the stimulus patch within the full stimulus.
        :rtype: (int, int)
        """

        # TODO:: select an actually relevant patch
        half_res = ceil((self._stim_res - self._patch_res) / 2)
        start = half_res

        return start, start

    def _get_stimulus_patch(self, stimulus):
        """
        Selects a patch of the stimulus.

        :param stimulus: the full stimulus.
        :type stimulus: ndarray[ndarray[float]]

        :return: the luminance matrix of a patch of the stimulus.
        :rtype: ndarray[ndarray[float]]
        """

        start = self._get_start_of_patch()
        end = (start[0] + self._patch_res, start[1] + self._patch_res)
        stimulus_patch = stimulus[start[0]:end[0], start[1]:end[1]]

        return stimulus_patch
