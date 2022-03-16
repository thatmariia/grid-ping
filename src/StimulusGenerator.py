from misc import *

import numpy as np
from math import pi, ceil

class StimulusGenerator:
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


    :ivar dist_scale: how far the circles are from each other.
    :type dist_scale: float

    :ivar contrast_range: contrast range for the figure.
    :type contrast_range: float

    :ivar spatial_freq: spatial frequency of the grating (cycles / degree).
    :type spatial_freq: float

    :ivar diameter: annulus diameter (degree).
    :type diameter: float

    :ivar side_length: side length (degree) of square stimulus region.
    :type side_length: TODO:: float or int?

    :ivar grating_res: resolution (number of pixels in a single row) of single grating.
    :type grating_res: int

    :ivar patch_res: resolution (number of pixels in a single row) of the stimulus patch.
    :type patch_res: int

    :ivar stim_res: TODO
    :type stim_res: TODO
    """

    def __init__(self, dist_scale, contrast_range, spatial_freq, diameter, side_length, grating_res, patch_res):

        assert dist_scale >= 1, "The distance between neighbouring annuli should be at least 1 diameter."
        assert (contrast_range > 0) and (contrast_range <= 1), "The contrast range should fall in range :math:`(0, 1]`."

        self.dist_scale = dist_scale
        self.contrast_range = contrast_range
        self.spatial_freq = spatial_freq
        self.diameter = diameter
        self.side_length = side_length
        self.grating_res = grating_res
        self.patch_res = patch_res

        self.stim_res = None

    def generate(self):
        """
        Goes through all the steps (generating grating, full stimulus, patch) to generate the stimulus patch.

        :return: the luminance matrix of the stimulus patch.
        :rtype: ndarray[ndarray[float]]
        """

        grating = self._get_grating()
        stimulus = self._get_full_stimulus(grating=grating)
        stimulus = self._get_stimulus_patch(stimulus=stimulus)
        return stimulus

    def _get_grating(self):
        """
        Generates a grating (single annulus) of the maximum contrast.

        :return: the luminance matrix of the single annulus.
        :rtype: ndarray[ndarray[float]]
        """

        r = np.linspace(
            -self.diameter / 2,
            self.diameter / 2,
            num=self.grating_res,
            endpoint=True
        )
        x, y = np.meshgrid(r, r)
        radius = np.power(np.power(x, 2) + np.power(y, 2), 1 / 2)
        mask = radius <= (self.diameter / 2)
        grating = np.cos(2 * pi * radius * self.spatial_freq + pi)
        grating = 0.5 * (np.multiply(grating, mask) + 1)
        return grating

    def _get_full_stimulus(self, grating):
        """
        Generates the whole stimulus.

        :return: the luminance matrix of a full stimulus.
        :rtype: ndarray[ndarray[float]]
        """

        new_diameter = self.dist_scale * self.diameter
        reps = ceil(self.side_length / new_diameter)
        new_res = ceil(self.grating_res * self.dist_scale)

        self.stim_res = reps * new_res

        total = self.grating_res**2

        stim_size = (total // self.grating_res) * new_res + self.grating_res
        stimulus = np.zeros((stim_size, stim_size))
        # TODO:: color the whole uncovered area with 0.5
        stimulus[:self.stim_res, :self.stim_res] = 0.5
        # stimulus = np.ones((self.stim_res, self.stim_res)) * 0.5

        for t in range(total):
            i = t // self.grating_res
            j = t % self.grating_res

            # TODO:: use contrast contrast_range for a figure only
            contrast = np.random.uniform() * self.contrast_range + 0.5 - self.contrast_range / 2
            lower = 0.5 - contrast / 2   # to keep the mean equals to 0.5
            element = grating * contrast + lower

            low = lambda x: x * new_res
            high = lambda x: low(x) + self.grating_res
            stimulus[low(i):high(i), low(j):high(j)] = element

        return stimulus

    def _get_stimulus_patch(self, stimulus):
        """
        Selects a patch of the stimulus.

        :param stimulus: the full stimulus.
        :type stimulus: ndarray[ndarray[float]]

        :return: the luminance matrix of a patch of the stimulus.
        :rtype: ndarray[ndarray[float]]
        """

        # TODO:: select an actually relevant patch
        half_res = ceil((self.stim_res - self.patch_res) / 2)

        low = half_res
        high = low + self.patch_res
        stimulus = stimulus[low:high, low:high]

        print("Plotting the stimulus patch")
        plot_binary_heatmap(im=stimulus, path="../plots/stimulus-patch.png")

        return stimulus#.flatten()