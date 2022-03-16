from misc import *

import numpy as np
from math import pi, ceil

class StimulusGenerator:
    """
    TODO

    :param scaling: how far the circles are from each other
    :param range: ??
    :param omega: spatial frequency (cycles / degree)
    :param diameter: annulus diameter (degree)
    :param side_length: side length (degree) of square stimulus region
    :param grating_res: resolution of single grating
    :param contrast_res: resolution of contrast image
    """

    def __init__(self, scaling, range, omega, diameter, side_length, grating_res, contrast_res):

        self.scaling = scaling
        self.range = range
        self.omega = omega
        self.diameter = diameter
        self.side_length = side_length
        self.grating_res = grating_res
        self.contrast_res = contrast_res

        self.stim_res = None

    def generate(self):
        """
        TODO
        """
        grating = self._get_grating()
        stimulus = self._get_full_stimulus(grating=grating)
        stimulus = self._get_stimulus_patch(stimulus=stimulus)
        return stimulus

    def _get_grating(self):
        """
        Generates a single grating (annulus) of the maximum contrast.
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
        grating = np.cos(2 * pi * radius * self.omega + pi)
        grating = 0.5 * (np.multiply(grating, mask) + 1)
        return grating

    def _get_full_stimulus(self, grating):
        """
        Generates the whole stimulus.
        """

        new_diameter = self.scaling * self.diameter
        reps = ceil(self.side_length / new_diameter)
        new_res = ceil(self.grating_res * self.scaling)

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

            # TODO:: use contrast range for a figure only
            contrast = np.random.uniform() * self.range + 0.5 - self.range / 2
            lower = 0.5 - contrast / 2   # to keep the mean equals to 0.5
            element = grating * contrast + lower

            low = lambda x: x * new_res
            high = lambda x: low(x) + self.grating_res
            stimulus[low(i):high(i), low(j):high(j)] = element

        return stimulus

    def _get_stimulus_patch(self, stimulus):
        """
        TODO
        Selects a patch of the stimulus.
        """

        # TODO:: select an actually relevant patch
        half_res = ceil((self.stim_res - self.contrast_res) / 2)

        low = half_res
        high = low + self.contrast_res
        stimulus = stimulus[low:high, low:high]
        plot_heatmap(im=stimulus)
        return stimulus.flatten()