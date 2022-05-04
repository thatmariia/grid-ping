from src.misc import *
from src.GaborLuminanceStimulus import *

import numpy as np
from math import pi, ceil
from tqdm import tqdm
from itertools import product


class GaborLuminanceStimulusFactory:
    """
    This class constructs the Gabor texture stimulus (and a patch from it).

    The external stimulus represents a rectangular grid of non-overlapping equispaced grating annuli -
    circular Gabor wavelets :cite:p:`MaryamPLACEHOLDER`. The luminance of the stimuli varies between 0 (black) and
    1 (white). All annuli have equal diameters but vary in contrast. The grid includes a figure - a rectangular subgrid
    in the bottom right quadrant of the stimulus, where all annuli share similar contrasts, and a background that
    constitutes the rest of the grid. There, annuli vary in contrast significantly. The contrast of every annulus is
    selected at random, depending on the location of the annulus. All areas in the stimulus uncovered by annuli (void)
    share the same luminance. A square-shaped patch of the stimulus' figure is selected as an input to the Izhikevich
    oscillatory network (see :obj:`IzhikevichNetworkSimulator`).
    """

    def create(
            self,
            spatial_freq: float, vlum: float, diameter_dg: float, diameter: int,
            dist_scale: float, full_width_dg: float, full_height_dg: float,
            contrast_range: float, figure_width_dg: float, figure_height_dg: float, figure_ecc_dg: float,
            patch_size_dg: float
    ) -> GaborLuminanceStimulus:
        """
        Goes through the steps to construct the luminance stimulus.

        :param spatial_freq: spatial frequency of the grating (cycles / degree).
        :type spatial_freq: float

        :param vlum: luminance of the void.
        :type vlum: float

        :param diameter_dg: annulus' diameter in degrees.
        :type diameter_dg: float

        :param diameter: resolution (number of pixels in a single row) of single grating.
        :type diameter: int

        :param dist_scale: how far the circles are from each other.
        :type dist_scale: float

        :param full_width_dg: width of the full stimulus in degrees.
        :type full_width_dg: float

        :param full_height_dg: height of the full stimulus in degrees.
        :type full_height_dg: float

        :param contrast_range: contrast range for the figure.
        :type contrast_range: float

        :param figure_width_dg: width of the figure in degrees.
        :type figure_width_dg: float

        :param figure_height_dg: height of the figure in degrees.
        :type figure_height_dg: float

        :param figure_ecc_dg: distance between the center of the stimulus and the center of the figure in degrees.
        :type figure_ecc_dg: float

        :param patch_size_dg: side length of the stimulus patch in degrees.
        :type patch_size_dg: float

        :raises:
            AssertionError: spatial frequency is not greater than 0.
        :raises:
            AssertionError: void luminance does not fall in range :math:`[0, 1]`.
        :raises:
            AssertionError: annulus diameter is not larger than 0 degrees.
        :raises:
            AssertionError: annulus diameter is smaller than 1 pixel.
        :raises:
            AssertionError: the distance between neighbouring annuli is less than 1 diameter.
        :raises:
            AssertionError: stimulus is less wide than annulus.
        :raises:
            AssertionError: stimulus is less tall than annulus.
        :raises:
            AssertionError: contrast range does not fall in range :math:`(0, 1]`.
        :raises:
            AssertionError: figure width is larger than half the width of the stimulus or is not larger than 0.
        :raises:
            AssertionError: figure height is larger than half the height of the stimulus or is not larger than 0.
        :raises:
            AssertionError: figure cannot be positioned so that it is contained within the stimulus quadrant.
        :raises:
            AssertionError: size of the patch is smaller than one of the figure sides.

        :return: the luminance stimulus.
        :rtype: GaborLuminanceStimulus
        """

        figure_half_diag_dg = euclidian_dist((0.5 * figure_width_dg, 0.5 * figure_height_dg))
        stim_half_diag_dg = euclidian_dist((0.5 * full_width_dg, 0.5 * full_height_dg))

        assert spatial_freq > 0, \
            "Spatial frequency must be greater than 0."
        assert (vlum >= 0) and (vlum <= 1), \
            "Void luminance must fall in range :math:`[0, 1]`."
        assert diameter_dg > 0, \
            "Annulus diameter must be larger than 0 degrees."
        assert diameter >= 1, \
            "Annulus diameter must be at least 1 pixel."
        assert dist_scale >= 1, \
            "The distance between neighbouring annuli must be at least 1 diameter."
        assert full_width_dg >= diameter_dg, \
            "The stimulus must be at least as wide as an annulus."
        assert full_height_dg >= diameter_dg, \
            "The stimulus must be at least as tall as an annulus."
        assert (contrast_range > 0) and (contrast_range <= 1), \
            "Contrast range must fall in range :math:`(0, 1]`."
        assert (figure_width_dg > 0) and (figure_width_dg <= 0.5 * full_width_dg), \
            "Figure width cannot be larger than half the width of the stimulus and must be larger than 0."
        assert (figure_height_dg > 0) and (figure_height_dg <= 0.5 * full_height_dg), \
            "Figure height cannot be larger than half the height of the stimulus and must be larger than 0."
        assert (figure_ecc_dg >= figure_half_diag_dg) and (figure_ecc_dg <= stim_half_diag_dg - figure_half_diag_dg), \
            "Figure must be positioned so that it is contained within the stimulus quadrant."
        assert (patch_size_dg > 0) and (patch_size_dg <= min(figure_width_dg, figure_height_dg)), \
            "The size of the patch cannot be smaller than either of the figure sides."

        atopix = diameter / diameter_dg  # conversion between pixels and degrees

        full_width: int = ceil(atopix * full_width_dg)
        full_height: int = ceil(atopix * full_height_dg)
        figure_width: int = ceil(atopix * figure_width_dg)
        figure_height: int = ceil(atopix * figure_height_dg)
        figure_ecc: float = atopix * figure_ecc_dg
        patch_size: int = ceil(atopix * patch_size_dg)

        grating = self._get_grating(spatial_freq, diameter_dg, diameter)
        figure_start, figure_end, figure_center = self._get_figure_coords(
            full_width, full_height, figure_width, figure_height, figure_ecc
        )
        stimulus = self._get_full_stimulus(
            full_width, full_height, grating, diameter, contrast_range, dist_scale, figure_start, figure_end
        )
        patch_start, stimulus_patch = self._select_stimulus_patch(stimulus, figure_center, patch_size)

        luminance_stimulus = GaborLuminanceStimulus(
            atopix=atopix,
            stimulus=stimulus,
            stimulus_center=(0.5 * full_height, 0.5 * full_width),
            stimulus_patch=stimulus_patch,
            patch_start=patch_start
        )

        return luminance_stimulus

    def _get_grating(self, spatial_freq: float, diameter_dg: float, diameter: int) -> np.ndarray[(int, int), float]:
        """
        Generates a grating (single annulus) of the maximum contrast.

        :param spatial_freq: spatial frequency of the grating (cycles / degree).
        :type spatial_freq: float

        :param diameter_dg: annulus diameter in degrees.
        :type diameter_dg: float

        :param diameter: resolution (number of pixels in a single row) of single grating.
        :type diameter: int

        :return: the luminance matrix of the single annulus.
        :rtype: numpy.ndarray[(int, int), float]
        """

        r = np.linspace(
            -diameter_dg / 2,
            diameter_dg / 2,
            num=diameter,
            endpoint=True
        )
        x, y = np.meshgrid(r, r)
        radii = np.power(np.power(x, 2) + np.power(y, 2), 1 / 2)
        mask = radii <= (diameter_dg / 2)
        grating = np.cos(2 * pi * radii * spatial_freq + pi)
        grating = 0.5 * (np.multiply(grating, mask) + 1)
        return grating

    def _get_figure_coords(
            self, full_width: int, full_height: int, figure_width: int, figure_height: int, figure_ecc: float
    ) -> tuple[tuple[int, int], tuple[int, int], tuple[float, float]]:
        """
        Determines the location of the figure within the stimulus.

        :param full_width: width of the full stimulus.
        :type full_width: int

        :param full_height: height of the full stimulus.
        :type full_height: int

        :param figure_width: width of the figure.
        :type figure_width: float

        :param figure_height: height of the figure.
        :type figure_height: float

        :param figure_ecc: distance between the center of the stimulus and the center of the figure.
        :param figure_ecc: float

        :return: top left, bottom right, and center coordinates of the figure.
        :rtype: tuple[tuple[int, int], tuple[int, int], tuple[float, float]]
        """

        dheight = full_height - figure_height
        dwidth = full_width - figure_width

        angle_max = np.arccos(0.5 * dheight / figure_ecc) if (figure_ecc > 0.5 * dheight) \
            else np.arcsin(0.5 * figure_width / figure_ecc)
        angle_max = pi / 2 - angle_max

        angle_min = np.arccos(0.5 * dwidth / figure_ecc) if (figure_ecc > 0.5 * dwidth) \
            else np.arcsin(0.5 * figure_height / figure_ecc)

        angle = np.random.uniform( angle_min, angle_max)

        figure_center = add_points([
            (full_height / 2, full_width / 2),
            (figure_ecc * np.sin(angle), figure_ecc * np.cos(angle))
        ])

        figure_start = point_ceil(add_points([
            figure_center,
            (-figure_height / 2, -figure_width / 2)
        ]))
        figure_end = add_points([
            figure_start,
            (figure_height, figure_width)
        ])

        return figure_start, figure_end, figure_center

    def _get_full_stimulus(
            self,
            full_width: int, full_height: int,
            grating: np.ndarray[(int, int), float], diameter: int, contrast_range: float, dist_scale: float,
            figure_start: tuple[int, int], figure_end: tuple[int, int]
    ) -> np.ndarray[(int, int), float]:
        """
        Generates the whole stimulus.

        :param full_width: width of the full stimulus.
        :type full_width: int

        :param full_height: height of the full stimulus.
        :type full_height: int

        :param grating: :param grating: the luminance matrix of the annulus.
        :type grating: numpy.ndarray[(int, int), float]

        :param diameter: annulus diameter.
        :type diameter: float

        :param contrast_range: contrast range for the figure.
        :type contrast_range: float

        :param dist_scale: how far the annuli are from each other.
        :type dist_scale: float

        :param figure_start: left top coordinate of the figure.
        :type figure_start: tuple[int, int]

        :param figure_end: bottom right coordinate of the figure.
        :type figure_end: tuple[int, int]

        :return: luminance matrix of the stimulus.
        :rtype: numpy.ndarray[(int, int), float]
        """

        # nr of pixels for annulus + void
        alloc_space = ceil(dist_scale * diameter)
        annulus_start_in_alloc = (alloc_space - diameter) // 2

        # nr of annuli in each row and column
        reps_in_row = ceil(full_width / alloc_space)  # nr of cols
        reps_in_col = ceil(full_height / alloc_space)  # nr of rows

        stimulus = np.ones((full_height, full_width)) * 0.5

        # for i in tqdm(range(reps_in_col)):
        #     for j in tqdm(range(reps_in_row), leave=False):
        # single loop in favor of the working progress bar:
        for r in (pbar := tqdm(range(reps_in_col * reps_in_row))):
            pbar.set_description("Stimulus composition")
            i = r // reps_in_row
            j = r % reps_in_row

            # annulus top left
            annulus_start = add_points([
                (i * alloc_space, j * alloc_space),
                (annulus_start_in_alloc, annulus_start_in_alloc)
            ])
            # annulus bottom right
            annulus_end = add_points([
                annulus_start,
                (diameter, diameter)
            ])
            # all annulus corners
            annulus_corners = list(product([annulus_start[0], annulus_end[0]], [annulus_start[1], annulus_end[1]]))

            # annulus contrast
            curr_cr = contrast_range if self._is_annulus_in_figure(annulus_corners, figure_start, figure_end) else 1
            modified_grating = 0.5 + (grating - 0.5) * (np.random.uniform() * curr_cr + 0.5 * (1 - curr_cr))

            # if the whole annulus doesn't fit
            annulus_cutoff = (
                min(annulus_end[0], full_height),
                min(annulus_end[1], full_width)
            )
            cut_amount = add_points([
                annulus_end,
                annulus_cutoff
            ], [1, -1])
            grating_cutoff = add_points([
                (diameter, diameter),
                cut_amount
            ], [1, -1])

            if all(i >= 0 for i in grating_cutoff):
                stimulus[annulus_start[0]:annulus_cutoff[0], annulus_start[1]:annulus_cutoff[1]] = \
                    modified_grating[:grating_cutoff[0], :grating_cutoff[1]]

        return stimulus

    def _is_annulus_in_figure(
            self, annulus_corners: list[tuple[int, int]], figure_start: tuple[int, int], figure_end: tuple[int, int]
    ) -> bool:
        """
        Checks if an annulus belongs to the figure.

        :param annulus_corners: list of corner coordinates of the annulus.
        :type annulus_corners: list[tuple[int, int]]

        :param figure_start: left top coordinate of the figure.
        :type figure_start: tuple[int, int]

        :param figure_end: bottom right coordinate of the figure.
        :type figure_end: tuple[int, int]

        :return: True if the annulus belongs to the figure, False otherwise.
        :rtype: bool
        """

        for c in annulus_corners:
            # if a corner is in the figure
            if (c[0] >= figure_start[0]) and (c[0] <= figure_end[0]) and \
                    (c[1] >= figure_start[1]) and (c[1] <= figure_end[1]):
                return True
        return False

    def _select_stimulus_patch(
            self, stimulus: np.ndarray[(int, int), float], figure_center: tuple[float, float], patch_size: int
    ) -> tuple[tuple[int, int], np.ndarray[(int, int), float]]:
        """
        Selects a patch of the stimulus.

        :param stimulus: luminance matrix of the stimulus.
        :type stimulus: numpy.ndarray[(int, int), float]

        :param figure_center: the center point of the figure.
        :type figure_center: tuple[float, float]

        :param patch_size: side length of the stimulus.
        :type patch_size: int

        :return: the luminance matrix of a patch of the stimulus and its top left coordinate.
        :rtype: tuple[tuple[int, int], numpy.ndarray[(int, int), float]]
        """

        patch_start = point_ceil(add_points([
            figure_center,
            (-patch_size / 2, -patch_size / 2)
        ]))
        patch_end = add_points([
            patch_start,
            (patch_size, patch_size)
        ])

        stimulus_patch = stimulus[patch_start[0]:patch_end[0], patch_start[1]:patch_end[1]]
        return patch_start, stimulus_patch

