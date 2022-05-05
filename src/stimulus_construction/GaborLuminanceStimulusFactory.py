from src.params.ParamsGaborStimulus import *

from src.stimulus_construction.GaborLuminanceStimulus import *

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
    selected at random, depending on the grid_location of the annulus. All areas in the stimulus uncovered by annuli (void)
    share the same luminance. A square-shaped patch of the stimulus' figure is selected as an input to the Izhikevich
    oscillatory network (see :obj:`IzhikevichNetworkSimulator`).
    """

    def create(self, params_gabor: ParamsGaborStimulus) -> GaborLuminanceStimulus:
        """
        Goes through the steps to construct the luminance stimulus.

        :param params_gabor: parameters for creating a Gabor luminance stimulus.
        :type params_gabor: ParamsGaborStimulus

        :return: the luminance stimulus.
        :rtype: GaborLuminanceStimulus
        """

        atopix = params_gabor.diameter / params_gabor.diameter_dg  # conversion between pixels and degrees

        full_width: int = ceil(atopix * params_gabor.full_width_dg)
        full_height: int = ceil(atopix * params_gabor.full_height_dg)
        figure_width: int = ceil(atopix * params_gabor.figure_width_dg)
        figure_height: int = ceil(atopix * params_gabor.figure_height_dg)
        figure_ecc: float = atopix * params_gabor.figure_ecc_dg
        patch_size: int = ceil(atopix * params_gabor.patch_size_dg)

        grating = self._get_grating(params_gabor.spatial_freq, params_gabor.diameter_dg, params_gabor.diameter)
        figure_start, figure_end, figure_center = self._get_figure_coords(
            full_width, full_height, figure_width, figure_height, figure_ecc
        )
        stimulus = self._get_full_stimulus(
            full_width, full_height, grating,
            params_gabor.diameter, params_gabor.contrast_range, params_gabor.dist_scale,
            figure_start, figure_end
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
        Determines the grid_location of the figure within the stimulus.

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

