from src.GaborLuminanceStimulusFactory import *
from src.PatchGeometryFactory import *
from src.LuminanceToContrastConverter import *
from src.ContrastToFrequencyConverter import *
from src.FrequencyToCurrentConverter import *
from src.Stimulus import *

import numpy as np


class CurrentStimulusFactory:
    """
    This class creates an external stimulus (Gabor texture) and prepares for the neural network input.
    """

    def create(
            self,
            spatial_freq: float, vlum: float, diameter_dg: float, diameter: int,
            dist_scale: float, full_width_dg: float, full_height_dg: float,
            contrast_range: float, figure_width_dg: float, figure_height_dg: float, figure_ecc_dg: float,
            patch_size_dg: float,
            nr_ping_networks: int, slope: float, intercept: float, min_diam_rf: float
    ) -> Stimulus:
        """
        Creates an external stimulus (Gabor texture) and prepares for the neural network input.

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

        :param nr_ping_networks: number of circuits created by applying the lattice.
        :type nr_ping_networks: int

        :param slope: slope of the receptive field size.
        :type slope: float

        :param intercept: intercept of the receptive field size.
        :type intercept: float

        :param min_diam_rf: minimal size of the receptive field.
        :type min_diam_rf: float

        :raises:
            AssertionError: if the minimal diameter of the receptive field is not larger than 0.
        :raises:
            AssertionError: if the number of circuits is not a square as these circuits should be arranged in a square
            grid.
        :raises:
            AssertionError: if vertical lines of lattice cut through pixels.
        :raises:
            AssertionError: if horizontal lines of lattice cut through pixels.

        :return: a stimulus ready for the usage in a neural network.
        :rtype: Stimulus
        """

        assert min_diam_rf > 0, \
            "The minimal diameter_dg of the receptive field should be larger than 0."
        assert int(math.sqrt(nr_ping_networks)) == math.sqrt(nr_ping_networks), \
            "The circuits created by lattice should be arranged in a square grid. Make sure the number of circuits " \
            "is a perfect square. "

        stimulus_luminance = GaborLuminanceStimulusFactory().create(
            spatial_freq, vlum, diameter_dg, diameter,
            dist_scale, full_width_dg, full_height_dg,
            contrast_range, figure_width_dg, figure_height_dg, figure_ecc_dg,
            patch_size_dg
        )

        stimulus_luminance.plot_stimulus(filename="test-full-stimulus")
        stimulus_luminance.plot_patch(filename="test-stimulus-patch")

        assert np.shape(stimulus_luminance.stimulus_patch)[0] % int(math.sqrt(nr_ping_networks)) == 0, \
            "Vertical lines of lattice should not cut through pixels."
        assert np.shape(stimulus_luminance.stimulus_patch)[1] % int(math.sqrt(nr_ping_networks)) == 0, \
            "Horizontal lines of lattice should not cut through pixels."

        patch_geometry = PatchGeometryFactory().create(
            nr_ping_networks=nr_ping_networks,
            stimulus_patch=stimulus_luminance.stimulus_patch,
            patch_start=stimulus_luminance.patch_start,
            stimulus_center=stimulus_luminance.stimulus_center,
            atopix=stimulus_luminance.atopix
        )

        stimulus_contrasts = LuminanceToContrastConverter().convert(
            slope, intercept, min_diam_rf, patch_geometry, stimulus_luminance
        )
        stimulus_frequencies = ContrastToFrequencyConverter().convert(
            stimulus_contrasts
        )
        stimulus_currents = FrequencyToCurrentConverter().convert(
            stimulus_frequencies
        )

        stimulus = Stimulus(
            stimulus_currents, patch_geometry
        )

        return stimulus

