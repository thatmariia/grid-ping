from src.Params import *
from src.misc import *


class ParamsGaborStimulus(Params):
    """
    This class contains parameters for creating a Gabor luminance stimulus.

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


    :ivar spatial_freq: spatial frequency of the grating (cycles / degree).
    :ivar vlum: luminance of the void.
    :ivar diameter_dg: annulus' diameter in degrees.
    :ivar diameter: resolution (number of pixels in a single row) of single grating.
    :ivar dist_scale: how far the circles are from each other.
    :ivar full_width_dg: width of the full stimulus in degrees.
    :ivar full_height_dg: height of the full stimulus in degrees.
    :ivar contrast_range: contrast range for the figure.
    :ivar figure_width_dg: width of the figure in degrees.
    :ivar figure_height_dg: height of the figure in degrees.
    :ivar figure_ecc_dg: distance between the center of the stimulus and the center of the figure in degrees.
    :ivar patch_size_dg: side length of the stimulus patch in degrees.
    """

    def __init__(
            self,
            spatial_freq: float, vlum: float, diameter_dg: float, diameter: int,
            dist_scale: float, full_width_dg: float, full_height_dg: float,
            contrast_range: float, figure_width_dg: float, figure_height_dg: float, figure_ecc_dg: float,
            patch_size_dg: float
    ):
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

        self.spatial_freq = spatial_freq
        self.vlum = vlum
        self.diameter_dg = diameter_dg
        self.diameter = diameter
        self.dist_scale = dist_scale
        self.full_width_dg = full_width_dg
        self.full_height_dg = full_height_dg
        self.contrast_range = contrast_range
        self.figure_width_dg = figure_width_dg
        self.figure_height_dg = figure_height_dg
        self.figure_ecc_dg = figure_ecc_dg
        self.patch_size_dg = patch_size_dg