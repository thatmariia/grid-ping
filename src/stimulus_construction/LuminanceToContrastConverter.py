from src.params.ParamsReceptiveField import *

from src.stimulus_construction.PatchGeometry import *
from src.stimulus_construction.GaborLuminanceStimulus import *

import numpy as np
from math import sqrt, exp
from statistics import mean
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


class LuminanceToContrastConverter:
    """
    This class converts the luminance stimulus into the local contrasts stimulus.
    """

    def convert(
            self,
            params_rf: ParamsReceptiveField, patch_geometry: PatchGeometry, stimulus_luminance: GaborLuminanceStimulus
    ) -> np.ndarray[int, float]:
        """
        Converts the luminance stimulus into the local contrasts stimulus.

        The approach is derived from :cite:p:`MaryamPLACEHOLDER`.

        :param params_rf: parameters for the receptive field.
        :type params_rf: ParamsReceptiveField

        :param patch_geometry: information about the grid layout of the stimulus patch in correspondence with PING networks.
        :type patch_geometry: PatchGeometry

        :param stimulus_luminance: luminance stimulus container.
        :type stimulus_luminance: GaborLuminanceStimulus

        :return: the stimulus converted to local contrasts.
        :rtype: numpy.ndarray[int, float]
        """

        return self._compute_local_contrasts(params_rf, patch_geometry, stimulus_luminance)


    def _get_weights(
            self,
            params_rf: ParamsReceptiveField, patch_geometry: PatchGeometry
    ) -> np.ndarray[int, float]:
        # compute weights for every pixel for every network

        # compute eccentricity of centers
        eccentricities = np.array([
            patch_geometry.eccentricity_in_patch(point=(x, y))
            for x, y in zip(patch_geometry.centers_x_dg, patch_geometry.centers_y_dg)
        ])
        # compute rf diameters of centers
        diam_rf = np.maximum(params_rf.slope * eccentricities + params_rf.intercept, params_rf.min_diam_rf)
        # compute std of centers
        std = diam_rf / 4.0

        # compute weights for every pixel for every network
        weights = np.zeros((patch_geometry.nr_ping_networks, len(patch_geometry.all_pixels_x_dg)))
        for ping_id in range(patch_geometry.nr_ping_networks):
            # with euclidian distance of every pixel to the center of the ping network
            weights[ping_id] = np.exp(
                -(
                        (patch_geometry.all_pixels_x_dg - patch_geometry.centers_x_dg[ping_id]) ** 2 +
                        (patch_geometry.all_pixels_y_dg - patch_geometry.centers_y_dg[ping_id]) ** 2
                ) / (2 * std[ping_id] ** 2)
            )

        # diagonal normalization weights by summing weights in networks
        weights_normalizer = np.diag(1.0 / np.sum(weights, axis=1))
        # normalize weights with the normalizer
        weights = weights.T * np.diag(weights_normalizer)
        weights = weights.T

        return weights

    def _compute_local_contrasts(
            self,
            params_rf: ParamsReceptiveField, patch_geometry: PatchGeometry, stimulus_luminance: GaborLuminanceStimulus
    )  -> np.ndarray[int, float]:
        # get weights
        weights = self._get_weights(params_rf, patch_geometry)

        # compute mean luminance
        mean_luminance = np.mean(stimulus_luminance.stimulus.flatten())
        # compute luminance differences in the patch
        luminance_diffs = stimulus_luminance.stimulus_patch.flatten() - mean_luminance
        # compute normalized luminance
        luminance_normalized = (luminance_diffs / mean_luminance) ** 2

        # compute local contrasts
        local_contrasts = np.sqrt(np.matmul(weights, luminance_normalized))

        return local_contrasts
