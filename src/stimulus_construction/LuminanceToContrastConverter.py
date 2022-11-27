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
            params_rf: ParamsReceptiveField, patch_geometry: PatchGeometry, stimulus_luminance: GaborLuminanceStimulus
    ) -> np.ndarray[int, float]:
        # compute weights for every pixel for every network

        # compute eccentricity of centers
        eccentricities = np.array([patch_geometry.eccentricity_in_patch(point=ping_pix.center_dg) for ping_pix in patch_geometry.ping_networks_pixels])
        # compute rf diameters of centers
        diam_rf = np.maximum(params_rf.slope * eccentricities + params_rf.intercept, params_rf.min_diam_rf)
        # compute std of centers
        std = diam_rf / 4.0

        # compute weights for every pixel for every network
        weights = np.zeros((len(patch_geometry.ping_networks_pixels), len(patch_geometry.all_pixels_x_dg)))
        for ping_id in range(len(patch_geometry.ping_networks_pixels)):
            # euclidian distance of every pixel to the center of the ping network
            weights[ping_id] = np.exp(
                -(
                        (patch_geometry.all_pixels_x_dg - patch_geometry.ping_networks_pixels[ping_id].center_dg[0]) ** 2 +
                        (patch_geometry.all_pixels_y_dg - patch_geometry.ping_networks_pixels[ping_id].center_dg[1]) ** 2
                ) / (2 * std[ping_id] ** 2)
            )

        # diagonal normalization weights by summing columns
        weights_normalizer = np.diag(1.0 / np.sum(weights, axis=0))
        # normalize weights with the normalizer
        weights = np.matmul(weights, weights_normalizer).T

        # # compute mean luminance
        # mean_luminance = np.mean(stimulus_luminance.stimulus)
        # # compute luminance differences in the patch
        # luminance_diffs = stimulus_luminance.stimulus_patch - mean_luminance
        # # compute normalized luminance squared
        # luminance_squared = (luminance_diffs / mean_luminance) ** 2
        #
        # # compute local contrasts
        # print("shape of weights: ", weights.shape)
        # print("shape of luminance_squared: ", luminance_squared.shape)
        # local_contrasts = np.sqrt(np.matmul(weights, luminance_squared))
        # print(local_contrasts.shape)

        return weights

    def _compute_local_contrasts_OLD(
            self,
            params_rf: ParamsReceptiveField, patch_geometry: PatchGeometry, stimulus_luminance: GaborLuminanceStimulus
    ) -> np.ndarray[int, float]:
        # get weights
        weights = self._get_weights(params_rf, patch_geometry, stimulus_luminance)

        # compute mean luminance
        mean_luminance = np.mean(stimulus_luminance.stimulus)
        # # compute luminance differences in the patch
        # luminance_diffs = stimulus_luminance.stimulus_patch - mean_luminance
        # # compute normalized luminance squared
        # luminance_squared = (luminance_diffs / mean_luminance) ** 2

        local_contrasts = []

        for curr_circuit_id in (pbar := tqdm(range(len(patch_geometry.ping_networks_pixels)))):
            pbar.set_description("Local contrast computation")
            # current circuit
            curr_circuit = patch_geometry.ping_networks_pixels[curr_circuit_id]

            num = 0
            denom = 0

            for pixel_id in range(len(curr_circuit.pixels)):
                # get weight of that pixel
                weight = weights[curr_circuit_id, pixel_id]
                # get normalized luminance squared of that pixel
                pix = curr_circuit.pixels[pixel_id]
                luminance_squared = (stimulus_luminance.stimulus_patch[pix[0]][pix[1]] - mean_luminance) ** 2 / (mean_luminance ** 2)
                num += weight * luminance_squared
                denom += weight

            local_contrast = sqrt(num / denom)
            local_contrasts.append(local_contrast)

        return np.array(local_contrasts)


    def _compute_local_contrasts(
            self,
            params_rf: ParamsReceptiveField, patch_geometry: PatchGeometry, stimulus_luminance: GaborLuminanceStimulus
    ) -> np.ndarray[int, float]:
        """
        Computes local contrasts for each circuit.

        :param patch_geometry: information about the grid layout of the stimulus patch in correspondence with PING networks.
        :type patch_geometry: PatchGeometry

        :param stimulus_luminance: luminance stimulus container.
        :type stimulus_luminance: GaborLuminanceStimulus

        :param params_rf: parameters for the receptive field.
        :type params_rf: ParamsReceptiveField

        :return: list containing local contrast values for each circuit.
        :rtype: numpy.ndarray[int, float]
        """

        # TODO:: optimize

        mean_luminance = mean(np.array(stimulus_luminance.stimulus).flatten())
        local_contrasts = []

        for curr_circuit in (pbar := tqdm(patch_geometry.ping_networks_pixels)):
            pbar.set_description("Local contrast computation")

            eccentricity = patch_geometry.eccentricity_in_patch(point=curr_circuit.center_dg)
            num = 0
            denum = 0

            #for circuit in patch_geometry.ping_networks_pixels:
            for i in range(len(curr_circuit.pixels)):
                pix = curr_circuit.pixels[i]
                pix_dg = curr_circuit.pixels_dg[i]

                weight = self._get_weight(
                    center=curr_circuit.center_dg,
                    pixel=pix_dg,
                    eccentricity=eccentricity,
                    params_rf=params_rf
                )

                num += weight * (stimulus_luminance.stimulus_patch[pix[0]][pix[1]] - mean_luminance) ** 2 / (mean_luminance ** 2)
                denum += weight

            local_contrast = sqrt(num / denum)
            local_contrasts.append(local_contrast)

        return np.array(local_contrasts)


    def _compute_local_contrasts_OLD2(
            self,
            params_rf: ParamsReceptiveField, patch_geometry: PatchGeometry, stimulus_luminance: GaborLuminanceStimulus
    ) -> np.ndarray[int, float]:
        """
        Computes local contrasts for each circuit.

        :param patch_geometry: information about the grid layout of the stimulus patch in correspondence with PING networks.
        :type patch_geometry: PatchGeometry

        :param stimulus_luminance: luminance stimulus container.
        :type stimulus_luminance: GaborLuminanceStimulus

        :param params_rf: parameters for the receptive field.
        :type params_rf: ParamsReceptiveField

        :return: list containing local contrast values for each circuit.
        :rtype: numpy.ndarray[int, float]
        """

        # TODO:: optimize

        mean_luminance = mean(np.array(stimulus_luminance.stimulus).flatten())
        local_contrasts = []

        for curr_circuit in (pbar := tqdm(patch_geometry.ping_networks_pixels)):
            pbar.set_description("Local contrast computation")

            eccentricity = patch_geometry.eccentricity_in_patch(point=curr_circuit.center_dg)
            num = 0
            denum = 0

            for circuit in patch_geometry.ping_networks_pixels:
                for i in range(len(circuit.pixels)):
                    pix = circuit.pixels[i]
                    pix_dg = circuit.pixels_dg[i]

                    weight = self._get_weight(
                        center=curr_circuit.center_dg,
                        pixel=pix_dg,
                        eccentricity=eccentricity,
                        params_rf=params_rf
                    )

                    num += weight * (stimulus_luminance.stimulus_patch[pix[0]][pix[1]] - mean_luminance) ** 2 / (mean_luminance ** 2)
                    denum += weight

            local_contrast = sqrt(num / denum)
            local_contrasts.append(local_contrast)

        return np.array(local_contrasts)

    def _get_weight(
            self,
            center: tuple[float, float], pixel: tuple[int, int], eccentricity: float,
            params_rf: ParamsReceptiveField
    ) -> float:
        """
        Computes weight of a pixel with respect to a circuit.

        :param center: coordinate of the circuit center.
        :type center: tuple[float, float]

        :param pixel: coordinate of the pixel.
        :type pixel: tuple[float, float]

        :param eccentricity: eccentricity of the circuit center.
        :type eccentricity: float

        :param params_rf: parameters for the receptive field.
        :type params_rf: ParamsReceptiveField

        :return: weight of a pixel with respect to a circuit.
        :rtype: float
        """

        diam_rf = max(params_rf.slope * eccentricity + params_rf.intercept, params_rf.min_diam_rf)
        std = diam_rf / 4.0
        return exp(-(euclidian_dist(pixel, center) ** 2) / (2 * std ** 2))
