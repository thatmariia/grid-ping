from src.GaborLuminanceStimulus import *
from src.StimulusCircuit import *

import numpy as np
from math import sqrt, exp
from itertools import product
from statistics import mean


class InputStimulus(GaborLuminanceStimulus):
    """
    This class transforms a luminance stimulus patch to current.

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

    :param nr_circuits: number of circuits created by applying the lattice.
    :type nr_circuits: int

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


    :ivar _nr_circuits: number of circuits created by applying the lattice.
    :type _nr_circuits: int

    :ivar current: list of currents produced by respective circuits in the stimulus.
    :type current: list[float]
    """

    def __init__(self,
                 dist_scale, contrast_range, spatial_freq, diameter, side_length, grating_res, patch_res,
                 nr_circuits, slope, intercept, min_diam_rf
                 ):
        super().__init__(dist_scale, contrast_range, spatial_freq, diameter, side_length, grating_res, patch_res)

        assert min_diam_rf > 0, "The minimal diameter of the receptive field should be larger than 0."
        assert int(math.sqrt(nr_circuits)) == math.sqrt(nr_circuits), \
            "The circuits created by lattice should be arranged in a square grid. Make sure the number of circuits " \
            "is a perfect square. "

        # TODO:: add assert on the _nr_circuits vs patch size if full pixels need to be contained in lattice: N|M

        self._nr_circuits = nr_circuits

        circuits = self._assign_circuits()
        self.current = self._get_input_current(circuits, slope, intercept, min_diam_rf)

    def _assign_circuits(self):
        """
        Creates circuits and assigns centers and pixels of the stimulus patch to them.

        :return: list of all circuits of the stimulus patch created by applying a lattice.
        :rtype: list[StimulusCircuit]
        """

        # assuming that circuits contain full pixels
        lattice_edges = (
            # horizontal edges
            np.linspace(
                0,
                np.shape(self.stimulus_patch)[0],
                num=int(sqrt(self._nr_circuits)) + 1,
                endpoint=True,
                dtype=int
            ),
            # vertical edges
            np.linspace(
                0,
                np.shape(self.stimulus_patch)[1],
                num=int(sqrt(self._nr_circuits)) + 1,
                endpoint=True,
                dtype=int
            )
        )

        circuits = []

        for hor_i in range(len(lattice_edges[0]) - 1):
            for ver_i in range(len(lattice_edges[1]) - 1):
                center = (
                    lattice_edges[0][hor_i] + (lattice_edges[0][hor_i + 1] - lattice_edges[0][hor_i]) / 2.0,
                    lattice_edges[0][ver_i] + (lattice_edges[0][ver_i + 1] - lattice_edges[0][ver_i]) / 2.0,
                )
                pixels = list(product(
                    np.arange(lattice_edges[0][hor_i], lattice_edges[0][hor_i + 1]),
                    np.arange(lattice_edges[0][ver_i], lattice_edges[0][ver_i + 1])
                ))
                circuit = StimulusCircuit(
                    center=center,
                    pixels=pixels
                )
                circuits.append(circuit)

        return circuits

    def _get_input_current(self, circuits, slope, intercept, min_diam_rf):
        """
        Performes all the neccessary steps to transform luminance to _current.

        :param circuits: list of all circuits of the stimulus patch created by applying a lattice.
        :type circuits: list[StimulusCircuit]

        :param slope: slope of the receptive field size.
        :type slope: float

        :param intercept: intercept of the receptive field size.
        :type intercept: float

        :param min_diam_rf: minimal size of the receptive field.
        :type min_diam_rf: float

        :return: list containing currents created by each circuit.
        :rtype: list[float]
        """

        local_contrasts = self._compute_local_contrasts(circuits, slope, intercept, min_diam_rf)
        frequencies = self._compute_frequencies(local_contrasts)
        current = self._compute_current(frequencies)

        return current

    def _get_weight(self, center, pixel, slope, intercept, min_diam_rf):
        """
        Computes weight of a pixel with respect to a circuit.

        :param center: coordinate of the circuit center.
        :type center: tuple(float, float)

        :param pixel: coordinate of the pixel.
        :type pixel: tuple(float, float)

        :param slope: slope of the receptive field size.
        :type slope: float

        :param intercept: intercept of the receptive field size.
        :type intercept: float

        :param min_diam_rf: minimal size of the receptive field.
        :type min_diam_rf: float

        :return: weight of a pixel with respect to a circuit.
        :rtype: float
        """

        eccentricity = self._eccentricity_in_patch(point=center)
        diam_rf = max(slope * eccentricity + intercept, min_diam_rf)
        std = diam_rf / 4.0
        return exp(-euclidian_dist_R2(pixel, center) / (2 * std ** 2))

    def _compute_local_contrasts(self, circuits, slope, intercept, min_diam_rf):
        """
        Computes local contrasts for each circuit.

        :param circuits: list of all circuits of the stimulus patch created by applying a lattice.
        :type circuits: list[StimulusCircuit]

        :param slope: slope of the receptive field size.
        :type slope: float

        :param intercept: intercept of the receptive field size.
        :type intercept: float

        :param min_diam_rf: minimal size of the receptive field.
        :type min_diam_rf: float

        :return: list containing local contrast values for each circuit.
        :rtype: list[float]
        """

        mean_luminance = mean(np.array(self.stimulus_patch).flatten())
        local_contrasts = []

        for circuit in circuits:
            num = 0
            denum = 0
            for pix in circuit.pixels:
                weight = self._get_weight(
                    center=circuit.center,
                    pixel=pix,
                    slope=slope,
                    intercept=intercept,
                    min_diam_rf=min_diam_rf
                )
                num += weight * (self.stimulus_patch[pix[0]][pix[1]] - mean_luminance) ** 2 / mean_luminance
                denum += weight

            local_contrast = sqrt(num / denum)
            local_contrasts.append(local_contrast)

        return local_contrasts

    def _compute_frequencies(self, local_contrasts):
        """
        Computes oscillation frequencies of the circuit through local contrasts.

        :param local_contrasts: list containing local contrast values for each circuit.
        :type local_contrasts: list[float]

        :return: list containing oscillation frequencies for each circuit.
        :rtype: ndarray[float]
        """

        return 25 + 0.25 * np.array(local_contrasts)

    def _compute_current(self, frequencies):
        """
        Computes _current through oscillation frequencies. ARTIFICIAL FUNCTION - REAL NOT IMPLEMENTED YET.

        :param frequencies: list containing oscillation frequencies for each circuit.
        :type frequencies: list[float]

        :return: TODO
        :rtype: list[float]
        """

        # TODO:: implement the real strategy
        return 100.0 / (0.5 + 0.5 * np.array(frequencies))
