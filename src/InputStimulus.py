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
    :raises:
        AssertionError: if vertical lines of lattice cut through pixels.
    :raises:
        AssertionError: if horizontal lines of lattice cut through pixels.


    :ivar _nr_circuits: number of circuits created by applying the lattice.
    :type _nr_circuits: int

    :ivar current: list of currents produced by respective circuits in the stimulus.
    :type current: list[float]
    """

    def __init__(
            self,
            spatial_freq: float, vlum: float, diameter_dg: float, diameter: int,
            dist_scale: float, full_width_dg: float, full_height_dg: float,
            contrast_range: float, figure_width_dg: float, figure_height_dg: float, figure_ecc_dg: float,
            patch_size_dg: float,
            nr_circuits: int, slope: float, intercept: float, min_diam_rf: float
    ):
        super().__init__(
            spatial_freq, vlum, diameter_dg, diameter,
            dist_scale, full_width_dg, full_height_dg,
            contrast_range, figure_width_dg, figure_height_dg, figure_ecc_dg,
            patch_size_dg
        )

        assert min_diam_rf > 0, \
            "The minimal diameter_dg of the receptive field should be larger than 0."
        assert int(math.sqrt(nr_circuits)) == math.sqrt(nr_circuits), \
            "The circuits created by lattice should be arranged in a square grid. Make sure the number of circuits " \
            "is a perfect square. "
        assert np.shape(self.stimulus_patch)[0] % int(math.sqrt(nr_circuits)) == 0, \
            "Vertical lines of lattice should not cut through pixels."
        assert np.shape(self.stimulus_patch)[1] % int(math.sqrt(nr_circuits)) == 0, \
            "Horizontal lines of lattice should not cut through pixels."

        self._nr_circuits = nr_circuits

        circuits: list[StimulusCircuit] = self._assign_circuits()
        self.current = self._get_input_current(circuits, slope, intercept, min_diam_rf)

    def _assign_circuits(self) -> list[StimulusCircuit]:
        """
        Creates circuits and assigns centers and pixels of the stimulus patch to them.

        :return: list of all circuits of the stimulus patch created by applying a lattice.
        :rtype: list[StimulusCircuit]
        """

        # assuming that circuits contain full pixels
        lattice_edges = np.linspace(
            0,
            np.shape(self.stimulus_patch)[0],
            num=int(sqrt(self._nr_circuits)) + 1,
            endpoint=True,
            dtype=int
        )

        circuits = []

        for i in range(len(lattice_edges) - 1):
            for j in range(len(lattice_edges) - 1):

                center = add_points([
                    (lattice_edges[i], lattice_edges[j]),
                    ((lattice_edges[i + 1] - lattice_edges[i]) / 2, (lattice_edges[j + 1] - lattice_edges[j]) / 2)
                ])
                pixels = list(product(
                    np.arange(lattice_edges[i], lattice_edges[i + 1]),
                    np.arange(lattice_edges[j], lattice_edges[j + 1])
                ))

                circuit = StimulusCircuit(
                    center=center,
                    pixels=pixels
                )
                circuits.append(circuit)

        return circuits

    def _get_input_current(
            self, circuits: list[StimulusCircuit], slope: float, intercept: float, min_diam_rf: float
    ) -> list[float]:
        """
        Performs all the necessary steps to transform luminance to current.

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

    def _get_weight(
            self,
            center: tuple[float, float], pixel: tuple[int, int], eccentricity: float,
            slope: float, intercept: float, min_diam_rf: float
    ) -> float:
        """
        Computes weight of a pixel with respect to a circuit.

        :param center: coordinate of the circuit center.
        :type center: tuple[float, float]

        :param pixel: coordinate of the pixel.
        :type pixel: tuple[float, float]

        :param eccentricity: eccentricity of the circuit center.
        :type eccentricity: float

        :param slope: slope of the receptive field size.
        :type slope: float

        :param intercept: intercept of the receptive field size.
        :type intercept: float

        :param min_diam_rf: minimal size of the receptive field.
        :type min_diam_rf: float

        :return: weight of a pixel with respect to a circuit.
        :rtype: float
        """

        diam_rf = max(slope * eccentricity + intercept, min_diam_rf)
        std = diam_rf / 4.0
        return exp(-euclidian_dist_R2(pixel, center) / (2 * std ** 2))

    def _compute_local_contrasts(
            self, circuits: list[StimulusCircuit], slope: float, intercept: float, min_diam_rf: float
    ) -> list[float]:
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
            eccentricity = self._eccentricity_in_patch(point=circuit.center)

            num = 0
            denum = 0
            for pix in circuit.pixels:
                weight = self._get_weight(
                    center=circuit.center,
                    pixel=pix,
                    eccentricity=eccentricity,
                    slope=slope,
                    intercept=intercept,
                    min_diam_rf=min_diam_rf
                )
                num += weight * (self.stimulus_patch[pix[0]][pix[1]] - mean_luminance) ** 2 / mean_luminance
                denum += weight

            local_contrast = sqrt(num / denum)
            local_contrasts.append(local_contrast)

        return local_contrasts

    def _compute_frequencies(self, local_contrasts: list[float]) -> np.ndarray[int, float]:
        """
        Computes oscillation frequencies of the circuit through local contrasts.

        :param local_contrasts: list containing local contrast values for each circuit.
        :type local_contrasts: list[float]

        :return: list containing oscillation frequencies for each circuit.
        :rtype: numpy.ndarray[int, float]
        """

        return 25 + 0.25 * np.array(local_contrasts)

    def _compute_current(self, frequencies: np.ndarray[int, float]) -> np.ndarray[int, float]:
        """
        Computes _current through oscillation frequencies. ARTIFICIAL FUNCTION - REAL NOT IMPLEMENTED YET.

        :param frequencies: list containing oscillation frequencies for each circuit.
        :type frequencies: list[float]

        :return: TODO
        :rtype: numpy.ndarray[int, float]
        """

        # TODO:: implement the real strategy
        return 100.0 / (0.5 + 0.5 * np.array(frequencies))
