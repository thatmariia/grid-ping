from src.GaborLuminanceStimulus import *
from src.StimulusCircuit import *
from src.StimulusLocations import *

import numpy as np
from math import sqrt, exp
from itertools import product
from statistics import mean
from tqdm import tqdm


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

    :ivar _circuits: list of info about circuits.
    :type _circuits: list[StimulusCircuit]

    :ivar current: list of currents produced by respective circuits in the stimulus.
    :type current: numpy.ndarray[int, float]
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

        self._circuits: list[StimulusCircuit] = self._assign_circuits()
        self.current = self._get_input_current(slope, intercept, min_diam_rf)

    def _assign_circuits(self) -> list[StimulusCircuit]:
        """
        Creates circuits and assigns centers and pixels of the stimulus patch to them.

        :return: list of all circuits of the stimulus patch created by applying a lattice.
        :rtype: list[StimulusCircuit]
        """

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
                    pixels=pixels,
                    atopix=self._atopix,
                    grid_index=(i, j)
                )
                circuits.append(circuit)

        return circuits

    def extract_stimulus_location(self) -> StimulusLocations:
        """
        Computes the location info of the stimulus patch and, thus, the PING networks, namely eccentricity and
        angle of each PING network.

        :return: location info of the network.
        :rtype: StimulusLocations
        """

        grid_side = int(math.sqrt(len(self._circuits)))
        eccentricities = np.zeros((grid_side, grid_side))
        angles = np.zeros((grid_side, grid_side))

        for circuit in (pbar := tqdm(self._circuits)):
            pbar.set_description("Coordinates conversion")

            i = circuit.grid_index[0]
            j = circuit.grid_index[1]

            eccentricities[i, j] = self._eccentricity_in_patch(point=circuit.center_dg)
            angles[i, j] = self._angle_in_patch(point=circuit.center_dg)

        stim_locations = StimulusLocations(
            eccentricities=eccentricities,
            angles=angles
        )

        return stim_locations

    def _get_input_current(
            self, slope: float, intercept: float, min_diam_rf: float
    ) -> np.ndarray[int, float]:
        """
        Performs all the necessary steps to transform luminance to current.

        :param slope: slope of the receptive field size.
        :type slope: float

        :param intercept: intercept of the receptive field size.
        :type intercept: float

        :param min_diam_rf: minimal size of the receptive field.
        :type min_diam_rf: float

        :return: list containing currents created by each circuit.
        :rtype: numpy.ndarray[int, float]
        """

        local_contrasts = self._compute_local_contrasts(slope, intercept, min_diam_rf)
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
        return exp(-euclidian_dist(pixel, center) / (2 * std ** 2))

    def _point_in_stimulus(self, point: tuple[float, float]) -> tuple[float, ...]:
        """
        Calculates the coordinate of a given point in the patch within the stimulus.

        :param point: coordinates of the point within the patch in degrees.
        :type point: tuple[float, float]

        :return: coordinates of the point within the stimulus in degrees.
        :rtype: tuple[float, float]
        """

        return add_points([
            point,
            (self._patch_start[0] / self._atopix, self._patch_start[1] / self._atopix)
        ])

    def _angle_in_patch(self, point: tuple[float, float]) -> float:
        """
        Calculates the angle between the horizontal axis and the line passing through the center of the stimulus and a
        given point within the patch.

        :param point: coordinates of the point within the patch in degrees.
        :type point: tuple[float, float]

        :return: angle of the point.
        :rtype: float
        """

        point_in_stimulus = self._point_in_stimulus(point=point)
        stimulus_center_dg = (0.5 * self._full_height / self._atopix, 0.5 * self._full_width / self._atopix)

        new_point = add_points([
            point_in_stimulus, stimulus_center_dg
        ], [1, -1])

        angle = np.arctan(
            new_point[1] / new_point[0]
        )
        return angle

    def _eccentricity_in_patch(self, point: tuple[float, float]) -> float:
        """
        Calculates eccentricity at the given point within the patch.

        :param point: coordinates of the point within the patch in degrees.
        :type point: tuple[float, float]

        :return: eccentricity in degrees.
        :rtype: float
        """

        point_in_stimulus = self._point_in_stimulus(point=point)
        stimulus_center_dg = (0.5 * self._full_height / self._atopix, 0.5 * self._full_width / self._atopix)

        ecc = euclidian_dist(
            stimulus_center_dg,
            point_in_stimulus
        )
        return ecc

    def _compute_local_contrasts(
            self, slope: float, intercept: float, min_diam_rf: float
    ) -> list[float]:
        """
        Computes local contrasts for each circuit.

        :param slope: slope of the receptive field size.
        :type slope: float

        :param intercept: intercept of the receptive field size.
        :type intercept: float

        :param min_diam_rf: minimal size of the receptive field.
        :type min_diam_rf: float

        :return: list containing local contrast values for each circuit.
        :rtype: list[float]
        """

        mean_luminance = mean(np.array(self.stimulus).flatten())
        local_contrasts = []

        for curr_circuit in (pbar := tqdm(self._circuits)):
            pbar.set_description("Local contrast computation")

            eccentricity = self._eccentricity_in_patch(point=curr_circuit.center_dg)
            num = 0
            denum = 0

            for circuit in self._circuits:
                for i in range(len(circuit.pixels)):
                    pix = circuit.pixels[i]
                    pix_dg = circuit.pixels_dg[i]

                    weight = self._get_weight(
                        center=curr_circuit.center_dg,
                        pixel=pix_dg,
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
        Computes _currents through oscillation frequencies. ARTIFICIAL FUNCTION - REAL NOT IMPLEMENTED YET.

        :param frequencies: list containing oscillation frequencies for each circuit.
        :type frequencies: list[float]

        :return: TODO
        :rtype: numpy.ndarray[int, float]
        """

        # TODO:: implement the real strategy
        return 100.0 / (0.5 + 0.5 * np.array(frequencies))
