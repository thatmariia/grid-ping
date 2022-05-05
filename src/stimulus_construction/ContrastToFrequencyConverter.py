import numpy as np


class ContrastToFrequencyConverter:
    """
    This class converts the local contrasts stimulus into the frequencies stimulus.
    """

    def convert(self, stimulus_contrast: np.ndarray[int, float]) -> np.ndarray[int, float]:
        """
        Converts the local contrasts stimulus into the frequencies stimulus.

        The approach is derived from :cite:p:`MaryamPLACEHOLDER`.

        :param stimulus_contrast: local contrasts stimulus.
        :type stimulus_contrast: numpy.ndarray[int, float]

        :return: the stimulus converted to frequencies.
        :rtype: numpy.ndarray[int, float]
        """

        return self._compute_frequencies(stimulus_contrast)

    def _compute_frequencies(self, stimulus_contrast: np.ndarray[int, float]) -> np.ndarray[int, float]:
        """
        Computes oscillation frequencies of the circuit through local contrasts.

        :param stimulus_contrast: list containing local contrast values for each circuit.
        :type stimulus_contrast: numpy.ndarray[int, float]

        :return: list containing oscillation frequencies for each circuit.
        :rtype: numpy.ndarray[int, float]
        """

        return 25 + 0.25 * np.array(stimulus_contrast)