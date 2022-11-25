from src.params.ParamsContrastToCurrent import ParamsContrastToCurrent

import numpy as np


class ContrastToFrequencyConverter:
    """
    This class converts the local contrasts stimulus into the frequencies stimulus.
    """

    def convert(self, stimulus_contrast: np.ndarray[int, float], params_c2f: ParamsContrastToCurrent) -> np.ndarray[int, float]:
        """
        Converts the local contrasts stimulus into the frequencies stimulus.

        The approach is derived from :cite:p:`MaryamPLACEHOLDER`.

        :param stimulus_contrast: local contrasts stimulus.
        :type stimulus_contrast: numpy.ndarray[int, float]

        :return: the stimulus converted to frequencies.
        :rtype: numpy.ndarray[int, float]
        """

        return self._compute_frequencies(stimulus_contrast, params_c2f)

    def _compute_frequencies(self, stimulus_contrast: np.ndarray[int, float], params_c2f: ParamsContrastToCurrent) -> np.ndarray[int, float]:
        """
        Computes oscillation frequencies of the circuit through local contrasts.

        :param stimulus_contrast: list containing local contrast values for each circuit.
        :type stimulus_contrast: numpy.ndarray[int, float]

        :return: list containing oscillation frequencies for each circuit.
        :rtype: numpy.ndarray[int, float]
        """

        return params_c2f.offset + params_c2f.slope * np.array(stimulus_contrast)
        #return 24.9 + 0.35 * np.array(stimulus_contrast)