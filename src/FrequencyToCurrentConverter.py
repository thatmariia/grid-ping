import numpy as np


class FrequencyToCurrentConverter:
    """
    This class converts the frequencies stimulus into the currents stimulus.
    TODO:: implement
    """

    def convert(self, stimulus_frequencies: np.ndarray[int, float]) -> np.ndarray[int, float]:
        """
        Converts the frequencies stimulus into the currents stimulus.

        TODO:: how do I cite this?

        :param stimulus_frequencies: frequencies stimulus.
        :type stimulus_frequencies: numpy.ndarray[int, float]

        :return: the stimulus converted to currents.
        :rtype: numpy.ndarray[int, float]
        """
        return self._compute_current(stimulus_frequencies)

    def _compute_current(self, stimulus_frequencies: np.ndarray[int, float]) -> np.ndarray[int, float]:
        """
        Computes _currents through oscillation frequencies. ARTIFICIAL FUNCTION - REAL NOT IMPLEMENTED YET.

        :param stimulus_frequencies: list containing oscillation frequencies for each circuit.
        :type stimulus_frequencies: list[float]

        :return: TODO
        :rtype: numpy.ndarray[int, float]
        """

        # TODO:: implement the real strategy
        return 100.0 / (0.5 + 0.5 * np.array(stimulus_frequencies))