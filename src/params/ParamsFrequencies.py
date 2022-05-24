import numpy as np
from math import floor, pi

class ParamsFrequencies:

    def __init__(
            self,
            frequency_range: range,
            gaussian_width: float
    ):
        self.frequencies = list(frequency_range)
        # set the width of the Gaussian
        self.gaussian_width = gaussian_width
        self.wt = np.linspace(-1, 1, floor(2 / 0.001) + 1)
        # half the size of the wavelet
        self.half_wave_size = floor((len(self.wt) - 1) / 2)

        g = np.exp(-np.power(self.wt, 2) / (2 * gaussian_width ** 2))
        complex_sines = [np.exp(1j * 2 * pi * f * self.wt) for f in self.frequencies]
        self.complex_wavelets = [complex_sine * g for complex_sine in complex_sines]

