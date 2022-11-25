from src.params.ParamsContrastToCurrent import ParamsContrastToCurrent

import numpy as np


class ContrastToCurrentConverter:

    def convert(self, stimulus_contrast: np.ndarray[int, float], params_c2c: ParamsContrastToCurrent):
        return params_c2c.min_current + (params_c2c.max_current - params_c2c.min_current) * np.array(stimulus_contrast)
