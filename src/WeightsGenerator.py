from misc import *

import numpy as np
from math import exp


class WeightsGenerator:
    """
    TODO
    """

    def __init__(self, N, mu_x, mu_y, offset, slope, intercept, theta, side_length, contrast_res):
        self.N = N  # TODO:: number of ping_networks (small squares)?
        self.mu_x = mu_x  # TODO + where do we get these from? - pixel location relative to (0, 0)
        self.mu_y = mu_y
        self.offset = offset
        self.slope = slope
        self.intercept = intercept
        self.theta = theta
        self.side_length = side_length
        self.contrast_res = contrast_res

    def generate(self):
        """
        TODO
        :return:
        """

        total_pix = self.contrast_res**2

        eccentricity = euclidian_dist_R2(self.mu_x, self.mu_y)
        rf_diameter = max(self.slope * eccentricity + self.intercept, self.theta)

        sigma = rf_diameter / 4.0
        r = np.linspace(
            0,
            self.side_length,
            num=self.contrast_res,
            endpoint=True
        )
        x, y = np.meshgrid(r, r)
        # TODO:: why are we flipping it?
        y = np.flipud(y)

        # TODO:: what do we need the offset for?
        x += self.offset
        y += self.offset

        weights = np.zeros((total_pix, self.N))

        for i in range(self.N):
            weights[:, i] = exp(
                np.divide(
                    -(np.power(x - self.mu_x[i], 2) + np.power(y - self.mu_y[i], 2)),
                    2 * sigma[i]**2
                )
            )

        # TODO:: why do we do this?
        D = np.diag(np.sum(weights, axis=0))
        weights = np.dot(weights, D).T

        return weights


