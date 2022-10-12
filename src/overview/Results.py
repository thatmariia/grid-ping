from src.plotter.results_plots import *

import pandas as pd
from copy import deepcopy
import scipy.interpolate

class Results:

    def __init__(self, dist_scales, contrast_ranges):
        dist_scales_indices = [self._format_ic(dist_scale) for dist_scale in dist_scales]
        contrast_ranges_columns = [self._format_ic(contrast_range) for contrast_range in contrast_ranges]
        empty_df = pd.DataFrame(index=dist_scales_indices, columns=contrast_ranges_columns, dtype=float)
        self.avg_phase_lockings_df = deepcopy(empty_df)
        self.frequency_stds_df = deepcopy(empty_df)

    def add_results(self, dist_scale, contrast_range, avg_phase_locking, frequency_std):
        self.avg_phase_lockings_df.at[self._format_ic(dist_scale), self._format_ic(contrast_range)] = avg_phase_locking
        self.frequency_stds_df.at[self._format_ic(dist_scale), self._format_ic(contrast_range)] = frequency_std

    def make_plots(self):
        plot_frequencies_std(self.frequency_stds_df)
        plot_avg_phase_lockings(self.avg_phase_lockings_df)
        plot_avg_phase_lockings(self._interpolate_df(self.avg_phase_lockings_df), smooth=True)

    def _format_ic(self, x):
        return f"{x:.3f}"

    def _interpolate_df(self, df, interpolation_len=30): #991
        columns = [float(i) for i in df.columns.tolist()]
        indices = [float(i) for i in df.index.tolist()]

        arr = df.to_numpy()
        interpolation = scipy.interpolate.RectBivariateSpline(indices, columns, arr)

        columns_interpolated = np.linspace(columns[0], columns[-1], num=interpolation_len)
        indices_interpolated = np.linspace(indices[0], indices[-1], num=interpolation_len)

        return interpolation(indices_interpolated, columns_interpolated)
