import numpy as np


class SyncEvaluation:

    def __init__(self, phase_values, phase_locking):
        self.phase_locking = phase_locking
        self.avg_phase_locking = np.mean(phase_locking)
        self.phase_values = phase_values
