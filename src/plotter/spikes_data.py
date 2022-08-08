from src.plotter.setup import DataNames, DataPaths, DATA_FORMAT

import numpy as np


def save_spikes_data(spikes_data):
    path = f"{DataPaths.SPIKES_DATA.value}/{DataNames.SPIKES_DATA.value}{DATA_FORMAT}"

    np.savetxt(path, spikes_data)


def fetch_spikes_data():
    path = f"{DataPaths.SPIKES_DATA.value}/{DataNames.SPIKES_DATA.value}{DATA_FORMAT}"

    spikes_data = np.loadtxt(path)
    return spikes_data
