from src.plotter.setup import DataNames, DataPaths, DATA_FORMAT

import numpy as np

####################################################################################################

def save_spikes_data(spikes_data):
    path = f"{DataPaths.SPIKES_DATA.value}/{DataNames.SPIKES_DATA.value}{DATA_FORMAT}"

    np.savetxt(path, spikes_data)


def fetch_spikes_data():
    path = f"{DataPaths.SPIKES_DATA.value}/{DataNames.SPIKES_DATA.value}{DATA_FORMAT}"

    spikes_data = np.loadtxt(path)
    return spikes_data


####################################################################################################


def save_cortical_dist_data(dist_data):
    path = f"{DataPaths.CORTICAL_DISTANCES_DATA.value}/{DataNames.CORTICAL_DISTANCES_DATA.value}{DATA_FORMAT}"

    np.savetxt(path, dist_data)

def fetch_cortical_dist_data():
    path = f"{DataPaths.CORTICAL_DISTANCES_DATA.value}/{DataNames.CORTICAL_DISTANCES_DATA.value}{DATA_FORMAT}"

    dist_data = np.loadtxt(path)
    return dist_data

