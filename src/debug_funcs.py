import numpy as np
from os.path import exists
from os import makedirs


USE_GRAY_STIMULUS = False
USE_CALCULATED_FC_RELATIONSHIP = False


# def try_pulling_stimulus_data(params_gabor, params_rf, params_ping, params_izhi, params_freqs):
#     folder_path = f"../local_storage/stimulus/{params_ping.nr_ping_networks}/"
#     if exists(folder_path):
#         stimulus_currents = np.loadtxt(folder_path + "currents.txt")
#         cortical_distances = np.loadtxt(folder_path + "distances.txt").reshape(
#             (params_ping.nr_ping_networks, params_ping.nr_ping_networks)
#         )
#     else:
#         print("Stimulus data does not exist. Creating stimulus data.")
#         stimulus = StimulusFactory().create(params_gabor, params_rf, params_ping, params_izhi, params_freqs)
#         stimulus_currents = stimulus.stimulus_currents
#         cortical_distances = stimulus.extract_stimulus_location().cortical_distances
#
#         makedirs(folder_path)
#         np.savetxt(folder_path + "currents.txt", stimulus_currents)
#         np.savetxt(folder_path + "distances.txt", cortical_distances)
#
#     return stimulus_currents, cortical_distances