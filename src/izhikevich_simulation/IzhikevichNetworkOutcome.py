from src.params.ParamsPING import *
from src.params.ParamsFrequencies import *
from src.izhikevich_simulation.GridGeometry import *

import numpy as np


class IzhikevichNetworkOutcome:
    """
    This class contains the collected information from the simulation of the Izhikevich network.

    :param spikes: indices of spikes.
    :type spikes: list[tuple[int, int]]

    TODO:: list other params

    :ivar spikes: indices of spikes.
    """

    def __init__(
            self,
            spikes: list[tuple[int, int]],
            params_ping: ParamsPING,
            params_freqs: ParamsFrequencies,
            simulation_time: int,
            grid_geometry: GridGeometry
    ):
        self.spikes: list[tuple[int, int]] = spikes
        self.params_ping = params_ping
        self.params_freqs = params_freqs
        self.simulation_time = simulation_time
        self.grid_geometry = grid_geometry
