


class IzhikevichNetworkOutcome:
    """
    This class contains the collected information from the simulation of the Izhikevich network.

    :param spikes: indices of spikes.
    :type spikes: list[tuple[int, int]]

    :ivar spikes: indices of spikes.
    """

    def __init__(self, spikes):
        self.spikes: list[tuple[int, int]] = spikes