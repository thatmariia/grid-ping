


class IzhikevichNetworkOutcome:
    """
    This class contains the collected information from the simulation of the Izhikevich network.

    :param firing_times: indices of spikes.
    :type firing_times: TODO
    """

    def __init__(self, firing_times):
        self.firing_times = firing_times