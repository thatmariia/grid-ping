from src.NeuronTypes import *
from src.Connectivity import *

from abc import ABC, abstractmethod


class CurrentComponents(ABC):
    """
    This lass contains methods of computing the neural network current components.

    In the network, neurons are not isolated, and the current involves the accumulated effect of interactions
    with other neurons:

    :math:`I_v = \\sum_{w \in V} K_{v, w} I_{\mathrm{syn}, w} + I_{\mathrm{stim}, v} \\mathbb{1} \{ \mathsf{type}(v) = \mathrm{ex} \}`,

    where
    * :math:`K` is the coupling weights (see :obj:`Connectivity`),
    * :math:`I_{syn}` represents the effect of synaptic potentials,
    * :math:`I_{stim}` is the current caused by external stimuli.

    :param connectivity: information about connectivity between neurons in the oscillatory network.
    :type connectivity: Connectivity

    :ivar connectivity: information about connectivity between neurons in the oscillatory network.
    """

    def __init__(self, connectivity: Connectivity):
        self.connectivity: Connectivity = connectivity

    @abstractmethod
    def get_synaptic_currents(self, gatings, dt, potentials):
        """
        Computes :math:`I_{syn}`.
        """

        pass

    @abstractmethod
    def get_current_input(self):
        """
        Computes :math:`I_{stim}`.
        """

        pass