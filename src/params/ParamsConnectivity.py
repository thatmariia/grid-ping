from src.params.NeuronTypes import *


class ParamsConnectivity:
    """
     This class contains parameters of the network connectivity.

     :param max_connect_strength_EE: maximum connection strength between excitatory neurons.
     :type max_connect_strength_EE: float

     :param max_connect_strength_EI: maximum connection strength between excitatory and inhibitory neurons.
     :type max_connect_strength_EI: float

     :param max_connect_strength_IE: maximum connection strength between inhibitory and excitatory neurons.
     :type max_connect_strength_IE: float

     :param max_connect_strength_II: maximum connection strength between inhibitory neurons.
     :type max_connect_strength_II: float

     :param spatial_const_EE: spatial constant between excitatory neurons.
     :type spatial_const_EE: float

     :param spatial_const_EI: spatial constant between excitatory and inhibitory neurons.
     :type spatial_const_EI: float

     :param spatial_const_IE: spatial constant between inhibitory and excitatory neurons.
     :type spatial_const_IE: float

     :param spatial_const_II: spatial constant between inhibitory neurons.
     :type spatial_const_II: float

     :ivar max_connect_strength: maximum connectivity strengths.
     :ivar spatial_const: spatial constants.
    """

    def __init__(
            self,
            max_connect_strength_EE: float,
            max_connect_strength_EI: float,
            max_connect_strength_IE: float,
            max_connect_strength_II: float,
            spatial_const_EE: float=0.5,
            spatial_const_EI: float=0.5,
            spatial_const_IE: float=0.5,
            spatial_const_II: float=0.5
    ):
        self.max_connect_strength: dict[tuple[NeuronTypes, NeuronTypes], float] = {
            (NeuronTypes.EX, NeuronTypes.EX): max_connect_strength_EE,
            (NeuronTypes.EX, NeuronTypes.IN): max_connect_strength_EI,
            (NeuronTypes.IN, NeuronTypes.EX): max_connect_strength_IE,
            (NeuronTypes.IN, NeuronTypes.IN): max_connect_strength_II
        }

        self.spatial_const: dict[tuple[NeuronTypes, NeuronTypes], float] = {
            (NeuronTypes.EX, NeuronTypes.EX): spatial_const_EE,
            (NeuronTypes.EX, NeuronTypes.IN): spatial_const_EI,
            (NeuronTypes.IN, NeuronTypes.EX): spatial_const_IE,
            (NeuronTypes.IN, NeuronTypes.IN): spatial_const_II
        }