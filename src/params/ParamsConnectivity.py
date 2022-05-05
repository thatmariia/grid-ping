from src.params.Params import *

from src.NeuronTypes import *


class ParamsConnectivity(Params):
    """
     This class contains parameters of the network connectivity.

     * max_connect_strength - maximum connection strength between neuron types;
     * spatial_consts - spatial constant between neuron types;

     :type max_connect_strength_EE: float
     :type max_connect_strength_EI: float
     :type max_connect_strength_IE: float
     :type max_connect_strength_II: float

     :type spatial_consts_EE: float
     :type spatial_consts_EI: float
     :type spatial_consts_IE: float
     :type spatial_consts_II: float

     :ivar max_connect_strength: maximum connectivity strengths.
     :iver spatial_consts: spatial constants.
    """

    def __init__(
            self,
            max_connect_strength_EE: float,
            max_connect_strength_EI: float,
            max_connect_strength_IE: float,
            max_connect_strength_II: float,
            spatial_consts_EE: float=0.4,
            spatial_consts_EI: float=0.3,
            spatial_consts_IE: float=0.3,
            spatial_consts_II: float=0.3
    ):
        self.max_connect_strength: dict[tuple[NeuronTypes, NeuronTypes], float] = {
            (NeuronTypes.EX, NeuronTypes.EX): max_connect_strength_EE,
            (NeuronTypes.EX, NeuronTypes.IN): max_connect_strength_EI,
            (NeuronTypes.IN, NeuronTypes.EX): max_connect_strength_IE,
            (NeuronTypes.IN, NeuronTypes.IN): max_connect_strength_II
        }

        self.spatial_consts: dict[tuple[NeuronTypes, NeuronTypes], float] = {
            (NeuronTypes.EX, NeuronTypes.EX): spatial_consts_EE,
            (NeuronTypes.EX, NeuronTypes.IN): spatial_consts_EI,
            (NeuronTypes.IN, NeuronTypes.EX): spatial_consts_IE,
            (NeuronTypes.IN, NeuronTypes.IN): spatial_consts_II
        }