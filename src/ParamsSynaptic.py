from src.Params import *

from src.NeuronTypes import *


class ParamsSynaptic(Params):
    """
    This class contains synaptic parameters.

    * rise - rise time;
    * decay - decay time;
    * conductance - synaptic conductance density;
    * reversal_potential - reversal (equilibrium) potentials.

    :type rise_E: float
    :type decay_E: float
    :type rise_I: float
    :type decay_I: float
    :type conductance_EE: float
    :type conductance_EI: float
    :type conductance_IE: float
    :type conductance_II: float
    :type reversal_potential_E: float
    :type reversal_potential_I: float

    :ivar rise: rise time.
    :ivar decay: decay time.
    :ivar conductance: synaptic conductance density.
    :ivar reversal_potential: reversal (equilibrium) potentials.
    """

    def __init__(
            self,
            rise_E: float,
            decay_E: float,
            rise_I: float,
            decay_I: float,
            conductance_EE: float=0.6,
            conductance_EI: float=0.06,
            conductance_IE: float=0.8,
            conductance_II: float=0.5,
            reversal_potential_E: float=-80,
            reversal_potential_I: float=0
    ):

        self.rise: dict[NeuronTypes, float] = {
            NeuronTypes.E: rise_E,
            NeuronTypes.I: rise_I,
        }
        self.decay: dict[NeuronTypes, float] = {
            NeuronTypes.E: decay_E,
            NeuronTypes.I: decay_I,
        }
        self.conductance: dict[tuple[NeuronTypes, NeuronTypes], float] = {
            (NeuronTypes.E, NeuronTypes.E): conductance_EE,
            (NeuronTypes.E, NeuronTypes.I): conductance_EI,
            (NeuronTypes.I, NeuronTypes.E): conductance_IE,
            (NeuronTypes.I, NeuronTypes.I): conductance_II
        }
        self.reversal_potential: dict[NeuronTypes, float] = {
            NeuronTypes.E: reversal_potential_E,
            NeuronTypes.I: reversal_potential_I
        }