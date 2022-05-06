from src.NeuronTypes import *


class ParamsSynaptic:
    """
    This class contains synaptic parameters.

    :param rise_E: rise time for excitatory presynaptic neurons.
    :type rise_E: float

    :param decay_E: decay time for excitatory presynaptic neurons.
    :type decay_E: float

    :param rise_I: rise time for inhibitory presynaptic neurons.
    :type rise_I: float

    :param decay_I: decay time for inhibitory presynaptic neurons.
    :type decay_I: float

    :param conductance_EE: synaptic conductance density between excitatory neurons.
    :type conductance_EE: float

    :param conductance_EI: synaptic conductance density between excitatory and inhibitory neurons.
    :type conductance_EI: float

    :param conductance_IE: synaptic conductance density between inhibitory and excitatory neurons.
    :type conductance_IE: float

    :param conductance_II: synaptic conductance density between inhibitory neurons.
    :type conductance_II: float

    :param reversal_potential_E: reversal potential of excitatory neurons.
    :type reversal_potential_E: float

    :param reversal_potential_I: reversal potential of inhibitory neurons.
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
            NeuronTypes.EX: rise_E,
            NeuronTypes.IN: rise_I,
        }
        self.decay: dict[NeuronTypes, float] = {
            NeuronTypes.EX: decay_E,
            NeuronTypes.IN: decay_I,
        }
        self.conductance: dict[tuple[NeuronTypes, NeuronTypes], float] = {
            (NeuronTypes.EX, NeuronTypes.EX): conductance_EE,
            (NeuronTypes.EX, NeuronTypes.IN): conductance_EI,
            (NeuronTypes.IN, NeuronTypes.EX): conductance_IE,
            (NeuronTypes.IN, NeuronTypes.IN): conductance_II
        }
        self.reversal_potential: dict[NeuronTypes, float] = {
            NeuronTypes.EX: reversal_potential_E,
            NeuronTypes.IN: reversal_potential_I
        }