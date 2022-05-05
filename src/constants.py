from src.NeuronTypes import *

"""Synaptic constants"""

# rise time
SYNAPTIC_RISE = {
    NeuronTypes.E: 1,
    NeuronTypes.I: 2,
}

# decay time
SYNAPTIC_DECAY = {
    NeuronTypes.E: 2.4,
    NeuronTypes.I: 20,
}

# synaptic conductance density
SYNAPTIC_CONDUCTANCE = {
    (NeuronTypes.E, NeuronTypes.E): 0.6,
    (NeuronTypes.E, NeuronTypes.I): 0.06,
    (NeuronTypes.I, NeuronTypes.E): 0.8,
    (NeuronTypes.I, NeuronTypes.I): 0.5
}

# reversal (equilibrium) potentials
REVERSAL_POTENTIAL = {
    NeuronTypes.E: -80,
    NeuronTypes.I: 0
}