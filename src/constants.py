from src.NeuronTypes import *

"""Network connectivity constants"""

MAX_CONNECT_STRENGTH = {
    (NeuronTypes.E, NeuronTypes.E): 0.004,
    (NeuronTypes.E, NeuronTypes.I): 0.07,
    (NeuronTypes.I, NeuronTypes.E): -0.04,
    (NeuronTypes.I, NeuronTypes.I): -0.015
}

SPATIAL_CONST = {
    (NeuronTypes.E, NeuronTypes.E): 0.4,
    (NeuronTypes.E, NeuronTypes.I): 0.3,
    (NeuronTypes.I, NeuronTypes.E): 0.3,
    (NeuronTypes.I, NeuronTypes.I): 0.3
}

"""Izhikevich neuron params"""

THRESHOLD_POTENTIAL = 30

# for timescale of _recovery variable _recovery
IZHI_ALPHA = {
    NeuronTypes.E: 0.02,
    NeuronTypes.I: 0.1
}

# for sensitivity of _recovery to sub-threshold oscillations of _potentials
IZHI_BETA = {
    NeuronTypes.E: 0.2,
    NeuronTypes.I: 0.2
}

# for membrane voltage after spike (after-spike reset of _potentials)
IZHI_GAMMA = {
    NeuronTypes.E: -65,
    NeuronTypes.I: -65
}

# for after-spike reset of _recovery variable _recovery
IZHI_ZETA = {
    NeuronTypes.E: 8,
    NeuronTypes.I: 2
}

# for initial values of _potentials = voltage (membrane _potentials)
INIT_MEMBRANE_POTENTIAL = -65

"""Gaussian input"""

GAUSSIAN_INPUT = {
    NeuronTypes.E: 1.5,
    NeuronTypes.I: 1.5
}

"""Synaptic constants"""

# TODO
SYNAPTIC_RISE = {
    NeuronTypes.E: 1,
    NeuronTypes.I: 1,
}

# TODO
SYNAPTIC_DECAY = {
    NeuronTypes.E: 1,
    NeuronTypes.I: 1,
}

# TODO
CONDUCTANCE_DENSITY = {
    (NeuronTypes.E, NeuronTypes.E): 1,
    (NeuronTypes.E, NeuronTypes.I): 1,
    (NeuronTypes.I, NeuronTypes.E): 1,
    (NeuronTypes.I, NeuronTypes.I): 1
}

REVERSAL_POTENTIALS = {
    NeuronTypes.E: -80,
    NeuronTypes.I: 0
}