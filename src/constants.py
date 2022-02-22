from src.NeuronTypes import *

NR_NEURONS = {
    NeuronTypes.E: 8,
    NeuronTypes.I: 4
}

NR_OSCILLATORS = 4

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

# for timescale of recovery variable recovery
IZHI_ALPHA = {
    NeuronTypes.E: 0.02,
    NeuronTypes.I: 0.1
}

# for sensitivity of recovery to sub-threshold oscillations of potential
IZHI_BETA = {
    NeuronTypes.E: 0.2,
    NeuronTypes.I: 0.2
}

# for membrane voltage after spike (after-spike reset of potential)
IZHI_GAMMA = {
    NeuronTypes.E: -65,
    NeuronTypes.I: -65
}

# for after-spike reset of recovery variable recovery
IZHI_ZETA = {
    NeuronTypes.E: 8,
    NeuronTypes.I: 2
}

# for initial values of potential = voltage (membrane potential)
INIT_MEMBRANE_POTENTIAL = -65

"""Gaussian input"""

GAUSSIAN_INPUT = {
    NeuronTypes.E: 1.5,
    NeuronTypes.I: 1.5
}

"""Synaptic constants"""

# TODO
SYNAPTIC_CONST_RISE = {
    NeuronTypes.E: 0.1,
    NeuronTypes.I: 0.1
}

# TODO
SYNAPTIC_CONST_DECAY = {
    NeuronTypes.E: 1.0,
    NeuronTypes.I: 4.0
}