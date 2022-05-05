from enum import Enum


class NeuronTypes(Enum):
    """
    Enum class containing neuron types: excitatory and inhibitory.
    """

    EX = "excitatory"
    IN = "inhibitory"
    