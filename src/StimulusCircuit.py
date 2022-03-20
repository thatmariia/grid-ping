
class StimulusCircuit:
    """
    Class containing information about a single _stimulus patch circuit.

    :param center: coordinates of the center of the circuit within the _stimulus patch.
    :type center: tuple(float, float)

    :param pixels: list of pixel coordinates that belong to the circuit.
    :type pixels: list[tuple(int, int)]

    :ivar center: coordinates of the center of the circuit within the _stimulus patch.
    :type center: tuple(float, float)

    :ivar pixels: list of pixel coordinates that belong to the circuit.
    :type pixels: list[tuple(int, int)]
    """

    def __init__(self, center, pixels):
        self.center = center
        self.pixels = pixels
