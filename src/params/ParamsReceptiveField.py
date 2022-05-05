from src.params.Params import *


class ParamsReceptiveField(Params):
    """
    This class contains parameters for the receptive field.

    :param slope: slope of the receptive field size.
    :type slope: float

    :param intercept: intercept of the receptive field size.
    :type intercept: float

    :param min_diam_rf: minimal size of the receptive field.
    :type min_diam_rf: float


    :raises:
        AssertionError: if the minimal diameter of the receptive field is not larger than 0.


    :ivar slope: slope of the receptive field size.
    :ivar intercept: intercept of the receptive field size.
    :ivar min_diam_rf: minimal size of the receptive field.
    """

    def __init__(self, slope: float, intercept: float, min_diam_rf: float):

        assert min_diam_rf > 0, \
            "The minimal diameter_dg of the receptive field should be larger than 0."

        self.slope = slope
        self.intercept = intercept
        self.min_diam_rf = min_diam_rf
