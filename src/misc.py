import math
from math import ceil
import numpy as np


def cartesian_product(*arrays) -> np.ndarray:
    """
    Computes the cartesian product of given arrays fast.

    :param arrays: arrays to compute the cartesian product of.

    :return: the cartesian product of the given arrays.
    :rtype: numpy.ndarray
    """
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def multiply_point(point: tuple[float, ...], coef: float) -> tuple[float, ...]:
    """
    Multiples each value in a given tuple by a coefficient.

    :param point: the point to multiply.
    :type point: tuple[float, ...]

    :param coef: the coefficient to multiply the points by.
    :type coef: float

    :return: the tuple, where each element is multiplied by the coefficient.
    :rtype: tuple[float]
    """

    return tuple(p * coef for p in point)


def add_points(points: list[tuple[float, ...]], coefs=None) -> tuple[float, ...]:
    """
    Adds values in tuples (for adding coordinates).

    :param points: list of tuples to add together.
    :type points: list[tuple[float, ...]]

    :param coefs: coefficients before the tuples (all 1's by default).
    :type coefs: list[float]

    :raises:
        AssertionError: if the number of tuples and coefficients are not equal.
    :raises:
        AssertionError: if the number of values is not equal in all tuples.

    :return: the sum of tuples.
    :rtype: tuple[float, ...]
    """

    if coefs != None:
        assert len(points) == len(coefs), "Number of points and number of coefficients must be equal."
    else:
        coefs = [1] * len(points)

    tuple_length = len(points[0])
    assert all(len(p) == tuple_length for p in points), "Number of values should be equal for all points."

    points_t = [
        tuple([points[p][i] * coefs[p] for p in range(len(points))]) for i in range(tuple_length)
    ]
    return tuple(map(sum, points_t))


def point_ceil(p: tuple[float, ...]) -> tuple[int, ...]:
    """
    Computes the ceiling of a tuple.

    :param p: a tuple.
    :type p: tuple[float, ...]

    :return: tuple of ceilings of all values in the given tuple.
    :rtype: tuple[int, ...]
    """

    return tuple(ceil(i) for i in p)


def euclidian_dist(p1: tuple[float, float], p2=(0, 0)) -> float:
    """
    Calculates the Euclidian distance between two points.

    :param p1: coordinates of point 1.
    :type p1: tuple[float, float]

    :param p2: coordinates of point 2.
    :type p2: tuple[float, float]

    :return: the Euclidean distance between two 2D points.
    :rtype: float
    """

    return math.sqrt(
        pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2)
    )
