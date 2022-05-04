from src.NeuronTypes import *

import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
from math import ceil


def plot_binary_heatmap(im: np.ndarray[(int, int), float], path: str) -> None:
    """
    Plots a heatmap with the binary color scheme

    :param im: matrix to plot
    :type im: numpy.ndarray[(int, int), float]

    :param path: path for saving the resulting plot
    :type path: str

    :rtype: None
    """

    fig, ax = plt.subplots(figsize=(30, 30))

    sns.heatmap(
        im,
        annot=False,
        vmin=0,
        vmax=1,
        cmap="gist_gray",
        cbar=False,
        square=True,
        xticklabels=False,
        yticklabels=False,
        ax=ax
    )

    fig.savefig(path, bbox_inches='tight', pad_inches=0)


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

    :param p1: coordinates of point_pix 1.
    :type p1: tuple[float, float]

    :param p2: coordinates of point_pix 2.
    :type p2: tuple[float, float]

    :return: the Euclidean distance between two 2D points.
    :rtype: float
    """

    return math.sqrt(
        pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2)
    )
