from src.NeuronTypes import *

import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
from math import ceil


def plot_binary_heatmap(im: list[list[float]], path: str) -> None:
    """
    Plots a heatmap with the binary color scheme

    :param im: matrix to plot
    :type im: list[list[float]]

    :param path: path for saving the resulting plot
    :type path: str

    :rtype: None
    """

    fig, ax = plt.subplots(figsize=(300, 300))
    sns.heatmap(
        im,
        annot=False,
        vmin=0,
        vmax=1,
        cmap="binary",
        square=True,
        cbar=False,
        xticklabels=False,
        yticklabels=False,
        ax=ax
    )
    # TODO:: remove white outer borders
    fig.savefig(path)


def neur_slice(neuron_type: NeuronTypes, nr_ex: int, nr_in: int) -> slice:
    """
    By design, we put excitatory neurons in lists before inhibitory. This function returns relevant slices.

    :param neuron_type: the type of neuron we need indices for.
    :type neuron_type: NeuronTypes

    :param nr_ex: number of excitatory neurons.
    :type nr_ex: int

    :param nr_in: number of inhibitory neurons.
    :type nr_in: int

    :return: a slice for the neurons of given type.
    :rtype: slice
    """

    if neuron_type == NeuronTypes.E:
        return slice(nr_ex)
    return slice(nr_ex, nr_ex + nr_in)


def neur_type(id: int, nr_ex: int) -> NeuronTypes:
    """
    Returns the type of the neuron at a particular ID.

    :param id: the ID of the neuron.
    :type id: int

    :param nr_ex: number of excitatory neurons in a network, where the neuron is located.
    :type nr_ex: int

    :return: the neuron type.
    :rtype: NeuronTypes
    """

    if id < nr_ex:
        return NeuronTypes.E
    return NeuronTypes.I


def add_points(points: tuple, coeffs=None) -> tuple:
    """
    Adds values in tuples (for adding coordinates).

    :param points: list of tuples to add together.
    :type points: list[tuple]

    :param coeffs: coefficients before the tuples (all 1's by default).
    :type coeffs: list[float]

    :raises:
        AssertionError: if the number of tuples and coefficients are not equal.
    :raises:
        AssertionError: if the number of values is not equal in all tuples.

    :return: the sum of tuples.
    :rtype: tuple
    """

    if coeffs != None:
        assert len(points) == len(coeffs), "Number of points and number of coefficients must be equal."
    else:
        coeffs = [1] * len(points)

    tuple_length = len(points[0])
    assert all(len(p) == tuple_length for p in points), "Number of values should be equal for all points."

    points_t = [
        tuple([points[p][i] * coeffs[p] for p in range(len(points))]) for i in range(tuple_length)
    ]
    return tuple(map(sum, points_t))


def point_ceil(p: tuple) -> tuple:
    """
    Computes the ceiling of a tuple.

    :param p: a tuple.
    :type p: tuple

    :return: tuple of ceilings of all values in the given tuple.
    :rtype: tuple
    """

    return tuple(ceil(i) for i in p)


def euclidian_dist_R2(p1: tuple[float, float], p2=(0, 0)) -> float:
    """
    Calculates the Eaclidian distance between two 2D points.

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

