from src.constants import *
from src.NeuronTypes import *

import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt

def plot_binary_heatmap(im, path):
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
    fig.savefig(path)

def neur_slice(neuron_type, nr_ex, nr_in):
    """
    The setup: we put excitatory neurons in lists before inhibitory. This function returns relevant slices.

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


def euclidian_dist_R2(p1, p2):
    """
    Calculates the Eaclidian distance between two 2D points.

    :param p1: coordinates of point 1.
    :type p1: tuple[float]

    :param p2: coordinates of point 2.
    :type p2: tuple[float]

    :return: the Euclidean distance between two 2D points.
    :rtype: float
    """

    return math.sqrt(
        pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2)
    )


def cust_range(*args, rtol=1e-05, atol=1e-08, include=[True, False]):
    """
    Code taken from https://stackoverflow.com/questions/50299172/python-range-or-numpy-arange-with-end-limit-include

    Combines numpy.arange and numpy.isclose to mimic open, half-open and closed intervals.
    Avoids also floating point rounding errors as with
    >>> numpy.arange(1, 1.3, 0.1)
    array([1. , 1.1, 1.2, 1.3])

    args: [start, ]stop, [step, ]
        as in numpy.arange
    rtol, atol: floats
        floating point tolerance as in numpy.isclose
    include: boolean list-like, length 2
        if start and end point are included
    """
    # process arguments
    if len(args) == 1:
        start = 0
        stop = args[0]
        step = 1
    elif len(args) == 2:
        start, stop = args
        step = 1
    else:
        assert len(args) == 3
        start, stop, step = tuple(args)

    # determine number of segments
    n = (stop - start) / step + 1

    # do rounding for n
    if np.isclose(n, np.round(n), rtol=rtol, atol=atol):
        n = np.round(n)

    # correct for start/end is exluded
    if not include[0]:
        n -= 1
        start += step
    if not include[1]:
        n -= 1
        stop -= step

    return np.linspace(start, stop, int(n))


def crange(*args, **kwargs):
    """
    Range excluding the end-point - from `cust_range`.
    """

    return cust_range(*args, **kwargs, include=[True, True])


def orange(*args, **kwargs):
    """
    Range including the end-point - from `cust_range`.
    """

    return cust_range(*args, **kwargs, include=[True, False])
