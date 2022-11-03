import numpy as np
from itertools import product
from tqdm import tqdm
from src.misc import *
from math import pi, exp
import cmath

from src.izhikevich_simulation.GridGeometry import *
from src.izhikevich_simulation.PINGNetworkNeurons import *
from src.params.ParamsPING import *

class RingGeometryFactory:
    """
    This class constructs a grid of PING networks and distributes neurons among them.
    """

    def create(self, params_ping: ParamsPING) -> GridGeometry:
        neuron_distances = np.zeros((params_ping.nr_neurons["total"], params_ping.nr_neurons["total"]))

        dist_EE = self._get_dist(
            nr1=params_ping.nr_neurons[NeuronTypes.EX],
            nr2=params_ping.nr_neurons[NeuronTypes.EX],
            op1=self._get_op(nr=params_ping.nr_neurons[NeuronTypes.EX]),
            op2=self._get_op(nr=params_ping.nr_neurons[NeuronTypes.EX])
        )
        dist_II = self._get_dist(
            nr1=params_ping.nr_neurons[NeuronTypes.IN],
            nr2=params_ping.nr_neurons[NeuronTypes.IN],
            op1=self._get_op(nr=params_ping.nr_neurons[NeuronTypes.IN]),
            op2=self._get_op(nr=params_ping.nr_neurons[NeuronTypes.IN])
        )
        dist_EI = self._get_dist(
            nr1=params_ping.nr_neurons[NeuronTypes.EX],
            nr2=params_ping.nr_neurons[NeuronTypes.IN],
            op1=self._get_op(nr=params_ping.nr_neurons[NeuronTypes.EX]),
            op2=self._get_op(nr=params_ping.nr_neurons[NeuronTypes.IN])
        )
        dist_IE = self._get_dist(
            nr1=params_ping.nr_neurons[NeuronTypes.IN],
            nr2=params_ping.nr_neurons[NeuronTypes.EX],
            op1=self._get_op(nr=params_ping.nr_neurons[NeuronTypes.IN]),
            op2=self._get_op(nr=params_ping.nr_neurons[NeuronTypes.EX])
        )

        neuron_distances[params_ping.neur_slice[NeuronTypes.EX], params_ping.neur_slice[NeuronTypes.EX]] = dist_EE
        neuron_distances[params_ping.neur_slice[NeuronTypes.IN], params_ping.neur_slice[NeuronTypes.IN]] = dist_II
        neuron_distances[params_ping.neur_slice[NeuronTypes.EX], params_ping.neur_slice[NeuronTypes.IN]] = dist_EI
        neuron_distances[params_ping.neur_slice[NeuronTypes.IN], params_ping.neur_slice[NeuronTypes.EX]] = dist_IE

        grid_geometry = GridGeometry(
            ping_networks=[],
            neuron_distances=neuron_distances
        )

        return grid_geometry


    def _get_op(self, nr):
        step = 2 * pi / (nr - 1)
        op = crange(-pi, pi, step)
        return op

    def _get_dist(self, nr1, nr2, op1, op2):
        dist = np.zeros((nr1, nr2))

        for i in range(nr1):
            zs_nom = [cmath.exp(complex(real=0.0, imag=op1[i]))] * len(op2)
            zs_denom = [cmath.exp(complex(real=0.0, imag=j)) for j in op2]
            zs = np.true_divide(zs_nom, zs_denom)
            angles = abs(np.angle(z=zs))
            dist[i] = angles

        dist[dist < 0.001] = None
        return dist

def cust_range(*args, rtol=1e-05, atol=1e-08, include=[True, False]):
    """
    Combines numpy.arange and numpy.isclose to mimic
    open, half-open and closed intervals.
    Avoids also floating point rounding errors as with
    >>> np.arange(1, 1.3, 0.1)
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
    n = (stop-start)/step + 1

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
    return cust_range(*args, **kwargs, include=[True, True])

