from src.misc import *
from src.NeuronTypes import *

class TestMisc:

    def test_euclidian_dist_R2(self):
        assert euclidian_dist((0, 0), (0, 4)) == 4
        assert round(euclidian_dist((-7.3, -4), (17, 6.5)), 4) == 26.4715
        assert round(euclidian_dist((-0.27, 3.91), (8.4, -0.5)), 4) == 9.7271

    def test_neuron_slice(self):
        assert neur_slice(NeuronTypes.EX, 10, 5) == slice(10)
        assert neur_slice(NeuronTypes.IN, 10, 5) == slice(10, 15)
