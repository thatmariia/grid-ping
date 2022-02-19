from src.misc import *

class TestMisc():

    def test_euclidian_dist_R2(self):
        assert euclidian_dist_R2((0, 0), (0, 4)) == 4
