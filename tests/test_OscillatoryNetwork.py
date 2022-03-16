from src.OscillatoryNetwork import *

class TestOscillatoryNetwork:

    network1 = OscillatoryNetwork(
        nr_excitatory=2,
        nr_inhibitory=2,
        nr_ping_networks=1
    )

    network2 = OscillatoryNetwork(
        nr_excitatory=8,
        nr_inhibitory=4,
        nr_ping_networks=4
    )

    def test_change_recovery(self):

        d_recovery1 = self.network1._change_recovery()
        d_recovery1_expected = [0, 0, 0, 0]

        # -----------------------------------------------------------------------

        d_recovery2 = self.network2._change_recovery()
        d_recovery2_expected = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        assert np.array_equal(d_recovery1, d_recovery1_expected)
        assert np.array_equal(d_recovery2, d_recovery2_expected)

    def test_change_potential(self):
        # FIXME:: fix this, update with the new func

        c1 = [0.1] * 4

        d_potential1 = self.network1._change_potential(current=c1)
        d_potential1_expected = [-1.45] * 4

        # -----------------------------------------------------------------------

        c2 = [0.1] * 12

        d_potential2 = self.network2._change_potential(current=c2)
        d_potential2_expected = [-1.45] * 12

        assert np.array_equal(d_potential1, d_potential1_expected)
        assert np.array_equal(d_potential2, d_potential2_expected)

    def test_change_synaptic_potentials(self):
        # TODO
        pass
