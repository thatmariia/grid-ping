from src.OscillatoryNetwork import *
from src.GridConnectivity import *

connectivity1 = GridConnectivity(
    nr_excit=2,
    nr_inhibit=2,
    nr_oscillators=1
)

oscillatory_network1 = OscillatoryNetwork(
    nr_excit=2,
    nr_inhibit=2,
    nr_oscillators=1
)

connectivity2 = GridConnectivity(
    nr_excit=8,
    nr_inhibit=4,
    nr_oscillators=4
)

oscillatory_network2 = OscillatoryNetwork(
    nr_excit=8,
    nr_inhibit=4,
    nr_oscillators=4
)

