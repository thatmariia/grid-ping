from src.OscillatoryNetwork import *


if __name__ == "__main__":
    oscillatory_network = OscillatoryNetwork(
        nr_excitatory=8,
        nr_inhibitory=4,
        nr_oscillators=4
    )
    oscillatory_network.run_simulation(
        simulation_time=8,
        dt=1
    )

