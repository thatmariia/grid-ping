from src.OscillatoryNetwork import *

if __name__ == "__main__":
    oscillatory_network = OscillatoryNetwork(
        nr_excit=8,
        nr_inhibit=4,
        nr_oscillators=4
    )
    oscillatory_network.run_simulation(
        simulation_time=8,
        dt=1
    )

