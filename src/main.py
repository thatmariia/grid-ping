from src.OscillatoryNetwork import *


if __name__ == "__main__":
    oscillatory_network = OscillatoryNetwork()
    oscillatory_network.run_simulation(
        simulation_time=8,
        dt=1
    )

