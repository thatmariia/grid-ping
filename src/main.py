from src.params.ParamsInitializer import *
from src.stimulus_construction.StimulusFactory import *
from src.izhikevich_simulation.ConnectivityGridPINGFactory import *
from src.izhikevich_simulation.CurrentComponentsGridPING import *
from src.izhikevich_simulation.IzhikevichNetworkSimulator import *
from src.SpikingFrequencyComputer import *

if __name__ == "__main__":
    params_initializer = ParamsInitializer()
    params_ping, params_gabor, params_rf, params_connectivity, params_izhi, params_synaptic, params_freqs = params_initializer.initialize()

    stimulus = StimulusFactory().create(params_gabor, params_rf, params_ping, params_izhi, params_freqs)

    stimulus_locations = stimulus.extract_stimulus_location()

    connectivity = ConnectivityGridPINGFactory().create(
        params_ping=params_ping,
        params_connectivity=params_connectivity,
        cortical_distances=stimulus_locations.cortical_distances
    )
    neural_model = CurrentComponentsGridPING(
        connectivity=connectivity,
        params_synaptic=params_synaptic,
        stimulus_currents=stimulus.stimulus_currents
    )
    simulation_outcome = IzhikevichNetworkSimulator(
        params_izhi=params_izhi,
        current_components=neural_model,
        pb_off=False
    ).simulate(
        simulation_time=1000,
        dt=1
    )

    ping_frequencies = SpikingFrequencyComputer().compute_for_all_pings(
        simulation_outcome=simulation_outcome,
        params_freqs=params_freqs
    )
    SpikingFrequencyComputer().plot_ping_frequencies(ping_frequencies)
