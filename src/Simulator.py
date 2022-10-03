from src.stimulus_construction.StimulusFactory import StimulusFactory
from src.izhikevich_simulation.ConnectivityGridPINGFactory import *
from src.izhikevich_simulation.CurrentComponentsGridPING import *
from src.after_simulation_analysis.SpikingFrequencyFactory import *
from src.izhikevich_simulation.IzhikevichNetworkSimulator import IzhikevichNetworkSimulator

from src.plotter.raw_data import save_spikes_data, save_cortical_dist_data



class Simulator:

    def run_simulation(
            self,
            simulation_time,
            params_gabor,
            params_rf,
            params_ping,
            params_izhi,
            params_freqs,
            params_connectivity,
            params_synaptic
    ) -> IzhikevichNetworkOutcome:

            stimulus = StimulusFactory().create(params_gabor, params_rf, params_ping, params_izhi, params_freqs)

            stimulus_currents = stimulus.stimulus_currents
            cortical_distances = stimulus.extract_stimulus_location().cortical_distances

            save_cortical_dist_data(cortical_distances)

            connectivity = ConnectivityGridPINGFactory().create(
                params_ping=params_ping,
                params_connectivity=params_connectivity,
                cortical_distances=cortical_distances
            )
            neural_model = CurrentComponentsGridPING(
                connectivity=connectivity,
                params_synaptic=params_synaptic,
                stimulus_currents=stimulus_currents
            )
            simulation_outcome = IzhikevichNetworkSimulator(
                params_izhi=params_izhi,
                current_components=neural_model,
                pb_off=False
            ).simulate(
                simulation_time=simulation_time,
                dt=1,
                params_freqs=params_freqs
            )

            save_spikes_data(simulation_outcome.spikes)

            return simulation_outcome




