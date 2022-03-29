Class diagram
=============

.. uml::

    @startuml

    enum NeuronTypes {
      excitatory
      inhibitory
    }

    class StimulusCircuit {
      +center: tuple
      +pixels: list
    }

    class InputStimulus {
      -nr_circuits: int
      +current: list

      -_assign_circuits()
      -_get_input_current()
      -_get_weight()
      -_compute_local_contrasts()
      -_compute_frequencies()
      -_compute_current()
    }

    abstract class GaborLuminanceStimulus {
      -_side_length: TODO
      -_patch_res: int
      -_stim_res: TODO
      +stimulus: list
      +stimulus_patch: list

      +plot_stimulus()
      #_eccentricity_in_patch()
      -_from_pixel_to_physical_stim()
      -_get_grating()
      -_get_full_stimulus()
      -_get_start_of_patch()
      -_get_stimulus_patch()
    }

    class OscillatoryNetwork {
      -_nr_neurons: int
      -_nr_ping_networks: int
      -_stimulus: list
      -_synaptic_currents: list
      -_current: TODO
      -_potentials: list
      -_recovery: list
      -_izhi_alpha: list
      -_izhi_beta: list
      -_izhi_gamma: list
      -_izhi_zeta: list

      +run_simulation()
      -_change_recovery()
      -_change_potential()
      -_get_synaptic_current()
      -_change_thalamic_input()
      -_create_main_input_stimulus()
    }

    class PINGNetwork {
      +location: tuple
      +ids: dict
    }

    class GridConnectivity {
      -_nr_neurons: int
      -_nr_ping_networks: int
      +coupling_weights: int

      -_assign_ping_networks()
      -_compute_coupling_weights()
      -_get_neurons_dist()
      -_compute_type_coupling_weights()
    }

    OscillatoryNetwork "1" *-- "1" GridConnectivity
    GridConnectivity "1" *-- "1..*" PINGNetwork

    GaborLuminanceStimulus <|-- InputStimulus
    InputStimulus "1" *-- "1..*" StimulusCircuit

    OscillatoryNetwork "1" o-- "1" InputStimulus

    NeuronTypes .. OscillatoryNetwork
    NeuronTypes .. GridConnectivity


    @enduml
