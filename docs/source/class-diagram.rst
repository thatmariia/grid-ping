Class diagram
=============

.. uml::

    @startuml

    enum NeuronTypes {
      excitatory
      inhibitory
    }

    class StimulusCircuit {
      +center
      +pixels
      +center_dg
      +pixels_dg
      +grid_index
    }

    class StimulusLocations {
      +cortical_coords

      -_compute_coordinates()
    }

    class InputStimulus {
      -_nr_circuits
      -_circuits
      +current

      -_assign_circuits()
      +extract_stimulus_location()
      -_get_input_current()
      -_get_weight()
      -_point_in_stimulus()
      -_angle_in_patch()
      -_eccentricity_in_patch()
      -_compute_local_contrasts()
      -_compute_frequencies()
      -_compute_current()
    }

    abstract class GaborLuminanceStimulus {
      #_atopix
      #_full_width
      #_full_height
      #_patch_start
      +stimulus
      +stimulus_patch

      +plot_stimulus()
      -_get_grating()
      -_get_figure_coords()
      -_get_full_stimulus()
      -_is_annulus_in_figure()
      -_select_stimulus_patch()
    }

    class OscillatoryNetwork {
      -_nr_neurons
      -_nr_ping_networks
      -_stimulus
      -_currents
      -_potentials
      -_recovery
      -_izhi_alpha
      -_izhi_beta
      -_izhi_gamma
      -_izhi_zeta

      +run_simulation()
      -_get_change_in_recovery()
      -_get_change_in_potentials()
      -_get_change_in_gatings()
      -_get_synaptic_currents()
      -_get_thalamic_input()
      -_create_main_input_stimulus()
    }

    class PINGNetwork {
      +location
      +ids
    }

    class GridConnectivity {
      -_nr_neurons
      -_nr_ping_networks
      -_cortical_coords
      +coupling_weights

      -_assign_ping_networks()
      -_compute_coupling_weights()
      -_get_neurons_dist()
      -_compute_type_coupling_weights()
    }

    OscillatoryNetwork "1" *- "1" GridConnectivity
    GridConnectivity "1" *-- "1..*" PINGNetwork

    GaborLuminanceStimulus <|-- InputStimulus
    InputStimulus "1" *-- "1..*" StimulusCircuit
    InputStimulus "1" *- "1" StimulusLocations

    StimulusLocations "1" <-- "1" GridConnectivity

    OscillatoryNetwork "1" o-- "1" InputStimulus

    NeuronTypes .. OscillatoryNetwork
    NeuronTypes .. GridConnectivity


    @enduml
