Class diagram
=============

.. uml::

    @startuml

    class PINGNetworkPixels {
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

    class Stimulus {
      +stimulus_currents
      -_patch_geometry

      +extract_stimulus_location()
    }

    class StimulusFactory {
      +create()
    }

    class FrequencyToCurrentConverter {
      +convert()
      -_some_conversion_functions_TODO()
    }

    class ContrastToFrequencyConverter {
      +convert()
      -_compute_frequencies()
    }

    class LuminanceToContrastConverter {
      +convert()
      -_get_weight()
      -_compute_local_contrasts()
    }

    class PatchGeometry {
      +ping_networks_pixels
      -_atopix
      -_patch_start
      -_stimulus_center

      +angle_in_patch()
      +eccentricity_in_patch()
      +point_in_stimulus()
    }

    class PatchGeometryFactory {
      +create()
      -_assign_circuits()
    }

    class GaborLuminanceStimulus {
      +atopix
      +stimulus
      +stimulus_center
      +stimulus_patch
      +patch_start

      +plot_stimulus()
      +plot_patch()
      -_plot()
    }

    class GaborLuminanceStimulusFactory {
      +create()
      -_get_grating()
      -_get_figure_coords()
      -_get_full_stimulus()
      -_is_annulus_in_figure()
      -_select_stimulus_patch()
    }

    GaborLuminanceStimulus <|.. GaborLuminanceStimulusFactory
    PatchGeometry <|.. PatchGeometryFactory
    GaborLuminanceStimulus *-- StimulusFactory
    PatchGeometry *-- StimulusFactory
    LuminanceToContrastConverter *-- StimulusFactory
    ContrastToFrequencyConverter *-- StimulusFactory
    FrequencyToCurrentConverter *-- StimulusFactory
    Stimulus <|.. StimulusFactory
    StimulusLocations <|.. Stimulus
    PINGNetworkPixels *-- PatchGeometryFactory

    @enduml