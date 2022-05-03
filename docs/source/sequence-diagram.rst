Sequence diagram
=================

.. uml::

    @startuml
    actor user
    activate user

    user -> InputStimulus : <<creates>>
    activate InputStimulus

    InputStimulus -> GaborLuminanceStimulusOld : super()
    activate GaborLuminanceStimulusOld

    GaborLuminanceStimulusOld -> GaborLuminanceStimulusOld: composes figure and ground
    GaborLuminanceStimulusOld -> GaborLuminanceStimulusOld: creates luminance stimulus
    GaborLuminanceStimulusOld -> GaborLuminanceStimulusOld: selects patch from figure

    GaborLuminanceStimulusOld --> InputStimulus

    InputStimulus -> InputStimulus: creates circuits from luminance stimulus
    InputStimulus -> InputStimulus: converts luminance to local contrast
    InputStimulus -> InputStimulus: converts local contrast to frequencies (TODO)
    InputStimulus -> InputStimulus: converts frequencies to current (TODO)

    InputStimulus --> user : initialized stimulus

    deactivate GaborLuminanceStimulusOld
    deactivate InputStimulus

    activate OscillatoryNetwork
    user -> OscillatoryNetwork: run_simulation()
    note right
      uses the stimulus
    end note

    OscillatoryNetwork -> GridConnectivity: <<creates>>
    activate GridConnectivity

    GridConnectivity -> GridConnectivity: assigns neurons to PING networks in a grid
    GridConnectivity -> GridConnectivity: computes coupling weights between neurons

    GridConnectivity --> OscillatoryNetwork: initialized grid connectivity
    deactivate GridConnectivity

    OscillatoryNetwork -> OscillatoryNetwork: simulates neural network (Eurler-forward)

    OscillatoryNetwork --> user: ended simulation
    deactivate OscillatoryNetwork
    deactivate user

    @enduml