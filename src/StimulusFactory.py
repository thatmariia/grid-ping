from src.ParamsPING import *
from src.ParamsGaborStimulus import *
from src.ParamsReceptiveField import *
from src.ParamsIzhikevich import *

from src.GaborLuminanceStimulusFactory import *
from src.PatchGeometryFactory import *
from src.LuminanceToContrastConverter import *
from src.ContrastToFrequencyConverter import *
from src.FrequencyToCurrentConverter import *
from src.Stimulus import *

import numpy as np
from typing import Any


class StimulusFactory:
    """
    This class creates an external stimulus (Gabor texture) and prepares for the neural network input.
    """

    def create(
            self, params_gabor: ParamsGaborStimulus,
            params_rf: ParamsReceptiveField,
            params_ping: ParamsPING,
            params_izhi: ParamsIzhikevich
    ) -> Stimulus:
        """
        Creates an external stimulus (Gabor texture) and prepares for the neural network input.

        :param params_gabor: parameters for creating a Gabor luminance stimulus.
        :type params_gabor: ParamsGaborStimulus

        :param params_rf: parameters for the receptive field.
        :type params_rf: ParamsReceptiveField

        :param params_ping: parameters describing PING networks and their composition.
        :type params_ping: ParamsPING

        :param params_izhi: contains Izhikevich parameters.
        :type params_izhi: ParamsIzhikevich

        :raises:
            AssertionError: if vertical lines of lattice cut through pixels.
        :raises:
            AssertionError: if horizontal lines of lattice cut through pixels.

        :return: a stimulus ready for the usage in a neural network.
        :rtype: Stimulus
        """

        stimulus_luminance = GaborLuminanceStimulusFactory().create(params_gabor)

        stimulus_luminance.plot_stimulus(filename="test-full-stimulus")
        stimulus_luminance.plot_patch(filename="test-stimulus-patch")

        assert np.shape(stimulus_luminance.stimulus_patch)[0] % int(math.sqrt(params_ping.nr_ping_networks)) == 0, \
            "Vertical lines of lattice should not cut through pixels."
        assert np.shape(stimulus_luminance.stimulus_patch)[1] % int(math.sqrt(params_ping.nr_ping_networks)) == 0, \
            "Horizontal lines of lattice should not cut through pixels."

        patch_geometry = PatchGeometryFactory().create(
            nr_ping_networks=params_ping.nr_ping_networks,
            stimulus_patch=stimulus_luminance.stimulus_patch,
            patch_start=stimulus_luminance.patch_start,
            stimulus_center=stimulus_luminance.stimulus_center,
            atopix=stimulus_luminance.atopix
        )

        stimulus_contrasts = LuminanceToContrastConverter().convert(
            params_rf, patch_geometry, stimulus_luminance
        )
        stimulus_frequencies = ContrastToFrequencyConverter().convert(
            stimulus_contrasts
        )
        stimulus_currents = FrequencyToCurrentConverter().convert(
            stimulus_frequencies, params_ping, params_izhi
        )

        stimulus = Stimulus(
            stimulus_currents, patch_geometry
        )

        return stimulus

