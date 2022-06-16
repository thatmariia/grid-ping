from enum import Enum

PLOT_FORMAT = ".png"
PLOT_SIZE = 10

class PlotNames(Enum):

    FULL_STIMULUS = "full-stimulus"
    STIMULUS_PATCH = "stimulus-patch"
    LOCAL_CONTRAST = "local-contrasts"
    FREQUENCY_VS_CURRENT = "frequency-vs-current"
    FREQUENCY_DISTRIBUTION_EVOLUTION = "frequency-distribution-evolution"
    FREQUENCY_DISTRIBUTION = "frequency-distribution"


class SubdirectoryNames(Enum):
    STIMULUS_CONSTRUCTION = "stimulus-construction"
    PING_FREQUENCIES = "ping-frequencies"
    FREQUENCY_DISTRIBUTION_EVOLUTION = f"ping-frequencies/{PlotNames.FREQUENCY_DISTRIBUTION_EVOLUTION.value}"


class PlotPaths(Enum):

    FULL_STIMULUS = f"{SubdirectoryNames.STIMULUS_CONSTRUCTION.value}/{PlotNames.FULL_STIMULUS.value}"
    STIMULUS_PATCH = f"{SubdirectoryNames.STIMULUS_CONSTRUCTION.value}/{PlotNames.STIMULUS_PATCH.value}"
    LOCAL_CONTRAST = f"{SubdirectoryNames.STIMULUS_CONSTRUCTION.value}/{PlotNames.LOCAL_CONTRAST.value}"
    FREQUENCY_VS_CURRENT = f"{SubdirectoryNames.STIMULUS_CONSTRUCTION.value}/{PlotNames.FREQUENCY_VS_CURRENT.value}"
    FREQUENCY_DISTRIBUTION_EVOLUTION = f"{SubdirectoryNames.PING_FREQUENCIES.value}/{PlotNames.FREQUENCY_DISTRIBUTION_EVOLUTION.value}"
    FREQUENCY_DISTRIBUTION = f"{SubdirectoryNames.PING_FREQUENCIES.value}/{PlotNames.FREQUENCY_DISTRIBUTION.value}"

