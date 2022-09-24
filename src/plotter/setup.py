from enum import Enum
import matplotlib as mpl

DATA_FORMAT = ".txt"
PLOT_FORMAT = ".png"
PLOT_SIZE = 10

mpl.rcParams['lines.linewidth'] = PLOT_SIZE / 2
mpl.rcParams['axes.linewidth'] = PLOT_SIZE / 10
mpl.rcParams['axes.labelsize'] = 2 * PLOT_SIZE
mpl.rcParams['axes.labelpad'] = 2 * PLOT_SIZE
mpl.rcParams['axes.titlesize'] = 3 * PLOT_SIZE
mpl.rcParams['axes.titlepad'] = 3 * PLOT_SIZE
mpl.rcParams['legend.fontsize'] = PLOT_SIZE
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Avenir'
mpl.rcParams['font.weight'] = 'ultralight'


class PlotNames(Enum):

    FULL_STIMULUS = "full-stimulus"
    STIMULUS_PATCH = "stimulus-patch"
    LOCAL_CONTRAST = "local-contrasts"
    STIMULUS_CURRENTS = "stimulus-currents"
    FREQUENCY_VS_CURRENT = "frequency-vs-current"
    FREQUENCY_DISTRIBUTION_EVOLUTION = "frequency-distribution-evolution"
    FREQUENCY_DISTRIBUTION = "frequency-distribution"
    FREQUENCY_SINGLE_PING_EVOLUTION = "frequency-single-ping-evolution"
    SPIKES_OVER_TIME = "spikes-over-time"
    FREQUENCY_STDS = "frequency-stds"


class DataNames(Enum):
    SPIKES_DATA = "spikes-data"
    CORTICAL_DISTANCES_DATA = "euclid-dist-data"


class ParticSubdirectoryNames(Enum):
    STIMULUS_CONSTRUCTION = "stimulus-construction"
    PING_FREQUENCIES = "ping-frequencies"
    FREQUENCY_DISTRIBUTION_EVOLUTION = f"ping-frequencies/{PlotNames.FREQUENCY_DISTRIBUTION_EVOLUTION.value}"
    RAW_DATA = "raw-data"


class GeneralSubdirectoryNames(Enum):
    OVERVIEWS = "overviews"


class PlotPaths(Enum):

    FULL_STIMULUS = f"{ParticSubdirectoryNames.STIMULUS_CONSTRUCTION.value}/{PlotNames.FULL_STIMULUS.value}"
    STIMULUS_PATCH = f"{ParticSubdirectoryNames.STIMULUS_CONSTRUCTION.value}/{PlotNames.STIMULUS_PATCH.value}"
    LOCAL_CONTRAST = f"{ParticSubdirectoryNames.STIMULUS_CONSTRUCTION.value}/{PlotNames.LOCAL_CONTRAST.value}"
    STIMULUS_CURRENTS = f"{ParticSubdirectoryNames.STIMULUS_CONSTRUCTION.value}/{PlotNames.STIMULUS_CURRENTS.value}"
    FREQUENCY_VS_CURRENT = f"{ParticSubdirectoryNames.STIMULUS_CONSTRUCTION.value}/{PlotNames.FREQUENCY_VS_CURRENT.value}"
    FREQUENCY_DISTRIBUTION_EVOLUTION = f"{ParticSubdirectoryNames.PING_FREQUENCIES.value}/{PlotNames.FREQUENCY_DISTRIBUTION_EVOLUTION.value}"
    FREQUENCY_DISTRIBUTION = f"{ParticSubdirectoryNames.PING_FREQUENCIES.value}/{PlotNames.FREQUENCY_DISTRIBUTION.value}"
    FREQUENCY_SINGLE_PING_EVOLUTION = f"{ParticSubdirectoryNames.PING_FREQUENCIES.value}/{PlotNames.FREQUENCY_SINGLE_PING_EVOLUTION.value}"
    SPIKES_OVER_TIME = f"{ParticSubdirectoryNames.PING_FREQUENCIES.value}/{PlotNames.SPIKES_OVER_TIME.value}"
    FREQUENCY_STDS = f"{PlotNames.FREQUENCY_STDS.value}"


class DataPaths(Enum):
    SPIKES_DATA = f"{ParticSubdirectoryNames.RAW_DATA.value}"
    CORTICAL_DISTANCES_DATA = f"{ParticSubdirectoryNames.RAW_DATA.value}"
