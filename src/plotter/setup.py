from enum import Enum
import matplotlib as mpl
import pandas as pd

DATA_FORMAT = ".txt"
PLOT_FORMAT = ".png"
PLOT_SIZE = 10
PLOT_COLORS = ["#FFA3AF", "#ACDDE7", "#FFD8A8", "#B5EAD7", "#C7CEEA", "#FF9AA2", "#FFB7B2", "#FFDAC1", "#E2F0CB", "#E0E1FF"]

mpl.rcParams['lines.linewidth'] = PLOT_SIZE / 2
mpl.rcParams['axes.linewidth'] = PLOT_SIZE / 10
mpl.rcParams['axes.labelsize'] = 2 * PLOT_SIZE
mpl.rcParams['axes.labelpad'] = 2 * PLOT_SIZE
mpl.rcParams['axes.titlesize'] = 2.5 * PLOT_SIZE
mpl.rcParams['axes.titlepad'] = 3 * PLOT_SIZE
mpl.rcParams['legend.fontsize'] = PLOT_SIZE
mpl.rcParams['font.family'] = 'serif'
# mpl.rcParams['font.serif'] = 'Avenir'
mpl.rcParams['font.weight'] = 'ultralight'

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class PlotNames(Enum):

    # stimulus
    FULL_STIMULUS = "full-stimulus"
    STIMULUS_PATCH = "stimulus-patch"
    LOCAL_CONTRAST = "local-contrasts"
    STIMULUS_CURRENTS = "stimulus-currents"
    FREQUENCY_VS_CURRENT = "frequency-vs-current"
    # after simulation
    FREQUENCY_DISTRIBUTION_EVOLUTION = "frequency-distribution-evolution"
    FREQUENCY_DISTRIBUTION = "frequency-distribution"
    FREQUENCY_SINGLE_PING_EVOLUTION = "frequency-single-ping-evolution"
    SPIKES_OVER_TIME = "spikes-over-time"

    RASTER = "raster"
    NR_NEURONS_SPIKED = "nr-neurons-spiked"
    STATS_PER_PING = "stats-per-ping"
    PHASE_LOCKING = "phase-locking"
    # overview
    FREQUENCY_STDS = "frequency-stds"
    AVG_PHASE_LOCKINGS = "avg-phase-lockings"
    AVG_PHASE_LOCKING_SMOOTH = "avg-phase-locking-smooth"


class DataNames(Enum):
    SPIKES_DATA = "spikes-data"
    CORTICAL_DISTANCES_DATA = "cortical-dist-data"


class ParticSubdirectoryNames(Enum):
    STIMULUS_CONSTRUCTION = "stimulus-construction"
    AFTER_SIMULATION_PLOTS = "after-simulation-plots"
    FREQUENCY_DISTRIBUTION_EVOLUTION = f"after-simulation-plots/{PlotNames.FREQUENCY_DISTRIBUTION_EVOLUTION.value}"
    RAW_DATA = "raw-data"


class GeneralSubdirectoryNames(Enum):
    OVERVIEWS = "overviews"


class PlotPaths(Enum):
    # stimulus
    FULL_STIMULUS = f"{ParticSubdirectoryNames.STIMULUS_CONSTRUCTION.value}/{PlotNames.FULL_STIMULUS.value}"
    STIMULUS_PATCH = f"{ParticSubdirectoryNames.STIMULUS_CONSTRUCTION.value}/{PlotNames.STIMULUS_PATCH.value}"
    LOCAL_CONTRAST = f"{ParticSubdirectoryNames.STIMULUS_CONSTRUCTION.value}/{PlotNames.LOCAL_CONTRAST.value}"
    STIMULUS_CURRENTS = f"{ParticSubdirectoryNames.STIMULUS_CONSTRUCTION.value}/{PlotNames.STIMULUS_CURRENTS.value}"
    FREQUENCY_VS_CURRENT = f"{ParticSubdirectoryNames.STIMULUS_CONSTRUCTION.value}/{PlotNames.FREQUENCY_VS_CURRENT.value}"
    # after simulation
    FREQUENCY_DISTRIBUTION_EVOLUTION = f"{ParticSubdirectoryNames.AFTER_SIMULATION_PLOTS.value}/{PlotNames.FREQUENCY_DISTRIBUTION_EVOLUTION.value}"
    FREQUENCY_DISTRIBUTION = f"{ParticSubdirectoryNames.AFTER_SIMULATION_PLOTS.value}/{PlotNames.FREQUENCY_DISTRIBUTION.value}"
    FREQUENCY_SINGLE_PING_EVOLUTION = f"{ParticSubdirectoryNames.AFTER_SIMULATION_PLOTS.value}/{PlotNames.FREQUENCY_SINGLE_PING_EVOLUTION.value}"
    SPIKES_OVER_TIME = f"{ParticSubdirectoryNames.AFTER_SIMULATION_PLOTS.value}/{PlotNames.SPIKES_OVER_TIME.value}"

    RASTER = f"{ParticSubdirectoryNames.AFTER_SIMULATION_PLOTS.value}/{PlotNames.RASTER.value}"
    NR_NEURONS_SPIKED = f"{ParticSubdirectoryNames.AFTER_SIMULATION_PLOTS.value}/{PlotNames.NR_NEURONS_SPIKED.value}"
    STATS_PER_PING = f"{ParticSubdirectoryNames.AFTER_SIMULATION_PLOTS.value}/{PlotNames.STATS_PER_PING.value}"
    PHASE_LOCKING = f"{ParticSubdirectoryNames.AFTER_SIMULATION_PLOTS.value}/{PlotNames.PHASE_LOCKING.value}"
    # overview
    FREQUENCY_STDS = f"{PlotNames.FREQUENCY_STDS.value}"
    AVG_PHASE_LOCKINGS = f"{PlotNames.AVG_PHASE_LOCKINGS.value}"
    AVG_PHASE_LOCKING_SMOOTH = f"{PlotNames.AVG_PHASE_LOCKING_SMOOTH.value}"


class DataPaths(Enum):
    SPIKES_DATA = f"{ParticSubdirectoryNames.RAW_DATA.value}"
    CORTICAL_DISTANCES_DATA = f"{ParticSubdirectoryNames.RAW_DATA.value}"
