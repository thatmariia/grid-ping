import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.image as mpimg
import seaborn as sns

from src.plotter.setup import PlotPaths


def plot_full_stimulus(stimulus: np.ndarray[(int, int), float]):
    print("Plotting stimulus.....", end="")
    _plot_bw_square_heatmap(stimulus, PlotPaths.FULL_STIMULUS.value)


def fetch_full_stimulus():
    return mpimg.imread(PlotPaths.FULL_STIMULUS.value + ".png")


def plot_stimulus_patch(stimulus_patch: np.ndarray[(int, int), float]):
    print("Plotting patch.....", end="")
    _plot_bw_square_heatmap(stimulus_patch, PlotPaths.STIMULUS_PATCH.value)


def fetch_stimulus_patch():
    return mpimg.imread(PlotPaths.STIMULUS_PATCH.value + ".png")


def plot_local_contrasts(local_contrasts: np.ndarray[(int, int), float]):
    """
    Plots the binary heatmap of local contrasts.

    :param local_contrasts: a local_contrasts matrix to plot.
    :type local_contrasts: numpy.ndarray[(int, int), float]

    :rtype: None
    """

    print("Plotting local contrasts.....", end="")
    _plot_bw_square_heatmap(local_contrasts, PlotPaths.LOCAL_CONTRAST.value)


def fetch_local_contrasts():
    return mpimg.imread(PlotPaths.LOCAL_CONTRAST.value + ".png")


def plot_frequency_vs_current(
        freqs: np.ndarray[int, float], currents: np.ndarray[int, float],
        freqs_line: np.ndarray[int, float], currents_line: np.ndarray[int, float]
) -> None:
    """
    Plots the relationship between frequency and current.

    :param freqs: frequencies from simulated data.
    :type freqs: numpy.ndarray[int, float]

    :param currents: currents from simulated data.
    :type currents: numpy.ndarray[int, float]

    :param freqs_line: frequencies from fitted line.
    :type freqs_line: numpy.ndarray[int, float]

    :param currents_line: currents from fitted line.
    :type currents_line: numpy.ndarray[int, float]

    :rtype: None
    """

    # TODO:: make pretty

    print("Plotting current-frequency.....", end="")
    path = f"{PlotPaths.FREQUENCY_VS_CURRENT.value}.png"

    font = FontProperties()
    font.set_family('serif')
    font.set_name('Avenir')
    font.set_weight('ultralight')

    fig, ax = plt.subplots(figsize=(30, 30))
    ax.tick_params(axis='both', which='major', labelsize=50)

    # simulation data
    plt.scatter(currents, freqs, linewidths=30, s=300, c="#ACDDE7")
    # fitted line
    plt.plot(currents_line, freqs_line, solid_capstyle='round', color="#FFA3AF", lw=10)

    plt.xlabel("Current", fontsize=70, fontproperties=font, labelpad=50)
    plt.ylabel("Frequency", fontsize=70, fontproperties=font, labelpad=50)

    fig.savefig(path, bbox_inches='tight')

    print(end="\r", flush=True)
    print(f"Plotting ended, result: {path}")


def fetch_frequency_vs_current():
    return mpimg.imread(PlotPaths.FREQUENCY_VS_CURRENT.value + ".png")



def _plot_bw_square_heatmap(data: np.ndarray[(int, int), float], filename: str) -> None:
    """
    Plots the binary heatmap of given data.

    :param filename: name of the file for the plot (excluding extension).
    :type filename: str

    :param data: the data to plot.
    :type data: numpy.ndarray[(int, int), float]

    :rtype: None
    """

    path = f"{filename}.png"

    fig, ax = plt.subplots(figsize=(30, 30))
    sns.heatmap(
        data,
        annot=False,
        vmin=0,
        vmax=1,
        cmap="gist_gray",
        cbar=False,
        square=True,
        xticklabels=False,
        yticklabels=False,
        ax=ax
    )

    fig.savefig(path, bbox_inches='tight', pad_inches=0)

    print(end="\r", flush=True)
    print(f"Plotting ended, result: {path}")
