import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.image as mpimg
import seaborn as sns

from src.plotter.setup import PlotPaths, PLOT_FORMAT, PLOT_SIZE


def display_stimulus_currents():
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.imshow(fetch_stimulus_currents())
    ax.axis('off')
    plt.show()

def display_patch_plots():
    fig, ax = plt.subplots(ncols=2, figsize=(10, 5))

    ax[0].imshow(fetch_stimulus_patch())
    ax[0].axis('off')
    ax[1].imshow(fetch_local_contrasts())
    ax[1].axis('off')
    plt.show()

def display_full_stimulus():
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.imshow(fetch_full_stimulus())
    ax.axis('off')
    plt.show()

def display_frequency_vs_current():
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.imshow(fetch_frequency_vs_current())
    ax.axis('off')
    plt.show()


def plot_stimulus_currents(stimulus_currents):
    """
    Plots the stimulus currents.

    :param stimulus_currents: the stimulus currents.
    :type stimulus_currents: numpy.ndarray[int, float]

    :rtype: None
    """

    print("Plotting stimulus currents.....", end="")
    path = f"{PlotPaths.STIMULUS_CURRENTS.value}{PLOT_FORMAT}"

    fig, ax = plt.subplots(figsize=(PLOT_SIZE, PLOT_SIZE))
    sns.heatmap(
        stimulus_currents.reshape((np.sqrt(stimulus_currents.shape[0]).astype(int), np.sqrt(stimulus_currents.shape[0]).astype(int))),
        annot=False,
        cbar=True,
        square=True,
        xticklabels=True,
        yticklabels=True,
        ax=ax
    )

    fig.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()

    print(end="\r", flush=True)
    print(f"Plotting ended, result: {path}")


def fetch_stimulus_currents():
    return mpimg.imread(PlotPaths.STIMULUS_CURRENTS.value + PLOT_FORMAT)


def plot_full_stimulus(stimulus: np.ndarray[(int, int), float]):
    """
    Plots full stimulus.

    :param stimulus:
    :type stimulus: numpy.ndarray[(int, int), float]
    """

    print("Plotting stimulus.....", end="")
    _plot_bw_square_heatmap(stimulus, PlotPaths.FULL_STIMULUS.value)


def fetch_full_stimulus():
    return mpimg.imread(PlotPaths.FULL_STIMULUS.value + PLOT_FORMAT)


def plot_stimulus_patch(stimulus_patch: np.ndarray[(int, int), float]):
    """
    Plots the stimulus patch.

    :param stimulus_patch:
    :type stimulus_patch: numpy.ndarray[(int, int), float]
    """

    print("Plotting patch.....", end="")
    _plot_bw_square_heatmap(stimulus_patch, PlotPaths.STIMULUS_PATCH.value)


def fetch_stimulus_patch():
    return mpimg.imread(PlotPaths.STIMULUS_PATCH.value + PLOT_FORMAT)


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
    return mpimg.imread(PlotPaths.LOCAL_CONTRAST.value + PLOT_FORMAT)


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
    path = f"{PlotPaths.FREQUENCY_VS_CURRENT.value}{PLOT_FORMAT}"

    fig, ax = plt.subplots(figsize=(PLOT_SIZE, PLOT_SIZE))

    # simulation data
    plt.scatter(currents, freqs, c="#ACDDE7")
    # fitted line
    plt.plot(currents_line, freqs_line, solid_capstyle='round', color="#FFA3AF")

    plt.xlabel("Current")
    plt.ylabel("Frequency")

    fig.savefig(path, bbox_inches='tight')
    plt.close()

    print(end="\r", flush=True)
    print(f"Plotting ended, result: {path}")


def fetch_frequency_vs_current():
    return mpimg.imread(PlotPaths.FREQUENCY_VS_CURRENT.value + PLOT_FORMAT)



def _plot_bw_square_heatmap(data: np.ndarray[(int, int), float], filename: str) -> None:
    """
    Plots the binary heatmap of given data.

    :param filename: name of the file for the plot (excluding extension).
    :type filename: str

    :param data: the data to plot.
    :type data: numpy.ndarray[(int, int), float]

    :rtype: None
    """

    path = f"{filename}{PLOT_FORMAT}"

    fig, ax = plt.subplots(figsize=(PLOT_SIZE, PLOT_SIZE))
    sns.heatmap(
        data,
        annot=False,
        vmin=0,
        vmax=1,
        cmap="gist_gray",
        cbar=True,
        square=True,
        xticklabels=False,
        yticklabels=False,
        ax=ax
    )

    fig.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()

    print(end="\r", flush=True)
    print(f"Plotting ended, result: {path}")
