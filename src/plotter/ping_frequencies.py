import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.image as mpimg
import seaborn as sns
from math import sqrt
import os

from src.plotter.setup import PlotPaths, PLOT_FORMAT, PLOT_SIZE


def display_ping_frequencies():
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.imshow(fetch_ping_frequencies())
    ax.axis('off')
    plt.show()


def plot_ping_frequencies(frequencies, t_ms=-1):
    # TODO:: make pretty

    if t_ms == -1:
        print("Plotting current-frequency.....", end="")
        path = f"{PlotPaths.FREQUENCY_DISTRIBUTION.value}{PLOT_FORMAT}"
    else:
        path = f"{PlotPaths.FREQUENCY_DISTRIBUTION_EVOLUTION.value}/{t_ms}ms{PLOT_FORMAT}"

    fig, ax = plt.subplots(ncols=2, figsize=(2 * PLOT_SIZE, PLOT_SIZE))
    # ax.tick_params(axis='both', which='major', labelsize=50)

    ax[0].hist(frequencies, color="#ACDDE7", rwidth=0.7)
    sns.heatmap(np.array(frequencies).reshape(int(sqrt(len(frequencies))), int(sqrt(len(frequencies)))), ax=ax[1])
    fig.savefig(path, bbox_inches='tight')
    plt.close()

    if t_ms == -1:
        print(end="\r", flush=True)
        print(f"Plotting ended, result: {path}")


def fetch_ping_frequencies():
    return mpimg.imread(PlotPaths.FREQUENCY_DISTRIBUTION.value + PLOT_FORMAT)


def fetch_ping_frequencies_evolution():
    filenames = sorted([
        f for f in os.listdir(PlotPaths.FREQUENCY_DISTRIBUTION_EVOLUTION.value)
        if os.path.isfile(os.path.join(PlotPaths.FREQUENCY_DISTRIBUTION_EVOLUTION.value, f))
    ])
    print(filenames)
    return [mpimg.imread(PlotPaths.FREQUENCY_DISTRIBUTION_EVOLUTION.value + "/" + f) for f in filenames]

