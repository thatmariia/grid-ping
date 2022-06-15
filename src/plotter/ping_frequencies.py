import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.image as mpimg
import seaborn as sns
from math import sqrt
import os

from src.plotter.setup import PlotPaths


def plot_ping_frequencies(frequencies, t_ms=-1):
    # TODO:: make pretty

    print("Plotting current-frequency.....", end="")
    if t_ms == -1:
        path = f"{PlotPaths.FREQUENCY_DISTRIBUTION.value}.png"
    else:
        path = f"{PlotPaths.FREQUENCY_DISTRIBUTION_EVOLUTION.value}/{t_ms}ms.png"

    fig, ax = plt.subplots(ncols=2, figsize=(60, 30))
    # ax.tick_params(axis='both', which='major', labelsize=50)

    ax[0].hist(frequencies, color="#ACDDE7", rwidth=0.7)
    sns.heatmap(np.array(frequencies).reshape(int(sqrt(len(frequencies))), int(sqrt(len(frequencies)))), ax=ax[1])
    fig.savefig(path, bbox_inches='tight')

    print(end="\r", flush=True)
    print(f"Plotting ended, result: {path}")


def fetch_ping_frequencies():
    return mpimg.imread(PlotPaths.FREQUENCY_DISTRIBUTION.value + ".png")


def fetch_ping_frequencies_evolution():
    filenames = sorted([
        f for f in os.listdir(PlotPaths.FREQUENCY_DISTRIBUTION_EVOLUTION.value)
        if os.path.isfile(os.path.join(PlotPaths.FREQUENCY_DISTRIBUTION_EVOLUTION.value, f))
    ])
    print(filenames)
    return [mpimg.imread(PlotPaths.FREQUENCY_DISTRIBUTION_EVOLUTION.value + "/" + f) for f in filenames]

