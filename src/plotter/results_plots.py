import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

from src.plotter.setup import PlotPaths, PLOT_FORMAT, PLOT_SIZE


####################################################################################################


def plot_frequencies_std(frequencies_std):

    path = f"{PlotPaths.FREQUENCY_STDS.value}{PLOT_FORMAT}"
    print("Plotting frequency std's.....", end="")

    fig, ax = plt.subplots(figsize=(PLOT_SIZE, PLOT_SIZE))
    sns.heatmap(
        frequencies_std,
        annot=True,
        square=True,
        ax=ax
    )

    fig.savefig(path, bbox_inches='tight')
    plt.close()

    print(end="\r", flush=True)
    print(f"Plotting ended, result: {path}")


def fetch_frequencies_stds():
    return mpimg.imread(PlotPaths.FREQUENCY_STDS.value + PLOT_FORMAT)


def display_frequencies_std():
    fig, ax = plt.subplots(figsize=(PLOT_SIZE, PLOT_SIZE))

    ax.imshow(fetch_frequencies_stds())
    ax.axis('off')
    plt.show()


####################################################################################################

def plot_avg_phase_lockings(avg_phase_lockings, smooth=False):

    if smooth:
        path = f"{PlotPaths.AVG_PHASE_LOCKING_SMOOTH.value}{PLOT_FORMAT}"
    else:
        path = f"{PlotPaths.AVG_PHASE_LOCKINGS.value}{PLOT_FORMAT}"
    print("Plotting avg phase-lockings.....", end="")

    fig, ax = plt.subplots(figsize=(PLOT_SIZE, PLOT_SIZE))
    sns.heatmap(
        avg_phase_lockings,
        annot=not smooth,
        square=True,
        ax=ax
    )

    fig.savefig(path, bbox_inches='tight')
    plt.close()

    print(end="\r", flush=True)
    print(f"Plotting ended, result: {path}")