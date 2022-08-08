import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.image as mpimg
import seaborn as sns
from math import sqrt
import os

from src.plotter.setup import PlotPaths, PLOT_FORMAT, PLOT_SIZE


def display_ping_frequencies():
    fig, ax = plt.subplots(figsize=(2 * PLOT_SIZE, PLOT_SIZE))

    ax.imshow(fetch_ping_frequencies())
    ax.axis('off')
    plt.show()


def display_frequencies_std():
    fig, ax = plt.subplots(figsize=(PLOT_SIZE, PLOT_SIZE))

    ax.imshow(fetch_frequencies_stds())
    ax.axis('off')
    plt.show()


def display_single_ping_frequency_evolution():
    fig, ax = plt.subplots(figsize=(PLOT_SIZE, PLOT_SIZE))

    ax.imshow(fetch_single_ping_frequency_evolution())
    ax.axis('off')
    plt.show()

def display_spikes_over_time():
    # TODO:: fig size?
    fig, ax = plt.subplots(figsize=(PLOT_SIZE, PLOT_SIZE))

    ax.imshow(fetch_spikes_over_time())
    ax.axis('off')
    plt.show()


def plot_neuron_spikes_over_time(spikes, simulation_time, ids):
    assert len(ids) > 0, "enter at least a single neuron ID"

    if len(spikes) == 0:
        print("No spikes found, skipping plot")
        return

    path = f"{PlotPaths.SPIKES_OVER_TIME.value}{PLOT_FORMAT}"
    print("Plotting frequency std's.....", end="")

    fig, ax = plt.subplots(ncols=len(ids), figsize=(len(ids) * PLOT_SIZE, PLOT_SIZE))

    spikes_T = np.array(spikes).T

    for i in range(len(ids)):
        # indices when excitatory neurons fired
        spikes_indices = np.argwhere(
            spikes_T[1] == ids[i]
        ).flatten()
        # times when excitatory neurons fired
        spikes_times = spikes_T[0][spikes_indices]
        spikes_per_times = [np.count_nonzero(spikes_indices == t) for t in range(simulation_time)]

        ax[i].plot(list(range(simulation_time)), spikes_per_times, solid_capstyle='round', color="#FFA3AF")
        ax[i].title.set_text(f"Neuron {ids[i]}")

    fig.savefig(path, bbox_inches='tight')
    plt.close()

    print(end="\r", flush=True)
    print(f"Plotting ended, result: {path}")

def fetch_spikes_over_time():
    return mpimg.imread(PlotPaths.SPIKES_OVER_TIME.value + PLOT_FORMAT)


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


def plot_ping_frequencies(frequencies, t_ms=-1):
    # TODO:: make pretty

    if t_ms == -1:
        print("Plotting current-frequency.....", end="")
        path = f"{PlotPaths.FREQUENCY_DISTRIBUTION.value}{PLOT_FORMAT}"
    else:
        path = f"{PlotPaths.FREQUENCY_DISTRIBUTION_EVOLUTION.value}/{t_ms}ms{PLOT_FORMAT}"

    fig, ax = plt.subplots(ncols=2, figsize=(2 * PLOT_SIZE, PLOT_SIZE))

    ax[0].hist(frequencies, color="#ACDDE7", rwidth=0.7)
    sns.heatmap(
        np.array(frequencies).reshape(int(sqrt(len(frequencies))), int(sqrt(len(frequencies)))),
        annot=True,
        square=True,
        ax=ax[1]
    )
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
    return [mpimg.imread(PlotPaths.FREQUENCY_DISTRIBUTION_EVOLUTION.value + "/" + f) for f in filenames]


def plot_single_ping_frequency_evolution(ping_freq_evol, time_fist_spike):

    print("Plotting single PING's frequency evolution.....", end="")
    path = f"{PlotPaths.FREQUENCY_SINGLE_PING_EVOLUTION.value}{PLOT_FORMAT}"

    fig, ax = plt.subplots(figsize=(PLOT_SIZE, PLOT_SIZE))

    time = list(range(time_fist_spike, len(ping_freq_evol) + time_fist_spike))
    plt.plot(time, ping_freq_evol, solid_capstyle='round', color="#FFA3AF")
    plt.xlabel("Time")
    plt.ylabel("Frequency")

    fig.savefig(path, bbox_inches='tight')
    plt.close()

    print(end="\r", flush=True)
    print(f"Plotting ended, result: {path}")


def fetch_single_ping_frequency_evolution():
    return mpimg.imread(PlotPaths.FREQUENCY_SINGLE_PING_EVOLUTION.value + PLOT_FORMAT)


