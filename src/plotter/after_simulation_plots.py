import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.image as mpimg
import seaborn as sns
import pandas as pd
from scipy import interpolate
import os
from math import sqrt

from src.plotter.setup import PlotPaths, PLOT_FORMAT, PLOT_SIZE, PLOT_COLORS


####################################################################################################


def plot_raster(all_spikes_ex, all_spikes_in):
    print("Plotting raster.....", end="")
    path = f"{PlotPaths.RASTER.value}{PLOT_FORMAT}"

    fig, ax = plt.subplots(figsize=(4 * PLOT_SIZE, PLOT_SIZE))

    ax.scatter(all_spikes_ex.T[0], all_spikes_ex.T[1], s=PLOT_SIZE / 5, color=PLOT_COLORS[0], label="EX")
    ax.scatter(all_spikes_in.T[0], all_spikes_in.T[1], s=PLOT_SIZE / 5, color=PLOT_COLORS[1], label="IN")
    ax.legend()
    ax.set_xlabel("time ms")
    ax.set_ylabel("neuron ID")
    ax.set_title("Spike raster")

    fig.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()

    print(end="\r", flush=True)
    print(f"Plotting ended, result: {path}")


####################################################################################################

def plot_number_neurons_spiked(windows: list[(int, int)], spikes_df: pd.DataFrame):
    print("Plotting number of neurons spiked.....", end="")
    path = f"{PlotPaths.NR_NEURONS_SPIKED.value}{PLOT_FORMAT}"

    mid_windows = [w[0] + (w[1] - w[0]) / 2 for w in windows]

    fig, ax = plt.subplots(figsize=(1.2 * PLOT_SIZE, 1.2 * PLOT_SIZE))

    heatmap = np.array([np.array(a) for a in spikes_df["nr_neurons_spiked_count"].to_numpy()]).T
    top_nonzero = np.max(np.nonzero(heatmap)[0]) + 1

    sns.heatmap(
        heatmap[:top_nonzero, :],
        annot=True,
        cbar=False,
        square=True,
        cmap="Blues",
        xticklabels=True,
        yticklabels=True,
        ax=ax
    )
    ax.set_xticklabels(mid_windows)
    ax.set_xlabel("mid window time, ms")
    ax.set_ylabel("spike count")
    ax.set_title("Number of neurons spiked")
    ax.invert_yaxis()

    fig.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()

    print(end="\r", flush=True)
    print(f"Plotting ended, result: {path}")


####################################################################################################


def plot_stats_per_ping(ping_networks, spikes_df, windows, step):
    print("Plotting statistics in each PING.....", end="")
    path = f"{PlotPaths.STATS_PER_PING.value}{PLOT_FORMAT}"

    ntypes = 3
    fig, ax = plt.subplots(
        nrows=ntypes,
        ncols=1 + len(ping_networks),
        sharex=True,
        figsize=(
            (1 + len(ping_networks)) * PLOT_SIZE,
            ntypes * PLOT_SIZE * 0.5
        )
    )
    plt.subplots_adjust(top=1.5, right=1.5)

    in_color = PLOT_COLORS[0]
    ex_color = PLOT_COLORS[1]
    none_color = PLOT_COLORS[2]

    mid_windows = [w[0] + (w[1] - w[0]) / 2 for w in windows]
    int_x = np.linspace(mid_windows[0], mid_windows[-1], num=300, endpoint=True)

    def int_y(arr):
        return interpolate.make_interp_spline(mid_windows, arr)(int_x)

    def plot_spikes(axis, spikes, spikes_ex, spikes_in, title="", ytitle="", ping_id=None, ping_location=None):
        axis.plot(int_x, int_y(spikes), c=none_color, label="both spikes", zorder=0)
        axis.scatter(mid_windows, spikes, s=10 * PLOT_SIZE, c="black", zorder=1)
        axis.plot(int_x, int_y(spikes_ex), c=ex_color, label="spikes EX", zorder=0)
        axis.scatter(mid_windows, spikes_ex, s=10 * PLOT_SIZE, c="black", zorder=1)
        axis.plot(int_x, int_y(spikes_in), c=in_color, label="spikes IN", zorder=0)
        axis.scatter(mid_windows, spikes_in, s=10 * PLOT_SIZE, c="black", zorder=1)
        axis.legend(loc="center right")
        axis.set_xlabel("mid window time, ms")
        axis.set_ylabel(ytitle)
        if not ping_id and not ping_location:
            axis.set_title(title)
        else:
            axis.set_title(f"{title}; PING {ping_id} at {ping_location}")

    nr_spikes_title = "Number of spikes"
    nr_spikes_ytitle = f"nr of spikes in a window of {step} ms"
    mean_spikes_title = "Mean number of spikes"
    mean_spikes_ytitle = f"mean nr of spikes per ms in a window of {step} ms"
    std_spikes_title = "STD of number of spikes"
    std_spikes_ytitle = f"std of nr of spikes per ms in a window of {step} ms"

    def draw_col_plots(ax_x, ping_id=None, ping_location=None):

        ping_id_str = "" if not ping_id else str(ping_id)
        plot_spikes(
            axis=ax[0][ax_x],
            spikes=spikes_df[f"nr_spikes{ping_id_str}"],
            spikes_ex=spikes_df[f"nr_spikes_ex{ping_id_str}"],
            spikes_in=spikes_df[f"nr_spikes_in{ping_id_str}"],
            title=nr_spikes_title,
            ytitle=nr_spikes_ytitle,
            ping_id=ping_id,
            ping_location=ping_location
        )

        plot_spikes(
            axis=ax[1][ax_x],
            spikes=spikes_df[f"mean_nr_spikes_per_ts{ping_id_str}"],
            spikes_ex=spikes_df[f"mean_nr_spikes_ex_per_ts{ping_id_str}"],
            spikes_in=spikes_df[f"mean_nr_spikes_in_per_ts{ping_id_str}"],
            title=mean_spikes_title,
            ytitle=mean_spikes_ytitle,
            ping_id=ping_id,
            ping_location=ping_location
        )

        plot_spikes(
            axis=ax[2][ax_x],
            spikes=spikes_df[f"std_nr_spikes_per_ts{ping_id_str}"],
            spikes_ex=spikes_df[f"std_nr_spikes_ex_per_ts{ping_id_str}"],
            spikes_in=spikes_df[f"std_nr_spikes_in_per_ts{ping_id_str}"],
            title=std_spikes_title,
            ytitle=std_spikes_ytitle,
            ping_id=ping_id,
            ping_location=ping_location
        )

    col_count = 0
    draw_col_plots(col_count)

    for ping_network in ping_networks:
        col_count += 1
        draw_col_plots(col_count, col_count - 1, ping_network.grid_location)

    fig.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()

    print(end="\r", flush=True)
    print(f"Plotting ended, result: {path}")


####################################################################################################


def plot_ping_frequencies(frequencies, t_ms=-1):
    # TODO:: make pretty

    if t_ms == -1:
        print("Plotting current-frequency.....", end="")
        path = f"{PlotPaths.FREQUENCY_DISTRIBUTION.value}{PLOT_FORMAT}"
    else:
        path = f"{PlotPaths.FREQUENCY_DISTRIBUTION_EVOLUTION.value}/{t_ms}ms{PLOT_FORMAT}"

    fig, ax = plt.subplots(ncols=2, figsize=(2 * PLOT_SIZE, PLOT_SIZE))

    ax[0].hist(frequencies, color=PLOT_COLORS[0], rwidth=0.7)
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


def display_ping_frequencies():
    fig, ax = plt.subplots(figsize=(2 * PLOT_SIZE, PLOT_SIZE))

    ax.imshow(fetch_ping_frequencies())
    ax.axis('off')
    plt.show()


####################################################################################################


def plot_phase_locking(phase_locking):
    print("Plotting phase-locking.....", end="")
    path = f"{PlotPaths.PHASE_LOCKING.value}{PLOT_FORMAT}"

    fig, ax = plt.subplots(figsize=(PLOT_SIZE, PLOT_SIZE))

    sns.heatmap(
        phase_locking,
        annot=False,
        cbar=True,
        square=True,
        cmap="Blues",
        vmin=0,
        vmax=1,
        xticklabels=False,
        yticklabels=False,
        ax=ax
    )
    ax.set_xlabel("IN ids")
    ax.set_ylabel("IN ids")
    ax.set_title(f"Phase-locking (mean off diag = {np.mean([phase_locking[i][j] for i in range(phase_locking.shape[0]) for j in range(phase_locking.shape[1]) if i != j]):.2f})")

    fig.savefig(path, bbox_inches='tight')
    plt.close()

    print(end="\r", flush=True)
    print(f"Plotting ended, result: {path}")

