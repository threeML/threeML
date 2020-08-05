from __future__ import division
from builtins import zip
from builtins import range
from past.utils import old_div
import matplotlib.pyplot as plt
import numpy as np

from threeML.config.config import threeML_config
from threeML.io.plotting.step_plot import step_plot


# this file contains routines for plotting binned light curves


def binned_light_curve_plot(
    time_bins, cnts, width, bkg=None, selection=None, bkg_selections=None
):
    """

    :param time_bins: stacked array of time intervals
    :param cnts: counts per bin
    :param bkg: background of the light curve
    :param width: with of the bins
    :param selection: bin selection
    :param bkg_selections:
    :param instrument:

    :return:
    """
    fig, ax = plt.subplots()

    top = max(old_div(cnts, width)) * 1.2
    min_cnts = min(old_div(cnts[cnts > 0], width[cnts > 0])) * 0.95
    bottom = min_cnts
    mean_time = np.mean(time_bins, axis=1)

    all_masks = []

    # round
    np.round(time_bins, decimals=4, out=time_bins)

    light_curve_color = threeML_config["lightcurve"]["lightcurve color"]
    selection_color = threeML_config["lightcurve"]["selection color"]
    background_color = threeML_config["lightcurve"]["background color"]
    background_selection_color = threeML_config["lightcurve"][
        "background selection color"
    ]

    # first plot the full lightcurve

    step_plot(
        time_bins,
        old_div(cnts, width),
        ax,
        color=light_curve_color,
        label="Light Curve",
    )

    if selection is not None:

        # now plot the temporal selections

        np.round(selection, decimals=4, out=selection)

        for tmin, tmax in selection:
            tmp_mask = np.logical_and(time_bins[:, 0] >= tmin, time_bins[:, 1] <= tmax)

            all_masks.append(tmp_mask)

        if len(all_masks) > 1:

            for mask in all_masks[1:]:
                step_plot(
                    time_bins[mask],
                    old_div(cnts[mask], width[mask]),
                    ax,
                    color=selection_color,
                    fill=True,
                    fill_min=min_cnts,
                )

        step_plot(
            time_bins[all_masks[0]],
            old_div(cnts[all_masks[0]], width[all_masks[0]]),
            ax,
            color=selection_color,
            fill=True,
            fill_min=min_cnts,
            label="Selection",
        )

    # now plot the background selections

    if bkg_selections is not None:

        np.round(bkg_selections, decimals=4, out=bkg_selections)

        all_masks = []
        for tmin, tmax in bkg_selections:
            tmp_mask = np.logical_and(time_bins[:, 0] >= tmin, time_bins[:, 1] <= tmax)

            all_masks.append(tmp_mask)

        if len(all_masks) > 1:

            for mask in all_masks[1:]:
                step_plot(
                    time_bins[mask],
                    old_div(cnts[mask], width[mask]),
                    ax,
                    color=background_selection_color,
                    fill=True,
                    alpha=0.4,
                    fill_min=min_cnts,
                )

        step_plot(
            time_bins[all_masks[0]],
            old_div(cnts[all_masks[0]], width[all_masks[0]]),
            ax,
            color=background_selection_color,
            fill=True,
            fill_min=min_cnts,
            alpha=0.4,
            label="Bkg. Selections",
            zorder=-30,
        )

    if bkg is not None:
        # now plot the estimated background

        ax.plot(mean_time, bkg, background_color, lw=2.0, label="Background")

    # ax.fill_between(selection, bottom, top, color="#fc8d62", alpha=.4)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Rate (cnts/s)")
    ax.set_ylim(bottom, top)
    ax.set_xlim(time_bins.min(), time_bins.max())
    ax.legend()

    return fig


def channel_plot(ax, chan_min, chan_max, counts, **kwargs):
    chans = np.vstack([chan_min, chan_max]).T
    width = chan_max - chan_min

    
    step_plot(chans, old_div(counts, width), ax, **kwargs)
    ax.set_xscale("log")
    ax.set_yscale("log")

    return ax


def disjoint_patch_plot(ax, bin_min, bin_max, top, bottom, mask, **kwargs):
    # type: (plt.Axes, np.array, np.array, float, float, np.array, dict) -> None
    """

    plots patches that are disjoint given by the mask

    :param ax: matplotlib Axes to plot to
    :param bin_min: bin starts
    :param bin_max: bin stops
    :param top: top y value to plot
    :param bottom: bottom y value to plot
    :param mask: mask of the bins
    :param kwargs: matplotlib plot keywords
    :return:
    """
    # Figure out the best limit

    # Find the contiguous regions that are selected

    non_zero = (mask).nonzero()[0]

    if len(non_zero) > 0:

        slices = slice_disjoint(non_zero)

        for region in slices:
            ax.fill_between(
                [bin_min[region[0]], bin_max[region[1]]], bottom, top, **kwargs
            )

        ax.set_ylim(bottom, top)


def slice_disjoint(arr):
    """
    Returns an array of disjoint indices from a bool array

    :param arr: and array of bools


    """

    slices = []
    start_slice = arr[0]
    counter = 0
    for i in range(len(arr) - 1):
        if arr[i + 1] > arr[i] + 1:
            end_slice = arr[i]
            slices.append([start_slice, end_slice])
            start_slice = arr[i + 1]
            counter += 1
    if counter == 0:
        return [[arr[0], arr[-1]]]
    if end_slice != arr[-1]:
        slices.append([start_slice, arr[-1]])
    return slices


def plot_tte_lightcurve(tte_file, start=-10, stop=50, dt=1):
    # type: (str, float, float, float) -> plt.Figure

    """
    quick plot of a TTE light curve
    :param tte_file: GBM TTE file name
    :param start: start of the light curve
    :param stop: stop of the light curve
    :param dt: with of the bins


    """

    # build a quick object that will extract the data
    # the local import is because GBMTTEFile is dependent
    # on other files
    from threeML.plugins.FermiGBMTTELike import GBMTTEFile

    tte = GBMTTEFile(ttefile=tte_file)

    # bin the data with np hist

    bins = np.arange(start, stop, step=dt)

    counts, bins = np.histogram(tte.arrival_times - tte.trigger_time, bins=bins)

    width = np.diff(bins)

    time_bins = np.array(list(zip(bins[:-1], bins[1:])))

    # plot the light curve

    binned_light_curve_plot(time_bins=time_bins, cnts=counts, width=width)
