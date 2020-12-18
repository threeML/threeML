from __future__ import division
from builtins import zip
from past.utils import old_div
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

import threeML.plugins.SpectrumLike
import threeML.plugins.PhotometryLike
from threeML.io.plotting.cmap_cycle import cmap_intervals
from threeML.exceptions.custom_exceptions import custom_warnings
from threeML.config.config import threeML_config
from threeML.io.plotting.step_plot import step_plot
from threeML.io.plotting.data_residual_plot import ResidualPlot

# This file contains plots which are plotted in data space after a model has been
# assigned to the plugin.

NO_REBIN = 1e-99


def display_spectrum_model_counts(analysis, data=(), **kwargs):
    """

    Display the fitted model count spectrum of one or more Spectrum plugins

    NOTE: all parameters passed as keyword arguments that are not in the list below, will be passed as keyword arguments
    to the plt.subplots() constructor. So for example, you can specify the size of the figure using figsize = (20,10)

    :param args: one or more instances of Spectrum plugin
    :param min_rate: (optional) rebin to keep this minimum rate in each channel (if possible). If one number is
    provided, the same minimum rate is used for each dataset, otherwise a list can be provided with the minimum rate
    for each dataset
    :param data_cmap: (str) (optional) the color map used to extract automatically the colors for the data
    :param model_cmap: (str) (optional) the color map used to extract automatically the colors for the models
    :param data_colors: (optional) a tuple or list with the color for each dataset
    :param model_colors: (optional) a tuple or list with the color for each folded model
    :param data_color: (optional) color for all datasets
    :param model_color: (optional) color for all folded models
    :param show_legend: (optional) if True (default), shows a legend
    :param step: (optional) if True (default), show the folded model as steps, if False, the folded model is plotted
    :param model_subplot: (optional) axe(s) to plot to for overplotting
    with linear interpolation between each bin
    :param data_per_plot: (optional) Can spezify how many detectors should be plotted in one plot. If there
    are more detectors than this number it will split it up in several plots
    :param show_background: (optional) Also show the background
    :param source_only: (optional) Plot only source (total data - background)
    :param background_cmap: (str) (optional) the color map used to extract automatically the colors for the background
    :param background_colors: (optional) a tuple or list with the color for each background
    :param background_color: (optional) color for all backgrounds
    :return: figure instance


    """

    # If the user supplies a subset of the data, we will use that

    if not data:

        data_keys = list(analysis.data_list.keys())

    else:

        data_keys = data

    # Now we want to make sure that we only grab OGIP plugins

    new_data_keys = []

    for key in data_keys:

        # Make sure it is a valid key
        if key in list(analysis.data_list.keys()):

            if isinstance(
                analysis.data_list[key], threeML.plugins.SpectrumLike.SpectrumLike
            ):

                new_data_keys.append(key)

            else:

                custom_warnings.warn(
                    "Dataset %s is not of the SpectrumLike kind. Cannot be plotted by "
                    "display_spectrum_model_counts" % key
                )

    if not new_data_keys:
        RuntimeError(
            "There were no valid SpectrumLike data requested for plotting. Please use the detector names in the data list"
        )

    data_keys = new_data_keys

    # default settings

    # Default is to show the model with steps
    step = True

    data_cmap = threeML_config["ogip"]["data plot cmap"]  # plt.cm.rainbow
    model_cmap = threeML_config["ogip"]["model plot cmap"]  # plt.cm.nipy_spectral_r

    # Legend is on by default
    show_legend = True

    show_residuals = True

    # Default colors

    data_colors = cmap_intervals(len(data_keys), data_cmap)
    model_colors = cmap_intervals(len(data_keys), model_cmap)
    background_colors = cmap_intervals(len(data_keys), model_cmap)

    # Now override defaults according to the optional keywords, if present

    if "show_data" in kwargs:

        show_data = bool(kwargs.pop("show_data"))

    else:

        show_data = True

    if "show_legend" in kwargs:
        show_legend = bool(kwargs.pop("show_legend"))

    if "show_residuals" in kwargs:
        show_residuals = bool(kwargs.pop("show_residuals"))

    if "step" in kwargs:
        step = bool(kwargs.pop("step"))

    if "min_rate" in kwargs:

        min_rate = kwargs.pop("min_rate")

        # If min_rate is a floating point, use the same for all datasets, otherwise use the provided ones

        try:

            min_rate = float(min_rate)

            min_rates = [min_rate] * len(data_keys)

        except TypeError:

            min_rates = list(min_rate)

            assert len(min_rates) >= len(data_keys), (
                "If you provide different minimum rates for each data set, you need"
                "to provide an iterable of the same length of the number of datasets"
            )

    else:

        # This is the default (no rebinning)

        min_rates = [NO_REBIN] * len(data_keys)
        
    if "data_per_plot" in kwargs:
        data_per_plot = int(kwargs.pop("data_per_plot"))
    else:
        data_per_plot = len(data_keys)

    if "data_cmap" in kwargs:
        if len(data_keys) <= data_per_plot:
            data_colors = cmap_intervals(len(data_keys), kwargs.pop("data_cmap"))
        else:
            data_colors_base = cmap_intervals(data_per_plot, kwargs.pop("data_cmap"))
            data_colors = []
            for i in range(len(data_keys)):
                data_colors.append(data_colors_base[i % data_per_plot])

    elif "data_colors" in kwargs:
        data_colors = kwargs.pop("data_colors")

        assert len(data_colors) >= len(data_keys), (
            "You need to provide at least a number of data colors equal to the "
            "number of datasets"
        )

    elif "data_color" in kwargs:

        data_colors = [kwargs.pop("data_color")] * len(data_keys)

    if "model_cmap" in kwargs:
        if len(data_keys) <= data_per_plot:
            model_colors = cmap_intervals(len(data_keys),
                                          kwargs.pop("model_cmap"))
        else:
            model_colors_base = cmap_intervals(data_per_plot,
                                               kwargs.pop("model_cmap"))
            model_colors = []
            for i in range(len(data_keys)):
                model_colors.append(model_colors_base[i % data_per_plot])

    elif "model_colors" in kwargs:
        model_colors = kwargs.pop("model_colors")

        assert len(model_colors) >= len(data_keys), (
            "You need to provide at least a number of model colors equal to the "
            "number of datasets"
        )

    elif "model_color" in kwargs:

        model_colors = [kwargs.pop("model_color")] * len(data_keys)

    if "background_cmap" in kwargs:
        if len(data_keys) <= data_per_plot:
            background_colors = cmap_intervals(len(data_keys),
                                          kwargs.pop("background_cmap"))
        else:
            background_colors_base = cmap_intervals(data_per_plot,
                                               kwargs.pop("background_cmap"))
            background_colors = []
            for i in range(len(data_keys)):
                background_colors.append(background_colors_base[i % data_per_plot])

    elif "background_colors" in kwargs:
        background_colors = kwargs.pop("background_colors")

        assert len(background_colors) >= len(data_keys), (
            "You need to provide at least a number of background colors equal to the "
            "number of datasets"
        )
    elif "background_color" in kwargs:

        background_colors = [kwargs.pop("background_color")] * len(data_keys)

    ratio_residuals = False
    if "ratio_residuals" in kwargs:
        ratio_residuals = bool(kwargs["ratio_residuals"])

    if "model_labels" in kwargs:

        model_labels = kwargs.pop("model_labels")

        assert len(model_labels) == len(
            data_keys
        ), "you must have the same number of model labels as data sets"

    else:

        model_labels = ["%s Model" % analysis.data_list[key]._name for key in data_keys]

    if "background_labels" in kwargs:

        background_labels = kwargs.pop("background_labels")

        assert len(background_labels) == len(
            data_keys
        ), "you must have the same number of background labels as data sets"

    else:

        background_labels = ["%s Background" % analysis.data_list[key]._name for key in data_keys]


    if "source_only" in kwargs:

        source_only = kwargs.pop("source_only")

        assert type(source_only) == bool, "source_only must be a boolean"

    else:

        source_only = True

    if "show_background" in kwargs:

        show_background = kwargs.pop("show_background")

        assert type(show_background) == bool, "show_background must be a boolean"

    else:

        show_background = False


    if len(data_keys) <= data_per_plot:
        # If less than data_per_plot detectors need to be plotted,
        # just plot it in one plot
        residual_plot = ResidualPlot(show_residuals=show_residuals, **kwargs)

        if show_residuals:

            axes = [residual_plot.data_axis, residual_plot.residual_axis]

        else:

            axes = residual_plot.data_axis

        # go thru the detectors
        for key, data_color, model_color, background_color, min_rate, model_label, background_label in zip(
                data_keys, data_colors, model_colors, background_colors, min_rates, model_labels, background_labels
        ):

            # NOTE: we use the original (unmasked) vectors because we need to rebin ourselves the data later on

            data = analysis.data_list[
                key
            ]  # type: threeML.plugins.SpectrumLike.SpectrumLike

            data.display_model(
                data_color=data_color,
                model_color=model_color,
                min_rate=min_rate,
                step=step,
                show_residuals=show_residuals,
                show_data=show_data,
                show_legend=show_legend,
                ratio_residuals=ratio_residuals,
                model_label=model_label,
                model_subplot=axes,
                show_background=show_background,
                source_only=source_only,
                background_color=background_color,
                background_label=background_label
            )

        return residual_plot.figure

    else:
        # Too many detectors to plot everything in one plot... Make indivi.
        # plots with data_per_plot dets per plot

        # How many plots do we need?
        n_plots = int(np.ceil(1.*len(data_keys)/data_per_plot))

        plots = []
        for i in range(n_plots):
            plots.append(ResidualPlot(show_residuals=show_residuals, **kwargs))

        # go thru the detectors
        for j, (key, data_color, model_color, background_color, min_rate, model_label, background_label) in enumerate(zip(
                data_keys, data_colors, model_colors, background_colors, min_rates, model_labels, background_labels
        )):
            axes = [plots[int(j/data_per_plot)].data_axis,
                    plots[int(j/data_per_plot)].residual_axis]
            # NOTE: we use the original (unmasked) vectors because we need to rebin ourselves the data later on

            data = analysis.data_list[
                key
            ]  # type: threeML.plugins.SpectrumLike.SpectrumLike

            data.display_model(
                data_color=data_color,
                model_color=model_color,
                min_rate=min_rate,
                step=step,
                show_residuals=show_residuals,
                show_data=show_data,
                show_legend=show_legend,
                ratio_residuals=ratio_residuals,
                model_label=model_label,
                model_subplot=axes,
                show_background=show_background,
                source_only=source_only,
                background_color=background_color,
                background_label=background_label
            )

        figs = []
        for p in plots:
            figs.append(p.figure)

        return figs

def display_photometry_model_magnitudes(analysis, data=(), **kwargs):
    """

    Display the fitted model count spectrum of one or more Spectrum plugins

    NOTE: all parameters passed as keyword arguments that are not in the list below, will be passed as keyword arguments
    to the plt.subplots() constructor. So for example, you can specify the size of the figure using figsize = (20,10)

    :param args: one or more instances of Spectrum plugin
    :param min_rate: (optional) rebin to keep this minimum rate in each channel (if possible). If one number is
    provided, the same minimum rate is used for each dataset, otherwise a list can be provided with the minimum rate
    for each dataset
    :param data_cmap: (str) (optional) the color map used to extract automatically the colors for the data
    :param model_cmap: (str) (optional) the color map used to extract automatically the colors for the models
    :param data_colors: (optional) a tuple or list with the color for each dataset
    :param model_colors: (optional) a tuple or list with the color for each folded model
    :param show_legend: (optional) if True (default), shows a legend
    :param step: (optional) if True (default), show the folded model as steps, if False, the folded model is plotted
    with linear interpolation between each bin
    :return: figure instance


    """

    # If the user supplies a subset of the data, we will use that

    if not data:

        data_keys = list(analysis.data_list.keys())

    else:

        data_keys = data

    # Now we want to make sure that we only grab OGIP plugins

    new_data_keys = []

    for key in data_keys:

        # Make sure it is a valid key
        if key in list(analysis.data_list.keys()):

            if isinstance(
                analysis.data_list[key], threeML.plugins.PhotometryLike.PhotometryLike
            ):

                new_data_keys.append(key)

            else:

                custom_warnings.warn(
                    "Dataset %s is not of the Photometery kind. Cannot be plotted by "
                    "display_photometry_model_magnitudes" % key
                )

    if not new_data_keys:
        RuntimeError(
            "There were no valid Photometry data requested for plotting. Please use the detector names in the data list"
        )

    data_keys = new_data_keys

    # Default is to show the model with steps
    step = True

    data_cmap = threeML_config["photo"]["data plot cmap"]  # plt.cm.rainbow
    model_cmap = threeML_config["photo"]["model plot cmap"]  # plt.cm.nipy_spectral_r

    # Legend is on by default
    show_legend = True

    # Default colors

    data_colors = cmap_intervals(len(data_keys), data_cmap)
    model_colors = cmap_intervals(len(data_keys), model_cmap)

    # Now override defaults according to the optional keywords, if present

    if "show_legend" in kwargs:
        show_legend = bool(kwargs.pop("show_legend"))

    if "step" in kwargs:
        step = bool(kwargs.pop("step"))

    if "data_cmap" in kwargs:
        data_cmap = plt.get_cmap(kwargs.pop("data_cmap"))
        data_colors = cmap_intervals(len(data_keys), data_cmap)

    if "model_cmap" in kwargs:
        model_cmap = kwargs.pop("model_cmap")
        model_colors = cmap_intervals(len(data_keys), model_cmap)

    if "data_colors" in kwargs:
        data_colors = kwargs.pop("data_colors")

        assert len(data_colors) >= len(data_keys), (
            "You need to provide at least a number of data colors equal to the "
            "number of datasets"
        )

    if "model_colors" in kwargs:
        model_colors = kwargs.pop("model_colors")

        assert len(model_colors) >= len(data_keys), (
            "You need to provide at least a number of model colors equal to the "
            "number of datasets"
        )

    residual_plot = ResidualPlot(**kwargs)

    # go thru the detectors
    for key, data_color, model_color in zip(data_keys, data_colors, model_colors):

        data = analysis.data_list[
            key
        ]  # type: threeML.plugins.PhotometryLike.PhotometryLike

        # get the expected counts

        avg_wave_length = (
            data._filter_set.effective_wavelength.value
        )  # type: np.ndarray

        # need to sort because filters are not always in order

        sort_idx = avg_wave_length.argsort()

        expected_model_magnitudes = data._get_total_expectation()[sort_idx]
        magnitudes = data.magnitudes[sort_idx]
        mag_errors = data.magnitude_errors[sort_idx]
        avg_wave_length = avg_wave_length[sort_idx]

        residuals = old_div((expected_model_magnitudes - magnitudes), mag_errors)

        widths = data._filter_set.wavelength_bounds.widths[sort_idx]

        residual_plot.add_data(
            x=avg_wave_length,
            y=magnitudes,
            xerr=widths,
            yerr=mag_errors,
            residuals=residuals,
            label=data._name,
            color=data_color,
        )

        residual_plot.add_model(
            avg_wave_length,
            expected_model_magnitudes,
            label="%s Model" % data._name,
            color=model_color,
        )

        return residual_plot.finalize(
            xlabel="Wavelength\n(%s)" % data._filter_set.waveunits,
            ylabel="Magnitudes",
            xscale="linear",
            yscale="linear",
            invert_y=True,
        )


# def display_histogram_fit(analysis, data=(), **kwargs):
#     if not data:
#
#         data_keys = analysis.data_list.keys()
#
#     else:
#
#         data_keys = data
#
#     # Now we want to make sure that we only grab OGIP plugins
#
#     new_data_keys = []
#
#     for key in data_keys:
#
#         # Make sure it is a valid key
#         if key in analysis.data_list.keys():
#
#             if isinstance(analysis.data_list[key], threeML.plugins.HistLike.HistLike):
#
#                 new_data_keys.append(key)
#
#             else:
#
#                 custom_warnings.warn("Dataset %s is not of the HistLike kind. Cannot be plotted by "
#                                      "display_histogram_fit" % key)
#
#     if not new_data_keys:
#         RuntimeError(
#             'There were no valid HistLike data requested for plotting. Please use the names in the data list')
#
#     data_keys = new_data_keys
#
#     # default settings
#
#     # Default is to show the model with steps
#     step = True
#
#     data_cmap = plt.get_cmap(threeML_config['ogip']['data plot cmap'])  # plt.cm.rainbow
#     model_cmap = plt.get_cmap(threeML_config['ogip']['model plot cmap'])  # plt.cm.nipy_spectral_r
#
#     # Legend is on by default
#     show_legend = True
#
#     log_axes = False
#
#     # Default colors
#
#     data_colors = map(lambda x: data_cmap(x), np.linspace(0.0, 1.0, len(data_keys)))
#     model_colors = map(lambda x: model_cmap(x), np.linspace(0.0, 1.0, len(data_keys)))
#
#     # Now override defaults according to the optional keywords, if present
#
#     if 'show_legend' in kwargs:
#         show_legend = bool(kwargs.pop('show_legend'))
#
#     if 'step' in kwargs:
#         step = bool(kwargs.pop('step'))
#
#     if 'log_axes' in kwargs:
#         log_axes = True
#
#     if 'data_cmap' in kwargs:
#         data_cmap = plt.get_cmap(kwargs.pop('data_cmap'))
#         data_colors = map(lambda x: data_cmap(x), np.linspace(0.0, 1.0, len(data_keys)))
#
#     if 'model_cmap' in kwargs:
#         model_cmap = kwargs.pop('model_cmap')
#         model_colors = map(lambda x: model_cmap(x), np.linspace(0.0, 1.0, len(data_keys)))
#
#     if 'data_colors' in kwargs:
#         data_colors = kwargs.pop('data_colors')
#
#         assert len(data_colors) >= len(data_keys), "You need to provide at least a number of data colors equal to the " \
#                                                    "number of datasets"
#
#     if 'model_colors' in kwargs:
#         model_colors = kwargs.pop('model_colors')
#
#         assert len(model_colors) >= len(
#             data_keys), "You need to provide at least a number of model colors equal to the " \
#                         "number of datasets"
#
#     fig, (ax, ax1) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [2, 1]}, **kwargs)
#
#     # go thru the detectors
#     for key, data_color, model_color in zip(data_keys, data_colors, model_colors):
#
#         data = analysis.data_list[key]
#
#         x_min, x_max = data.histogram.absolute_start, data.histogram.absolute_stop
#
#         # Observed counts
#         observed_counts = data.histogram.contents
#
#         if data.is_poisson:
#
#             cnt_err = np.sqrt(observed_counts)
#
#         elif data.has_errors:
#
#             cnt_err = data.histogram.errors
#
#         width = data.histogram.widths
#
#         expected_model = data.get_model()
#
#         mean_x = []
#
#         # For each bin find the weighted average of the channel center
#
#         delta_x = [[], []]
#
#         for bin in data.histogram:
#
#             # Find all channels in this rebinned bin
#             idx = (data.histogram.mid_points >= bin.start) & (data.histogram.mid_points <= bin.stop)
#
#             # Find the rates for these channels
#             r = expected_model[idx]
#
#             if r.max() == 0:
#
#                 # All empty, cannot weight
#                 this_mean = bin.mid_point
#
#             else:
#
#                 # Do the weighted average of the mean energies
#                 weights = r / np.sum(r)
#
#                 this_mean = np.average(data.histogram.mid_points[idx], weights=weights)
#
#             # Compute "errors" for X (which aren't really errors, just to mark the size of the bin)
#
#             delta_x[0].append(this_mean - bin.start)
#             delta_x[1].append(bin.stop - this_mean)
#             mean_x.append(this_mean)
#
#         if data.has_errors:
#
#             ax.errorbar(mean_x,
#                         data.histogram.contents / width,
#                         yerr=cnt_err / width,
#                         xerr=delta_x,
#                         fmt='.',
#                         markersize=3,
#                         linestyle='',
#                         # elinewidth=.5,
#                         alpha=.9,
#                         capsize=0,
#                         label=data._name,
#                         color=data_color)
#
#         else:
#
#             ax.errorbar(mean_x,
#                         data.histogram.contents / width,
#                         xerr=delta_x,
#                         fmt='.',
#                         markersize=3,
#                         linestyle='',
#                         # elinewidth=.5,
#                         alpha=.9,
#                         capsize=0,
#                         label=data._name,
#                         color=data_color)
#
#         if step:
#
#             step_plot(data.histogram.bin_stack,
#                       expected_model / width,
#                       ax, alpha=.8,
#                       label='%s Model' % data._name, color=model_color)
#
#         else:
#
#             ax.plot(data.histogram.mid_points, expected_model / width, alpha=.8, label='%s Model' % data._name,
#                     color=model_color)
#
#         if data.is_poisson:
#
#             # this is not correct I believe
#
#             residuals = data.histogram.contents - expected_model
#
#         else:
#
#             if data.has_errors:
#
#                 residuals = (data.histogram.contents - expected_model) / data.histogram.errors
#
#             else:
#
#                 residuals = data.histogram.contents - expected_model
#
#         ax1.axhline(0, linestyle='--', color='k')
#         ax1.errorbar(mean_x,
#                      residuals,
#                      yerr=np.ones_like(residuals),
#                      capsize=0,
#                      fmt='.',
#                      markersize=3,
#                      color=data_color)
#
#     if show_legend:
#         ax.legend(fontsize='x-small', loc=0)
#
#     ax.set_ylabel("Y")
#
#     if log_axes:
#         ax.set_xscale('log')
#         ax.set_yscale('log', nonposy='clip')
#
#         ax1.set_xscale("log")
#
#     locator = MaxNLocator(prune='upper', nbins=5)
#     ax1.yaxis.set_major_locator(locator)
#
#     ax1.set_xlabel("X")
#     ax1.set_ylabel("Residuals\n($\sigma$)")
#
#     # This takes care of making space for all labels around the figure
#
#     fig.tight_layout()
#
#     # Now remove the space between the two subplots
#     # NOTE: this must be placed *after* tight_layout, otherwise it will be ineffective
#
#     fig.subplots_adjust(hspace=0)
#
#     return fig
#
#
