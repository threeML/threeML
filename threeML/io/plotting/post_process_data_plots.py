import matplotlib.pyplot as plt
import numpy as np
import threeML.plugins.PhotometryLike as photolike
import threeML.plugins.SpectrumLike as speclike


try:
    from threeML.plugins.FermiLATLike import FermiLATLike

    LATLike = True
except:
    LATLike = False

from threeML.config.config import threeML_config
from threeML.config.plotting_structure import BinnedSpectrumPlot
from threeML.exceptions.custom_exceptions import custom_warnings
from threeML.io.logging import setup_logger
from threeML.io.package_data import get_path_of_data_file
from threeML.io.plotting.cmap_cycle import cmap_intervals
from threeML.io.plotting.data_residual_plot import ResidualPlot
from threeML.io.plotting.step_plot import step_plot

plt.style.use(str(get_path_of_data_file("threeml.mplstyle")))

log = setup_logger(__name__)

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
    :param data_per_plot: (optional) Can specify how many detectors should be plotted in one plot. If there
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

            if isinstance(analysis.data_list[key], speclike.SpectrumLike):
                new_data_keys.append(key)
            elif LATLike and isinstance(analysis.data_list[key], FermiLATLike):
                new_data_keys.append(key)
            else:
                log.warning(
                    "Dataset %s is not of the SpectrumLike or FermiLATLike  kind. Cannot be plotted by display_spectrum_model_counts"
                    % key
                )

    if not new_data_keys:

        log.error(
            "There were no valid SpectrumLike or FermiLATLike data requested for plotting. Please use the detector names in the data list"
        )

        RuntimeError(
            "There were no valid SpectrumLike or FermiLATLike data requested for plotting. Please use the detector names in the data list"
        )

    data_keys = new_data_keys

    # default settings

    _sub_menu: BinnedSpectrumPlot = threeML_config.plugins.ogip.fit_plot

    # Default is to show the model with steps
    step = _sub_menu.step

    data_cmap = _sub_menu.data_cmap.value
    model_cmap = _sub_menu.model_cmap.value
    background_cmap = _sub_menu.background_cmap.value

    # Legend is on by default
    show_legend = _sub_menu.show_legend

    show_residuals = _sub_menu.show_residuals

    show_background: bool = _sub_menu.show_background

    # Default colors

    _cmap_len = max(len(data_keys), _sub_menu.n_colors)

    data_colors = cmap_intervals(_cmap_len, data_cmap)
    model_colors = cmap_intervals(_cmap_len, model_cmap)
    background_colors = cmap_intervals(_cmap_len, background_cmap)

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

            if len(min_rates) < len(data_keys):
                log.error(
                    "If you provide different minimum rates for each data set, you need"
                    "to provide an iterable of the same length of the number of datasets"
                )
                raise ValueError()

    else:

        # This is the default (no rebinning)

        min_rates = [NO_REBIN] * len(data_keys)

    if "data_per_plot" in kwargs:
        data_per_plot = int(kwargs.pop("data_per_plot"))
    else:
        data_per_plot = len(data_keys)

    if "data_cmap" in kwargs:
        if len(data_keys) <= data_per_plot:

            _cmap_len = max(len(data_keys), _sub_menu.n_colors)

            data_colors = cmap_intervals(_cmap_len, kwargs.pop("data_cmap"))
        else:

            _cmap_len = max(data_per_plot, _sub_menu.n_colors)

            data_colors_base = cmap_intervals(
                _cmap_len, kwargs.pop("data_cmap")
            )
            data_colors = []
            for i in range(len(data_keys)):
                data_colors.append(data_colors_base[i % data_per_plot])

    elif "data_colors" in kwargs:
        data_colors = kwargs.pop("data_colors")

        if len(data_colors) < len(data_keys):
            log.error(
                "You need to provide at least a number of data colors equal to the "
                "number of datasets"
            )
            raise ValueError()

    elif _sub_menu.data_color is not None:

        data_colors = [_sub_menu.data_color] * len(data_keys)

    # always override
    if "data_color" in kwargs:

        data_colors = [kwargs.pop("data_color")] * len(data_keys)

    if "model_cmap" in kwargs:
        if len(data_keys) <= data_per_plot:

            _cmap_len = max(len(data_keys), _sub_menu.n_colors)

            model_colors = cmap_intervals(_cmap_len, kwargs.pop("model_cmap"))
        else:

            _cmap_len = max(data_per_plot, _sub_menu.n_colors)

            model_colors_base = cmap_intervals(
                _cmap_len, kwargs.pop("model_cmap")
            )
            model_colors = []
            for i in range(len(data_keys)):
                model_colors.append(model_colors_base[i % data_per_plot])

    elif "model_colors" in kwargs:
        model_colors = kwargs.pop("model_colors")

        if len(model_colors) < len(data_keys):
            log.error(
                "You need to provide at least a number of model colors equal to the "
                "number of datasets"
            )
            raise ValueError()

    elif _sub_menu.model_color is not None:

        model_colors = [_sub_menu.model_color] * len(data_keys)

    # always overide
    if "model_color" in kwargs:

        model_colors = [kwargs.pop("model_color")] * len(data_keys)

    if "background_cmap" in kwargs:
        if len(data_keys) <= data_per_plot:
            background_colors = cmap_intervals(
                len(data_keys), kwargs.pop("background_cmap")
            )
        else:
            background_colors_base = cmap_intervals(
                data_per_plot, kwargs.pop("background_cmap")
            )
            background_colors = []
            for i in range(len(data_keys)):
                background_colors.append(
                    background_colors_base[i % data_per_plot]
                )

    elif "background_colors" in kwargs:
        background_colors = kwargs.pop("background_colors")

        if len(background_colors) < len(data_keys):
            log.error(
                "You need to provide at least a number of background colors equal to the "
                "number of datasets"
            )
            raise ValueError()

    elif _sub_menu.background_color is not None:

        background_colors = [_sub_menu.background_color] * len(data_keys)

    # always override
    if "background_color" in kwargs:

        background_colors = [kwargs.pop("background_color")] * len(data_keys)

    ratio_residuals = False
    if "ratio_residuals" in kwargs:
        ratio_residuals = bool(kwargs["ratio_residuals"])

    if "model_labels" in kwargs:

        model_labels = kwargs.pop("model_labels")

        if len(model_labels) != len(data_keys):
            log.error(
                "You must have the same number of model labels as data sets"
            )
            raise ValueError()
    else:

        model_labels = [
            "%s Model" % analysis.data_list[key]._name for key in data_keys
        ]

    if "background_labels" in kwargs:

        background_labels = kwargs.pop("background_labels")

        if len(background_labels) != len(data_keys):
            log.error(
                "You must have the same number of background labels as data sets"
            )
            raise ValueError()

    else:

        background_labels = [
            "%s Background" % analysis.data_list[key]._name for key in data_keys
        ]

    if "source_only" in kwargs:

        source_only = kwargs.pop("source_only")

        if type(source_only) != bool:
            log.error("source_only must be a boolean")
            raise TypeError()

    else:

        source_only = True

    if "show_background" in kwargs:

        show_background = kwargs.pop("show_background")

        if type(show_background) != bool:
            log.error("show_background must be a boolean")
            raise TypeError()

    data_kwargs = None

    if "data_kwargs" in kwargs:

        data_kwargs = kwargs.pop("data_kwargs")

    model_kwargs = None

    if "model_kwargs" in kwargs:

        model_kwargs = kwargs.pop("model_kwargs")

    if len(data_keys) <= data_per_plot:
        # If less than data_per_plot detectors need to be plotted,
        # just plot it in one plot
        residual_plot = ResidualPlot(show_residuals=show_residuals, **kwargs)

        axes = residual_plot.axes

        # go thru the detectors
        for (
            key,
            data_color,
            model_color,
            background_color,
            min_rate,
            model_label,
            background_label,
        ) in zip(
            data_keys,
            data_colors,
            model_colors,
            background_colors,
            min_rates,
            model_labels,
            background_labels,
        ):

            # NOTE: we use the original (unmasked) vectors because we need to rebin ourselves the data later on

            data = analysis.data_list[key]  # type: speclike

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
                background_label=background_label,
                model_kwargs=model_kwargs,
                data_kwargs=data_kwargs,
            )

        return residual_plot.figure

    else:
        # Too many detectors to plot everything in one plot... Make indivi.
        # plots with data_per_plot dets per plot

        # How many plots do we need?
        n_plots = int(np.ceil(1.0 * len(data_keys) / data_per_plot))

        plots = []
        for i in range(n_plots):
            plots.append(ResidualPlot(show_residuals=show_residuals, **kwargs))

        # go thru the detectors
        for j, (
            key,
            data_color,
            model_color,
            background_color,
            min_rate,
            model_label,
            background_label,
        ) in enumerate(
            zip(
                data_keys,
                data_colors,
                model_colors,
                background_colors,
                min_rates,
                model_labels,
                background_labels,
            )
        ):
            axes = [
                plots[int(j / data_per_plot)].data_axis,
                plots[int(j / data_per_plot)].residual_axis,
            ]
            # NOTE: we use the original (unmasked) vectors because we need to rebin ourselves the data later on

            data = analysis.data_list[key]  # type: speclike

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
                background_label=background_label,
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

            if isinstance(analysis.data_list[key], photolike.PhotometryLike):

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

    if "show_data" in kwargs:

        show_data = bool(kwargs.pop("show_data"))

    else:

        show_data = True

    show_residuals = True

    if "show_residuals" in kwargs:

        show_residuals = kwargs.pop("show_residuals")

    # Default is to show the model with steps
    step = threeML_config.plugins.photo.fit_plot.step

    data_cmap = (
        threeML_config.plugins.photo.fit_plot.data_cmap.value
    )  # plt.cm.rainbow

    model_cmap = threeML_config.plugins.photo.fit_plot.model_cmap.value

    # Legend is on by default
    show_legend = True

    # Default colors

    data_colors = cmap_intervals(len(data_keys), data_cmap)
    model_colors = cmap_intervals(len(data_keys), model_cmap)

    if "data_color" in kwargs:

        data_colors = [kwargs.pop("data_color")] * len(data_keys)

    if "model_color" in kwargs:

        model_colors = [kwargs.pop("model_color")] * len(data_keys)

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

        if len(data_colors) < len(data_keys):
            log.error(
                "You need to provide at least a number of data colors equal to the "
                "number of datasets"
            )
            raise ValueError()

    if "model_colors" in kwargs:
        model_colors = kwargs.pop("model_colors")

        if len(model_colors) < len(data_keys):
            log.error(
                "You need to provide at least a number of model colors equal to the "
                "number of datasets"
            )
            raise ValueError()

    data_kwargs = None

    if "data_kwargs" in kwargs:

        data_kwargs = kwargs.pop("data_kwargs")

    model_kwargs = None

    if "model_kwargs" in kwargs:

        model_kwargs = kwargs.pop("model_kwargs")

    residual_plot = ResidualPlot(show_residuals=show_residuals, **kwargs)

    if "model_subplot" in kwargs:

        kwargs.pop("model_subplot")

    axes = residual_plot.axes

    # go thru the detectors
    for key, data_color, model_color in zip(
        data_keys, data_colors, model_colors
    ):

        data: photolike.PhotometryLike = analysis.data_list[key]

        data.plot(
            model_subplot=axes,
            model_color=model_color,
            data_color=data_color,
            model_kwargs=model_kwargs,
            data_kwargs=data_kwargs,
            show_residuals=show_residuals,
            show_legend=show_legend,
            **kwargs,
        )

    return residual_plot
