from __future__ import division
from builtins import zip
from builtins import object
from past.utils import old_div
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from threeML.io.plotting.step_plot import step_plot
from threeML.config.config import threeML_config

from threeML.exceptions.custom_exceptions import custom_warnings
from threeML.io.logging import setup_logger

log = setup_logger(__name__)

class ResidualPlot(object):
    def __init__(self, **kwargs):
        """
        A class that makes data/residual plots

        :param show_residuals: to show the residuals
        :param ratio_residuals: to use ratios instead of sigma
        :param model_subplot: and axis or list of axes to plot to rather than create a new one
        """

        self._ratio_residuals = False
        self._show_residuals = True

        if "show_residuals" in kwargs:

            self._show_residuals = bool(kwargs.pop("show_residuals"))

        if "ratio_residuals" in kwargs:
            self._ratio_residuals = bool(kwargs.pop("ratio_residuals"))

        # this lets you overplot other fits

        if "model_subplot" in kwargs:

            model_subplot = kwargs.pop("model_subplot")

            # turn on or off residuals

            if self._show_residuals:

                assert (
                    type(model_subplot) == list
                ), "you must supply a list of axes to plot to residual"

                assert (
                    len(model_subplot) == 2
                ), "you have requested to overplot a model with residuals, but only provided one axis to plot"

                self._data_axis, self._residual_axis = model_subplot

            else:

                try:

                    self._data_axis = model_subplot

                    self._fig = self._data_axis.get_figure()

                except (AttributeError):

                    # the user supplied a list of axes

                    self._data_axis = model_subplot[0]

            # we will use the figure associated with
            # the data axis

            self._fig = self._data_axis.get_figure()

        else:

            # turn on or off residuals

            if self._show_residuals:

                self._fig, (self._data_axis, self._residual_axis) = plt.subplots(
                    2, 1, sharex=True, gridspec_kw={"height_ratios": [2, 1]}, **kwargs
                )

            else:

                self._fig, self._data_axis = plt.subplots(**kwargs)

    @property
    def figure(self):
        """

        :return: the figure instance
        """

        return self._fig

    @property
    def data_axis(self):
        """

        :return: the top or data axis
        """

        return self._data_axis

    @property
    def residual_axis(self):
        """

        :return: the bottom or residual axis
        """

        assert self._show_residuals, "this plot has no residual axis"

        return self._residual_axis

    @property
    def show_residuals(self):
        return self._show_residuals

    @property
    def ratio_residuals(self):
        return self._ratio_residuals

    def add_model_step(self, xmin, xmax, xwidth, y, label, **kwargs):
        """
        Add a model but use discontinuous steps for the plotting.

        :param xmin: the low end boundaries
        :param xmax: the high end boundaries
        :param xwidth: the width of the bins
        :param y: the height of the bins
        :param label: the label of the model
        :param **kwargs: any kwargs passed to plot
        :return: None
        """

        step_plot(
            np.asarray(list(zip(xmin, xmax))),
            old_div(y, xwidth),
            self._data_axis,
            label=label,
            **kwargs
        )

    def add_model(self, x, y, label, **kwargs):
        """
        Add a model and interpolate it across the energy span for the plotting.

        :param x: the evaluation energies
        :param y: the model values
        :param label: the label of the model
        :param **kwargs: any kwargs passed to plot
        :return: None
        """
        self._data_axis.plot(x, y, label=label, **kwargs)

    def add_data(
        self,
        x,
        y,
        residuals,
        label,
        xerr=None,
        yerr=None,
        residual_yerr=None,
        show_data=True,
        **kwargs
    ):
        """
        Add the data for the this model

        :param x: energy of the data
        :param y: value of the data
        :param residuals: the residuals for the data
        :param label: label of the data
        :param xerr: the error in energy (or bin width)
        :param yerr: the errorbars of the data
        :param **kwargs: any kwargs passed to plot
        :return:
        """

        # if we want to show the data

        if show_data:
            self._data_axis.errorbar(x, y, yerr=yerr, xerr=xerr, label=label, **kwargs)

        # if we want to show the residuals

        if self._show_residuals:

            # normal residuals from the likelihood

            if not self.ratio_residuals:

                residual_yerr = np.ones_like(residuals)

            self._residual_axis.axhline(0, linestyle="--", color="k")

            self._residual_axis.errorbar(x, residuals, yerr=residual_yerr, **kwargs)

    def finalize(
        self,
        xlabel="x",
        ylabel="y",
        xscale="log",
        yscale="log",
        show_legend=True,
        invert_y=False,
    ):
        """

        :param xlabel:
        :param ylabel:
        :param xscale:
        :param yscale:
        :param show_legend:
        :return:
        """

        if show_legend:
            self._data_axis.legend(fontsize="x-small", loc=0)

        self._data_axis.set_ylabel(ylabel)

        self._data_axis.set_xscale(xscale)
        if yscale == "log":

            self._data_axis.set_yscale(yscale, nonposy="clip")

        else:

            self._data_axis.set_yscale(yscale)

        if self._show_residuals:

            self._residual_axis.set_xscale(xscale)

            locator = MaxNLocator(prune="upper", nbins=5)
            self._residual_axis.yaxis.set_major_locator(locator)

            self._residual_axis.set_xlabel(xlabel)

            if self.ratio_residuals:
                log.warning(
                    "Residuals plotted as ratios: beware that they are not statistical quantites, and can not be used to asses fit quality"
                )
                self._residual_axis.set_ylabel("Residuals\n(fraction of model)")
            else:
                self._residual_axis.set_ylabel("Residuals\n($\sigma$)")

        else:

            self._data_axis.set_xlabel(xlabel)

            # This takes care of making space for all labels around the figure

        self._fig.tight_layout()

        # Now remove the space between the two subplots
        # NOTE: this must be placed *after* tight_layout, otherwise it will be ineffective

        self._fig.subplots_adjust(hspace=0)

        if invert_y:
            self._data_axis.set_ylim(self._data_axis.get_ylim()[::-1])

        return self._fig
