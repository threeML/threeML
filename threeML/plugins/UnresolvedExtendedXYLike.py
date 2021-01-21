import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astromodels import Model, PointSource

from threeML.classicMLE.goodness_of_fit import GoodnessOfFit
from threeML.classicMLE.joint_likelihood import JointLikelihood
from threeML.data_list import DataList
from threeML.exceptions.custom_exceptions import custom_warnings
from threeML.plugin_prototype import PluginPrototype
from threeML.plugins.XYLike import XYLike
from threeML.utils.statistics.likelihood_functions import (
    half_chi2, poisson_log_likelihood_ideal_bkg)

__instrument_name = "n.a."


class UnresolvedExtendedXYLike(XYLike):
    def __init__(
        self,
        name,
        x,
        y,
        yerr=None,
        exposure=None,
        poisson_data=False,
        quiet=False,
        source_name=None,
    ):

        super(UnresolvedExtendedXYLike, self).__init__(
            name=name,
            x=x,
            y=y,
            yerr=yerr,
            exposure=exposure,
            poisson_data=poisson_data,
            quiet=quiet,
            source_name=source_name,
        )

    def assign_to_source(self, source_name):
        """
        Assign these data to the given source (instead of to the sum of all sources, which is the default)

        :param source_name: name of the source (must be contained in the likelihood model)
        :return: none
        """

        if self._likelihood_model is not None and source_name is not None:

            assert source_name in self._likelihood_model.sources, (
                "Source %s is not contained in " "the likelihood model" % source_name
            )

        self._source_name = source_name

    def set_model(self, likelihood_model_instance):
        """
        Set the model to be used in the joint minimization. Must be a LikelihoodModel instance.

        :param likelihood_model_instance: instance of Model
        :type likelihood_model_instance: astromodels.Model
        """

        if likelihood_model_instance is None:

            return

        if self._source_name is not None:

            # Make sure that the source is in the model
            assert self._source_name in likelihood_model_instance.sources, (
                "This XYLike plugin refers to the source %s, "
                "but that source is not in the likelihood model" % (self._source_name)
            )

        self._likelihood_model = likelihood_model_instance

    def _get_total_expectation(self):

        if self._source_name is None:

            n_point_sources = self._likelihood_model.get_number_of_point_sources()
            n_ext_sources = self._likelihood_model.get_number_of_extended_sources()

            assert (
                n_point_sources + n_ext_sources > 0
            ), "You need to have at least one source defined"

            # Make a function which will stack all point sources (XYLike do not support spatial dimension)

            expectation_point = np.sum(
                [
                    source(self._x, tag=self._tag)
                    for source in list(self._likelihood_model.point_sources.values())
                ],
                axis=0,
            )

            expectation_ext = np.sum(
                [
                    source.get_spatially_integrated_flux(self._x)
                    for source in list(self._likelihood_model.extended_sources.values())
                ],
                axis=0,
            )

            expectation = expectation_point + expectation_ext

        else:

            # This XYLike dataset refers to a specific source

            # Note that we checked that self._source_name is in the model when the model was set

            if self._source_name in self._likelihood_model.point_sources:

                expectation = self._likelihood_model.point_sources[self._source_name](
                    self._x
                )

            elif self._source_name in self._likelihood_model.extended_sources:

                expectation = self._likelihood_model.extended_sources[
                    self._source_name
                ].get_spatially_integrated_flux(self._x)

            else:

                raise KeyError(
                    "This XYLike plugin has been assigned to source %s, "
                    "which is neither a point soure not an extended source in the current model"
                    % self._source_name
                )

        return expectation

    def plot(self, x_label="x", y_label="y", x_scale="linear", y_scale="linear"):

        fig, sub = plt.subplots(1, 1)

        sub.errorbar(self.x, self.y, yerr=self.yerr, fmt=".", label="data")

        sub.set_xscale(x_scale)
        sub.set_yscale(y_scale)

        sub.set_xlabel(x_label)
        sub.set_ylabel(y_label)

        if self._likelihood_model is not None:

            flux = self._get_total_expectation()

            label = (
                "model"
                if self._source_name is None
                else "model (%s)" % self._source_name
            )
            sub.plot(self.x, flux, "--", label=label)

            sub.legend(loc=0)

        return fig
