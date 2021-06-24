from __future__ import division, print_function

from builtins import object

import numpy as np
from astromodels import ModelAssertionViolation, use_astromodels_memoization
from astromodels.core.model import Model

from threeML.config import threeML_config
from threeML.data_list import DataList
from threeML.io.logging import setup_logger

log = setup_logger(__name__)


_possible_samplers = ["emcee", "mutinest", "zeus", "ultranest",
                      "dynesty_nested", "dynesty_dynamic", "autoemcee"]


_available_samplers = {}

try:

    import emcee

    from threeML.bayesian.emcee_sampler import EmceeSampler

    _available_samplers["emcee"] = EmceeSampler

except(ImportError):

    log.debug("no emcee")


try:

    import dynesty

    from threeML.bayesian.dynesty_sampler import (DynestyDynamicSampler,
                                                  DynestyNestedSampler)
    _available_samplers["dynesty_nested"] = DynestyNestedSampler
    _available_samplers["dynesty_dynamic"] = DynestyDynamicSampler


except(ImportError):

    log.debug("no dynesty")

try:

    import pymultinest

    from threeML.bayesian.multinest_sampler import MultiNestSampler

    _available_samplers["multinest"] = MultiNestSampler

except(ImportError):

    log.debug("no multinest")

try:

    import zeus

    from threeML.bayesian.zeus_sampler import ZeusSampler

    _available_samplers["zeus"] = ZeusSampler

except(ImportError):

    log.debug("no zeus")


try:

    import ultranest

    from threeML.bayesian.ultranest_sampler import UltraNestSampler

    _available_samplers["ultranest"] = UltraNestSampler


except(ImportError):

    log.debug("no ultranest")


try:

    import autoemcee

    from threeML.bayesian.autoemcee_sampler import AutoEmceeSampler

    _available_samplers["autoemcee"] = AutoEmceeSampler

except(ImportError):

    log.debug("no autoemcee")

# we should always have at least emcee available
if len(_available_samplers) == 0:

    log.error("There are NO samplers available!")
    log.error("emcee is installed by default, something is wrong!")

    raise RuntimeError()

class BayesianAnalysis(object):
    def __init__(self, likelihood_model: Model, data_list: DataList, **kwargs):
        """
        Bayesian analysis.

        :param likelihood_model: the likelihood model
        :param data_list: the list of datasets to use (normally an instance of DataList)
        :param kwargs: use 'verbose=True' for verbose operation
        :return:
        """

        self._analysis_type = "bayesian"

        self._is_registered = False

        self._register_model_and_data(likelihood_model, data_list)

        # # Make sure that the current model is used in all data sets
        #
        # for dataset in self.data_list.values():
        #     dataset.set_model(self._likelihood_model)

        # Init the samples to None

        self._samples = None
        self._raw_samples = None
        self._sampler = None
        self._log_like_values = None
        self._results = None

        self._sampler = None

    def _register_model_and_data(self, likelihood_model: Model, data_list: DataList):
        """

        make sure the model and data list are set up

        :param likelihood_model:
        :param data_list:
        :returns:
        :rtype:

        """

        log.debug("REGISTER MODEL")

        # Verify that all the free parameters have priors
        for parameter_name, parameter in likelihood_model.free_parameters.items():

            if not parameter.has_prior():
                log.error(
                    "You need to define priors for all free parameters before instancing a "
                    "Bayesian analysis"
                    f"{parameter_name} does NOT have a prior!"
                )

                raise RuntimeError()

        # Process optional keyword parameters

        self._likelihood_model = likelihood_model

        self._data_list = data_list

        for dataset in list(self._data_list.values()):

            dataset.set_model(self._likelihood_model)

            # Now get the nuisance parameters from the data and add them to the model
            # NOTE: it is important that this is *after* the setting of the model, as some
            # plugins might need to adjust the number of nuisance parameters depending on the
            # likelihood model

            for parameter_name, parameter in list(dataset.nuisance_parameters.items()):
                # Enforce that the nuisance parameter contains the instance name, because otherwise multiple instance
                # of the same plugin will overwrite each other's nuisance parameters

                assert dataset.name in parameter_name, (
                    "This is a bug of the plugin for %s: nuisance parameters "
                    "must contain the instance name" % type(dataset)
                )

                self._likelihood_model.add_external_parameter(parameter)

        log.debug("MODEL REGISTERED!")

        self._is_registered = True

    def set_sampler(self, sampler_name: str = "default", **kwargs):
        """
        Set the sampler
        :param sampler_name: (str) Name of sampler

        :param share_spectrum: (optional) Option to share the spectrum calc
        between detectors with the same input energy bins
        """

        using_default = False

        if sampler_name == "default":

            sampler_name = threeML_config.bayesian.default_sampler.value

            log.info(f"using default sampler {sampler_name}")

            using_default = True

        if sampler_name not in _available_samplers:

            log.error(
                f"{sampler_name} is not a valid sampler please choose from {','.join(list(_available_samplers.keys()))}")

            raise RuntimeError()

        self._sampler = _available_samplers[sampler_name](
            self._likelihood_model, self._data_list, **kwargs
        )

        if not using_default:

            log.info(f"sampler set to {sampler_name}")

        else:

            # now we will setup the samnpler with the
            # paramters from thre config

            default_params = threeML_config.bayesian.default_setup

            self.sampler.setup(**default_params)

            log.info("sampler is setup with default parameters")

    @property
    def sampler(self):

        return self._sampler

    def sample(self, quiet=False):

        with use_astromodels_memoization(False):

            self._sampler.sample(quiet=quiet)

    @property
    def results(self):

        return self._sampler.results

    @property
    def analysis_type(self):
        return self._analysis_type

    @property
    def log_like_values(self):
        """
        Returns the value of the log_likelihood found by the bayesian sampler while sampling from the posterior. If
        you need to find the values of the parameters which generated a given value of the log. likelihood, remember
        that the samples accessible through the property .raw_samples are ordered in the same way as the vector
        returned by this method.

        :return: a vector of log. like values
        """
        return self._sampler.log_like_values

    @property
    def log_probability_values(self):
        """
        Returns the value of the log_probability (posterior) found by the bayesian sampler while sampling from the posterior. If
        you need to find the values of the parameters which generated a given value of the log. likelihood, remember
        that the samples accessible through the property .raw_samples are ordered in the same way as the vector
        returned by this method.

        :return: a vector of log probabilty values
        """

        return self._sampler.log_probability_values

    @property
    def log_marginal_likelihood(self):
        """
                Return the log marginal likelihood (evidence
        ) if computed
                :return:
        """

        return self._sampler.marginal_likelihood

    @property
    def raw_samples(self):
        """
        Access the samples from the posterior distribution generated by the selected sampler in raw form (i.e.,
        in the format returned by the sampler)

        :return: the samples as returned by the sampler
        """

        return self._sampler.raw_samples

    @property
    def samples(self):
        """
        Access the samples from the posterior distribution generated by the selected sampler

        :return: a dictionary with the samples from the posterior distribution for each parameter
        """
        return self._sampler.samples

    @property
    def sampler(self):
        """
        Access the instance of the sampler used to sample the posterior distribution
        :return: an instance of the sampler
        """

        return self._sampler

    def plot_chains(self, thin=None):
        """
        Produce a plot of the series of samples for each parameter

        :parameter thin: use only one sample every 'thin' samples
        :return: a matplotlib.figure instance
        """

        return self.results.plot_chains(thin)

    @property
    def likelihood_model(self):
        """
        :return: likelihood model (a Model instance)
        """
        return self._likelihood_model

    @property
    def data_list(self):
        """
        :return: data list for this analysis
        """

        return self._data_list

    def convergence_plots(self, n_samples_in_each_subset, n_subsets):
        """
        Compute the mean and variance for subsets of the samples, and plot them. They should all be around the same
        values if the MCMC has converged to the posterior distribution.

        The subsamples are taken with two different strategies: the first is to slide a fixed-size window, the second
        is to take random samples from the chain (bootstrap)

        :param n_samples_in_each_subset: number of samples in each subset
        :param n_subsets: number of subsets to take for each strategy
        :return: a matplotlib.figure instance
        """

        return self.results.convergence_plots(n_samples_in_each_subset, n_subsets)

    def restore_median_fit(self):
        """
        Sets the model parameters to the mean of the marginal distributions
        """

        self._sampler.restore_median_fit()
