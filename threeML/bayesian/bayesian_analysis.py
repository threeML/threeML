from typing import Dict, Optional

import numpy as np
from astromodels import use_astromodels_memoization
from astromodels.core.model import Model
from threeML.analysis_results import BayesianResults
from threeML.bayesian.sampler_base import SamplerBase
from threeML.config import threeML_config
from threeML.data_list import DataList
from threeML.io.logging import setup_logger

log = setup_logger(__name__)


_possible_samplers = {
    "emcee": [
        "emcee",
        "from threeML.bayesian.emcee_sampler import EmceeSampler",
        "EmceeSampler",
    ],
    "multinest": [
        "pymultinest",
        "from threeML.bayesian.multinest_sampler import MultiNestSampler",
        "MultiNestSampler",
    ],
    "zeus": [
        "zeus",
        "from threeML.bayesian.zeus_sampler import ZeusSampler",
        "ZeusSampler",
    ],
    "ultranest": [
        "ultranest",
        "from threeML.bayesian.ultranest_sampler import UltraNestSampler",
        "UltraNestSampler",
    ],
    "dynesty_nested": [
        "dynesty",
        "from threeML.bayesian.dynesty_sampler import DynestyNestedSampler",
        "DynestyNestedSampler",
    ],
    "dynesty_dynamic": [
        "dynesty",
        "from threeML.bayesian.dynesty_sampler import DynestyDynamicSampler",
        "DynestyDynamicSampler",
    ],
    "autoemcee": [
        "autoemcee",
        "from threeML.bayesian.autoemcee_sampler import AutoEmceeSampler",
        "AutoEmceeSampler",
    ],
}


_available_samplers = {}

for k, v in _possible_samplers.items():

    try:

        exec(f"import {v[0]}")
        exec(f"{v[1]}")
        exec(f"_available_samplers['{k}'] = {v[2]}")

    except (ImportError):

        log.debug(f"no {v[0]}")


# we should always have at least emcee available
if len(_available_samplers) == 0:

    log.error("There are NO samplers available!")
    log.error("emcee is installed by default, something is wrong!")

    raise RuntimeError("There are NO samplers available!")


class BayesianAnalysis:
    def __init__(self, likelihood_model: Model, data_list: DataList, **kwargs):
        """
        Perform Bayesian analysis by passing your model and data.
        All free parameters must have priors set.

        :param likelihood_model: the likelihood model
        :param data_list: the list of datasets to use (normally an instance of DataList)
        :param kwargs: use 'verbose=True' for verbose operation
        :return:

        :example:

        """

        self._analysis_type: str = "bayesian"

        self._is_registered: bool = False

        self._register_model_and_data(likelihood_model, data_list)

        self._sampler: Optional[SamplerBase] = None

    def _register_model_and_data(
        self, likelihood_model: Model, data_list: DataList
    ) -> None:
        """

        make sure the model and data list are set up

        :param likelihood_model:
        :param data_list:
        :returns:
        :rtype:

        """

        log.debug("REGISTER MODEL")

        # Verify that all the free parameters have priors
        for (
            parameter_name,
            parameter,
        ) in likelihood_model.free_parameters.items():

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

            for parameter_name, parameter in list(
                dataset.nuisance_parameters.items()
            ):
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

            log.info(f"using default sampler [blue]{sampler_name}[/blue]")

            using_default = True

        if sampler_name not in _available_samplers:

            log.error(
                f"{sampler_name} is not a valid/available sampler please choose from [blue]{','.join(list(_available_samplers.keys()))}[/blue]"
            )

            raise RuntimeError()

        self._sampler = _available_samplers[sampler_name](
            self._likelihood_model, self._data_list, **kwargs
        )

        if not using_default:

            log.info(f"sampler set to [blue]{sampler_name}[/blue]")

        else:

            # now we will setup the samnpler with the
            # paramters from thre config

            default_params = threeML_config.bayesian[f"{sampler_name}_setup"]

            self.sampler.setup(**default_params)

            log.info("sampler is setup with default parameters")

    def sample(self, quiet=False) -> None:
        """
        sample the posterior of the model with the selected algorithm

        If no algorithm as been set, then the configured default algorithm
        we default parameters will be run

        :param quiet: if True, then no output is displayed
        :type quiet:
        :returns:

        """
        if self._sampler is None:

            # assuming the default sampler
            self.set_sampler()

        with use_astromodels_memoization(False):

            self._sampler.sample(quiet=quiet)

    @property
    def results(self) -> Optional[BayesianResults]:

        return self._sampler.results

    @property
    def analysis_type(self) -> str:
        return self._analysis_type

    @property
    def log_like_values(self) -> Optional[np.ndarray]:
        """
        Returns the value of the log_likelihood found by the bayesian sampler while sampling from the posterior. If
        you need to find the values of the parameters which generated a given value of the log. likelihood, remember
        that the samples accessible through the property .raw_samples are ordered in the same way as the vector
        returned by this method.

        :return: a vector of log. like values
        """
        return self._sampler.log_like_values

    @property
    def log_probability_values(self) -> Optional[np.ndarray]:
        """
        Returns the value of the log_probability (posterior) found by the bayesian sampler while sampling from the posterior. If
        you need to find the values of the parameters which generated a given value of the log. likelihood, remember
        that the samples accessible through the property .raw_samples are ordered in the same way as the vector
        returned by this method.

        :return: a vector of log probabilty values
        """

        return self._sampler.log_probability_values

    @property
    def log_marginal_likelihood(self) -> Optional[float]:
        """
                Return the log marginal likelihood (evidence
        ) if computed
                :return:
        """

        return self._sampler.marginal_likelihood

    @property
    def raw_samples(self) -> Optional[np.ndarray]:
        """
        Access the samples from the posterior distribution generated by the selected sampler in raw form (i.e.,
        in the format returned by the sampler)

        :return: the samples as returned by the sampler
        """

        return self._sampler.raw_samples

    @property
    def samples(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Access the samples from the posterior distribution generated by the selected sampler

        :return: a dictionary with the samples from the posterior distribution for each parameter
        """
        return self._sampler.samples

    @property
    def sampler(self) -> Optional[SamplerBase]:
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
    def likelihood_model(self) -> Model:
        """
        :return: likelihood model (a Model instance)
        """
        return self._likelihood_model

    @property
    def data_list(self) -> DataList:
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

        return self.results.convergence_plots(
            n_samples_in_each_subset, n_subsets
        )

    def restore_median_fit(self):
        """
        Sets the model parameters to the mean of the marginal distributions
        """

        self._sampler.restore_median_fit()

    def restore_MAP_fit(self) -> None:
        """
        Sets the model parameters to the MAP of the probability
        """

        self._sampler.restore_MAP_fit()
