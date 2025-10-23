import math
from typing import Optional, Literal
from packaging.version import Version

import numpy as np
from astromodels import use_astromodels_memoization

from threeML.bayesian.sampler_base import UnitCubeSampler
from threeML.config.config import threeML_config
from threeML.io.logging import setup_logger
from threeML.parallel.parallel_client import ParallelClient

try:
    from dynesty import DynamicNestedSampler, NestedSampler
    import dynesty

    DYNESTY_DOC_URL = (
        f"https://dynesty.readthedocs.io/en/v{dynesty.__version__}/api.html"
    )
except Exception:
    has_dynesty = False

else:
    has_dynesty = True

log = setup_logger(__name__)


def fill_docs(**kwargs):
    def decorator(func):
        if func.__doc__:
            func.__doc__ = func.__doc__.format(**kwargs)
        return func

    return decorator


class DynestyPool(object):
    """A simple wrapper for `dview`."""

    def __init__(self, dview):
        self.dview = dview
        self.size = len(dview)

    def map(self, function, tasks):
        return self.dview.map_sync(function, tasks)


class DynestyNestedSampler(UnitCubeSampler):
    def __init__(self, likelihood_model=None, data_list=None, **kwargs):
        assert has_dynesty, "You must install Dynesty to use this sampler"

        super(DynestyNestedSampler, self).__init__(
            likelihood_model, data_list, **kwargs
        )

    @fill_docs(BASE_URL=DYNESTY_DOC_URL)
    def setup(
        self,
        nlive: int = 500,
        bound: Optional[Literal["multi", "single", "none", "balls", "cubes"]] = "multi",
        history_filename: Optional[str] = None,
        **kwargs,
    ):
        """
        Setup the Dynesty nested sampler.
        All available parameters can be found in the respective version of
        {BASE_URL}#dynesty.dynesty.NestedSampler

        :param nlive: Number of live points. Defaults to 500.
        :type nlive: int
        :param bound: Method to approximately bound the prior using the current set of
            live points. Options are "multi", "single", "none", "balls" or "cubes".
            Defaults to "multi".
        :param history_filename: Path to save the history. Defaults to None
        :type history_filename: str
        :param kwargs: Additional keyword arguments - must be same name and type as
            paramters in constructor of the dynesty.NestedSampler class.
            Defaults to the values used by dynesty.
        :type kwargs: dict
        """

        log.debug("Setup dynesty sampler")
        if history_filename is not None:
            if Version(dynesty.__version__) < Version("1.2.0"):
                log.warning(
                    f"Your dynesty version is {dynesty.__version__} but "
                    + "saving to a file was introduced in version 1.2.0. We will "
                    + "ignore your input."
                )
                history_filename = None

        self._sampler_kwargs = {}

        self._kwargs = {}
        self._kwargs["nlive"] = nlive
        self._kwargs["bound"] = bound
        self._kwargs["history_filename"] = history_filename

        self._kwargs.update(kwargs)

        self._is_setup = True

    def sample(self, quiet: bool = False, **kwargs):
        """Sample using the Dynesty NestedSampler class

        :param quiet: verbosity. Defaults to False.
        :type quiet: bool
        :param kwargs: Additional keywords that get passed to the run_nested() function.
        :type kwargs: dict

        :rtype:
        :returns:
        """
        if not self._is_setup:
            log.info("You forgot to setup the sampler!")
            return

        loud = not quiet

        self._sampler_kwargs.update(kwargs)
        self._update_free_parameters()

        param_names = list(self._free_parameters.keys())

        ndim = len(param_names)

        self._kwargs["ndim"] = ndim

        loglike, dynesty_prior = self._construct_unitcube_posterior(return_copy=True)

        # check if we are doing to do things in parallel

        if threeML_config["parallel"]["use_parallel"]:
            c = ParallelClient()
            view = c[:]

            self._kwargs["pool"] = view
            self._kwargs["queue_size"] = len(view)

        sampler = NestedSampler(loglike, dynesty_prior, **self._kwargs)

        self._sampler_kwargs["print_progress"] = loud

        with use_astromodels_memoization(False):
            log.debug("Start dynesty run")

            sampler.run_nested(**self._sampler_kwargs)

            log.debug("Dynesty run done")

        self._sampler = sampler

        results = self._sampler.results

        # draw posterior samples
        weights = np.exp(results["logwt"] - results["logz"][-1])

        SQRTEPS = math.sqrt(float(np.finfo(np.float64).eps))

        rstate = np.random

        if abs(np.sum(weights) - 1.0) > SQRTEPS:  # same tol as in np.random.choice.
            raise ValueError("Weights do not sum to 1.")

        # Make N subdivisions and choose positions with a consistent random offset.
        nsamples = len(weights)
        positions = (rstate.random() + np.arange(nsamples)) / nsamples

        # Resample the data.
        idx = np.zeros(nsamples, dtype=int)

        cumulative_sum = np.cumsum(weights)
        i, j = 0, 0

        while i < nsamples:
            if positions[i] < cumulative_sum[j]:
                idx[i] = j
                i += 1
            else:
                j += 1

        samples_dynesty = results["samples"][idx]

        self._raw_samples = samples_dynesty

        # now do the same for the log likes

        logl_dynesty = results["logl"][idx]

        self._log_like_values = logl_dynesty

        self._log_probability_values = self._log_like_values + np.array(
            [self._log_prior(samples) for samples in self._raw_samples]
        )

        self._marginal_likelihood = self._sampler.results["logz"][-1] / np.log(10.0)

        self._build_samples_dictionary()

        self._build_results()

        # Display results
        if loud:
            self._results.display()

        # now get the marginal likelihood

        return self.samples


class DynestyDynamicSampler(UnitCubeSampler):
    def __init__(self, likelihood_model=None, data_list=None, **kwargs):
        assert has_dynesty, "You must install Dynesty to use this sampler"

        super(DynestyDynamicSampler, self).__init__(
            likelihood_model, data_list, **kwargs
        )

    @fill_docs(BASE_URL=DYNESTY_DOC_URL)
    def setup(
        self,
        nlive: int = 500,
        history_filename=None,
        **kwargs,
    ):
        """
        Setup the Dynesty dynamic nested sampler.
        All available parameters can be found in the respective version of
        {BASE_URL}#dynesty.dynesty.DynamicNestedSampler

        :param nlive: Number of live points used during the inital nested sampling run
        :type nlive: int
        :param history_filename: Path to save the history. Defaults to None
        :type history_filename: str
        :param kwargs: Additional keyword arguments - must be same name and type as
            paramters in constructor of the dynesty.DynamicNestedSampler class.
            Defaults to the values used by dynesty.
        :type kwargs: dict
        """

        log.debug("Setup dynesty dynamic sampler")
        if history_filename is not None:
            if Version(dynesty.__version__) < Version("1.2.0"):
                log.warning(
                    f"Your dynesty version is {dynesty.__version__} but "
                    + "saving to a file was introduced in version 1.2.0"
                )
                history_filename = None
        self._kwargs = {}
        self._sampler_kwargs = {}

        self._kwargs["nlive"] = nlive

        self._kwargs.update(kwargs)

        self._is_setup = True

    def sample(self, quiet: bool = False, **kwargs):
        """Sample using the Dynestey DynamicNestedSampler class.

        :param quiet: verbosity. Defaults to False.
        :type quiet: bool
        :param kwargs: Additional keywords that get passed to the run_nested() function.
        :type kwargs: dict

        :rtype:
        :returns:
        """
        if not self._is_setup:
            log.info("You forgot to setup the sampler!")
            return

        loud = not quiet
        self._sampler_kwargs.update(kwargs)

        self._update_free_parameters()

        param_names = list(self._free_parameters.keys())

        ndim = len(param_names)

        self._kwargs["ndim"] = ndim

        loglike, dynesty_prior = self._construct_unitcube_posterior(return_copy=True)

        # check if we are doing to do things in parallel

        if threeML_config["parallel"]["use_parallel"]:
            c = ParallelClient()
            view = c[:]

            self._kwargs["pool"] = view
            self._kwargs["queue_size"] = len(view)

        sampler = DynamicNestedSampler(loglike, dynesty_prior, **self._kwargs)

        self._sampler_kwargs["print_progress"] = loud

        with use_astromodels_memoization(False):
            log.debug("Start dynestsy run")
            sampler.run_nested(**self._sampler_kwargs)
            log.debug("Dynesty run done")

        self._sampler = sampler

        results = self._sampler.results

        # draw posterior samples
        weights = np.exp(results["logwt"] - results["logz"][-1])

        SQRTEPS = math.sqrt(float(np.finfo(np.float64).eps))

        rstate = np.random

        if abs(np.sum(weights) - 1.0) > SQRTEPS:  # same tol as in np.random.choice.
            raise ValueError("Weights do not sum to 1.")

        # Make N subdivisions and choose positions with a consistent random offset.
        nsamples = len(weights)
        positions = (rstate.random() + np.arange(nsamples)) / nsamples

        # Resample the data.
        idx = np.zeros(nsamples, dtype=int)
        cumulative_sum = np.cumsum(weights)
        i, j = 0, 0
        while i < nsamples:
            if positions[i] < cumulative_sum[j]:
                idx[i] = j
                i += 1
            else:
                j += 1

        samples_dynesty = results["samples"][idx]

        self._raw_samples = samples_dynesty

        # now do the same for the log likes

        logl_dynesty = results["logl"][idx]

        self._log_like_values = logl_dynesty

        self._log_probability_values = self._log_like_values + np.array(
            [self._log_prior(samples) for samples in self._raw_samples]
        )

        self._marginal_likelihood = self._sampler.results["logz"][-1] / np.log(10.0)

        self._build_samples_dictionary()

        self._build_results()

        # Display results
        if loud:
            self._results.display()

        # now get the marginal likelihood

        return self.samples
