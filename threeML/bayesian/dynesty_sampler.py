import math

import numpy as np
from astromodels import use_astromodels_memoization

from threeML.bayesian.sampler_base import UnitCubeSampler
from threeML.config.config import threeML_config
from threeML.io.logging import setup_logger
from threeML.parallel.parallel_client import ParallelClient

try:
    from dynesty import DynamicNestedSampler, NestedSampler

except Exception:
    has_dynesty = False

else:
    has_dynesty = True

log = setup_logger(__name__)


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

    def setup(
        self,
        n_live_points=500,
        bound='multi',
        sample='auto',
        periodic=None,
        reflective=None,
        update_interval=None,
        first_update=None,
        rstate=None,
        queue_size=None,
        pool=None,
        use_pool=None,
        live_points=None,
        logl_args=None,
        logl_kwargs=None,
        ptform_args=None,
        ptform_kwargs=None,
        enlarge=None,
        bootstrap=None,
        walks=None,
        facc=0.5,
        slices=None,
        ncdim=None,
        blob=False,
        save_evaluation_history=False,
        history_filename=None,
        **kwargs,
    ):
        log.debug("Setup dynesty sampler")

        self._sampler_kwargs = {}

        self._kwargs = {}
        self._kwargs["nlive"] = n_live_points
        self._kwargs["bound"] = bound

        self._kwargs["sample"] = sample
        self._kwargs["periodic"] = periodic
        self._kwargs["reflective"] = reflective
        self._kwargs["update_interval"] = update_interval
        self._kwargs["first_update"] = first_update
        self._kwargs["rstate"] = rstate
        self._kwargs["pool"] = pool

        # TODO: have to figure out why
        # this is not working properly
        if use_pool is None:
            use_pool = dict(
                prior_transform=False,
                loglikelihood=False,
                propose_point=False,
                update_bound=True,
            )

        self._kwargs["use_pool"] = use_pool

        self._kwargs["live_points"] = live_points
        self._kwargs["logl_args"] = logl_args
        self._kwargs["logl_kwargs"] = logl_kwargs
        self._kwargs["ptform_args"] = ptform_args
        self._kwargs["ptform_kwargs"] = ptform_kwargs
        self._kwargs["enlarge"] = enlarge
        self._kwargs["bootstrap"] = bootstrap

        self._kwargs["walks"] = walks
        self._kwargs["facc"] = facc
        self._kwargs["slices"] = slices

        for k, v in kwargs.items():
            self._kwargs[k] = v

        self._is_setup = True

    def sample(self, quiet=False):
        """Sample using the UltraNest numerical integration method :rtype:

        :returns:
        """
        if not self._is_setup:
            log.info("You forgot to setup the sampler!")
            return

        loud = not quiet

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

    def setup(
        self,
        nlive_init=500,
        bound='multi',
        sample='auto',
        periodic=None,
        reflective=None,
        update_interval=None,
        first_update=None,
        rstate=None,
        queue_size=None,
        pool=None,
        use_pool=None,
        logl_args=None,
        logl_kwargs=None,
        ptform_args=None,
        ptform_kwargs=None,
        enlarge=None,
        bootstrap=None,
        walks=None,
        facc=0.5,
        slices=None,
        ncdim=None,
        blob=False,
        history_filename=None,
        save_evaluation_history=False,
        **kwargs,
    ):
        log.debug("Setup dynesty dynamic sampler")
        self._sampler_kwargs = {}
        self._sampler_kwargs["nlive_init"] = nlive_init

        self._kwargs = {}

        self._kwargs["bound"] = bound

        self._kwargs["sample"] = sample
        self._kwargs["periodic"] = periodic
        self._kwargs["reflective"] = reflective
        self._kwargs["update_interval"] = update_interval
        self._kwargs["first_update"] = first_update
        self._kwargs["rstate"] = rstate
        self._kwargs["pool"] = None

        # TODO: have to figure out why
        # this is not working properly
        if use_pool is None:
            use_pool = dict(
                prior_transform=False,
                loglikelihood=False,
                propose_point=False,
                update_bound=True,
            )

        self._kwargs["use_pool"] = use_pool

        self._kwargs["logl_args"] = logl_args
        self._kwargs["logl_kwargs"] = logl_kwargs
        self._kwargs["ptform_args"] = ptform_args
        self._kwargs["ptform_kwargs"] = ptform_kwargs
        self._kwargs["enlarge"] = enlarge
        self._kwargs["bootstrap"] = bootstrap

        self._kwargs["walks"] = walks
        self._kwargs["facc"] = facc
        self._kwargs["slices"] = slices

        for k, v in kwargs.items():
            self._kwargs[k] = v

        self._is_setup = True

    def sample(self, quiet=False):
        """Sample using the UltraNest numerical integration method :rtype:

        :returns:
        """
        if not self._is_setup:
            log.info("You forgot to setup the sampler!")
            return

        loud = not quiet

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
