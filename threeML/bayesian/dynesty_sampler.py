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
        bound="multi",
        sample="auto",
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
        """
        Setup the Dynesty nested sampler.

        :param n_live_points: Number of live points used in the nested sampling run.
            Defaults to ``500``.
        :type n_live_points: int
        :param bound: Method for bounding the live points. Options include
            ``"none"``, ``"single"``, ``"multi"``, or ``"balls"``.
            Defaults to ``"multi"``.
        :type bound: str
        :param sample: Sampling method used for proposing new points. Options include
            ``"auto"``, ``"uniform"``, ``"rwalk"``, ``"slice"``, or ``"rslice"``.
            Defaults to ``"auto"``.
        :type sample: str
        :param periodic: Indices of parameters with periodic boundary conditions.
        :type periodic: list[int] or None
        :param reflective: Indices of parameters with reflective boundary conditions.
        :type reflective: list[int] or None
        :param update_interval: Interval (in iterations)
        :type update_interval: int or float or None
        :param first_update: Initial update behavior
        :type first_update: tuple or dict or None
        :param rstate: Random number generator for reproducibility.
        :type rstate: numpy.random.Generator or numpy.random.RandomState or None
        :param queue_size: Number of threads or processes for parallel sampling.
        :type queue_size: int or None
        :param pool: Multiprocessing pool for parallel likelihood evaluations.
        :type pool: multiprocessing.Pool or None
        :param use_pool: use the pool
        :type use_pool: dict or None
        :param live_points: Initial live points for starting the nested sampling run.
        :type live_points: array-like or None
        :param logl_args: Positional arguments for the log-likelihood function.
        :type logl_args: tuple or None
        :param logl_kwargs: Keyword arguments for the log-likelihood function.
        :type logl_kwargs: dict or None
        :param ptform_args: Positional arguments for the prior transform function.
        :type ptform_args: tuple or None
        :param ptform_kwargs: Keyword arguments for the prior transform function.
        :type ptform_kwargs: dict or None
        :param enlarge: Factor by which to enlarge the bounding ellipsoids.
        :type enlarge: float or None
        :param bootstrap: Number of bootstrap resamples for bounding ellipsoids.
        :type bootstrap: int or None
        :param walks: Number of random-walk steps used in certain sampling methods.
        :type walks: int or None
        :param facc: Target acceptance fraction for slice sampling.
            Defaults to ``0.5``.
        :type facc: float
        :param slices: Number of slices used in the slice sampling method.
        :type slices: int or None
        :param ncdim: Dimensionality of coordinate transforms (if applicable).
        :type ncdim: int or None
        :param blob: Whether to store additional data for each likelihood evaluation.
            Defaults to ``False``.
        :type blob: bool
        :param save_evaluation_history: Save likelihood evaluation history to file.
            Defaults to ``False``.
        :type save_evaluation_history: bool
        :param history_filename: Path to the file where evaluation history is saved.
        :type history_filename: str or None
        :param kwargs: Additional keyword arguments.
        :type kwargs: dict
        """
        log.debug("Setup dynesty sampler")

        self._sampler_kwargs = {}

        self._kwargs = locals().copy()
        for k in ["self", "n_live_points"]:
            self._kwargs.pop(k)

        self._kwargs["nlive"] = n_live_points
        kwargs = self._kwargs.get("kwargs")
        self._kwargs.pop("kwargs")
        self._kwargs.update(kwargs)

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
        bound="multi",
        sample="auto",
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
        """
        Setup the Dynesty dynamic nested sampler.

        :param nlive_init: Number of live points used for the initial phase.
            Defaults to ``500``.
        :type nlive_init: int
        :param bound: Method for bounding the live points. Options include
            ``"none"``, ``"single"``, ``"multi"``, or ``"balls"``.
            Defaults to ``"multi"``.
        :type bound: str
        :param sample: Sampling method used for proposing new points. Options include
            ``"auto"``, ``"uniform"``, ``"rwalk"``, ``"slice"``, or ``"rslice"``.
            Defaults to ``"auto"``.
        :type sample: str
        :param periodic: Indices of parameters with periodic boundary conditions.
        :type periodic: list[int] or None
        :param reflective: Indices of parameters with reflective boundary conditions.
        :type reflective: list[int] or None
        :param update_interval: Interval (in iterations) for updating.
        :type update_interval: int or float or None
        :param first_update: Initial update behavior.
        :type first_update: tuple or dict or None
        :param rstate: Random number generator for reproducibility.
        :type rstate: numpy.random.Generator or numpy.random.RandomState or None
        :param queue_size: Number of threads or processes for parallel sampling.
        :type queue_size: int or None
        :param pool: Multiprocessing pool for parallel likelihood evaluations.
        :type pool: multiprocessing.Pool or None
        :param use_pool: Dict specifying which parts of the algorithm to parallelize.
            Defaults to disabling prior transform and likelihood parallelization.
        :type use_pool: dict or None
        :param logl_args: Positional arguments for the log-likelihood function.
        :type logl_args: tuple or None
        :param logl_kwargs: Keyword arguments for the log-likelihood function.
        :type logl_kwargs: dict or None
        :param ptform_args: Positional arguments for the prior transform function.
        :type ptform_args: tuple or None
        :param ptform_kwargs: Keyword arguments for the prior transform function.
        :type ptform_kwargs: dict or None
        :param enlarge: Factor by which to enlarge the bounding ellipsoids.
        :type enlarge: float or None
        :param bootstrap: Number of bootstrap resamples for bounding ellipsoids.
        :type bootstrap: int or None
        :param walks: Number of random-walk steps used in certain sampling methods.
        :type walks: int or None
        :param facc: Target acceptance fraction for slice sampling.
            Defaults to ``0.5``.
        :type facc: float
        :param slices: Number of slices used in the slice sampling method.
        :type slices: int or None
        :param ncdim: Dimensionality of coordinate transforms (if applicable).
        :type ncdim: int or None
        :param blob: Store additional data for each likelihood evaluation.
            Defaults to ``False``.
        :type blob: bool
        :param history_filename: Path to the file where evaluation history is saved.
        :type history_filename: str or None
        :param save_evaluation_history: Save likelihood evaluation history to file.
            Defaults to ``False``.
        :type save_evaluation_history: bool
        :param kwargs: Additional keyword arguments.
        :type kwargs: dict
        """

        log.debug("Setup dynesty dynamic sampler")
        self._sampler_kwargs = {}
        self._sampler_kwargs["nlive_init"] = nlive_init

        self._kwargs = locals().copy()
        for k in ["self", "nlive_init"]:
            self._kwargs.pop(k)
        kwargs = self._kwargs.get("kwargs")
        self._kwargs.pop("kwargs")
        self._kwargs.update(kwargs)

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
