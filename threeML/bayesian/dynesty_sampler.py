import math
import os
import time

import numpy as np
from astromodels import ModelAssertionViolation, use_astromodels_memoization

from threeML.bayesian.sampler_base import UnitCubeSampler
from threeML.config.config import threeML_config
from threeML.io.logging import setup_logger
from threeML.parallel.parallel_client import ParallelClient

try:

    from dynesty import DynamicNestedSampler, NestedSampler

except:

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
        n_live_points=400,
        maxiter=None,
        maxcall=None,
        dlogz=None,
        logl_max=np.inf,
        n_effective=None,
        add_live=True,
        print_func=None,
        save_bounds=True,
        bound="multi",
        sample="auto",
        periodic=None,
        reflective=None,
        update_interval=None,
        first_update=None,
        npdim=None,
        rstate=None,
        use_pool=None,
        live_points=None,
        logl_args=None,
        logl_kwargs=None,
        ptform_args=None,
        ptform_kwargs=None,
        gradient=None,
        grad_args=None,
        grad_kwargs=None,
        compute_jac=False,
        enlarge=None,
        bootstrap=0,
        walks=25,
        facc=0.5,
        slices=5,
        fmove=0.9,
        max_move=100,
        update_func=None,
        **kwargs
    ):
        """TODO describe function

        :param n_live_points: 
        :type n_live_points: 
        :param maxiter: 
        :type maxiter: 
        :param maxcall: 
        :type maxcall: 
        :param dlogz: 
        :type dlogz: 
        :param logl_max: 
        :type logl_max: 
        :param n_effective: 
        :type n_effective: 
        :param add_live: 
        :type add_live: 
        :param print_func: 
        :type print_func: 
        :param save_bounds: 
        :type save_bounds: 
        :param bound: 
        :type bound: 
        :param sample:
        :type sample: 
        :param periodic: 
        :type periodic: 
        :param reflective: 
        :type reflective: 
        :param update_interval: 
        :type update_interval: 
        :param first_update: 
        :type first_update: 
        :param npdim: 
        :type npdim: 
        :param rstate: 
        :type rstate: 
        :param use_pool: 
        :type use_pool: 
        :param live_points: 
        :type live_points: 
        :param logl_args: 
        :type logl_args: 
        :param logl_kwargs: 
        :type logl_kwargs: 
        :param ptform_args: 
        :type ptform_args: 
        :param ptform_kwargs: 
        :type ptform_kwargs: 
        :param gradient: 
        :type gradient: 
        :param grad_args: 
        :type grad_args: 
        :param grad_kwargs: 
        :type grad_kwargs: 
        :param compute_jac: 
        :type compute_jac: 
        :param enlarge: 
        :type enlarge: 
        :param bootstrap: 
        :type bootstrap: 
        :param vol_dec: 
        :type vol_dec: 
        :param vol_check: 
        :type vol_check: 
        :param walks: 
        :type walks: 
        :param facc: 
        :type facc: 
        :param slices: 
        :type slices: 
        :param fmove: 
        :type fmove: 
        :param max_move: 
        :type max_move: 
        :param update_func: 
        :type update_func: 
        :returns: 

        """
        log.debug("Setup dynesty sampler")

        self._sampler_kwargs = {}
        self._sampler_kwargs["maxiter"] = maxiter
        self._sampler_kwargs["maxcall"] = maxcall
        self._sampler_kwargs["dlogz"] = dlogz
        self._sampler_kwargs["logl_max"] = logl_max
        self._sampler_kwargs["n_effective"] = n_effective
        self._sampler_kwargs["add_live"] = add_live
        self._sampler_kwargs["print_func"] = print_func
        self._sampler_kwargs["save_bounds"] = save_bounds

        self._kwargs = {}
        self._kwargs["nlive"] = n_live_points
        self._kwargs["bound"] = bound


        self._kwargs["sample"] = sample
        self._kwargs["periodic"] = periodic
        self._kwargs["reflective"] = reflective
        self._kwargs["update_interval"] = update_interval
        self._kwargs["first_update"] = first_update
        self._kwargs["npdim"] = npdim
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

        self._kwargs["live_points"] = live_points
        self._kwargs["logl_args"] = logl_args
        self._kwargs["logl_kwargs"] = logl_kwargs
        self._kwargs["ptform_args"] = ptform_args
        self._kwargs["ptform_kwargs"] = ptform_kwargs
        self._kwargs["gradient"] = gradient
        self._kwargs["grad_args"] = grad_args
        self._kwargs["grad_kwargs"] = grad_kwargs
        self._kwargs["compute_jac"] = compute_jac
        self._kwargs["enlarge"] = enlarge
        self._kwargs["bootstrap"] = bootstrap

        self._kwargs["walks"] = walks
        self._kwargs["facc"] = facc
        self._kwargs["slices"] = slices
        self._kwargs["fmove"] = fmove
        self._kwargs["max_move"] = max_move
        self._kwargs["update_func"] = update_func

        for k, v in kwargs.items():

            self._kwargs[k] = v

        self._is_setup = True

    def sample(self, quiet=False):
        """
        sample using the UltraNest numerical integration method
        :rtype: 

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
        idx = np.zeros(nsamples, dtype=np.int)

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
        maxiter_init=None,
        maxcall_init=None,
        dlogz_init=0.01,
        logl_max_init=np.inf,
        n_effective_init=np.inf,
        nlive_batch=500,
        wt_function=None,
        wt_kwargs=None,
        maxiter_batch=None,
        maxcall_batch=None,
        maxiter=None,
        maxcall=None,
        maxbatch=None,
        n_effective=np.inf,
        stop_function=None,
        stop_kwargs=None,
        use_stop=True,
        save_bounds=True,
        print_func=None,
        live_points=None,
        bound="multi",
        sample="auto",
        periodic=None,
        reflective=None,
        update_interval=None,
        first_update=None,
        npdim=None,
        rstate=None,
        use_pool=None,
        logl_args=None,
        logl_kwargs=None,
        ptform_args=None,
        ptform_kwargs=None,
        gradient=None,
        grad_args=None,
        grad_kwargs=None,
        compute_jac=False,
        enlarge=None,
        bootstrap=0,
        walks=25,
        facc=0.5,
        slices=5,
        fmove=0.9,
        max_move=100,
        update_func=None,
        **kwargs
    ):
        """TODO describe function

        :param nlive_init: 
        :type nlive_init: 
        :param maxiter_init: 
        :type maxiter_init: 
        :param maxcall_init: 
        :type maxcall_init: 
        :param dlogz_init: 
        :type dlogz_init: 
        :param logl_max_init: 
        :type logl_max_init: 
        :param n_effective_init: 
        :type n_effective_init: 
        :param nlive_batch: 
        :type nlive_batch: 
        :param wt_function: 
        :type wt_function: 
        :param wt_kwargs: 
        :type wt_kwargs: 
        :param maxiter_batch: 
        :type maxiter_batch: 
        :param maxcall_batch: 
        :type maxcall_batch: 
        :param maxiter: 
        :type maxiter: 
        :param maxcall: 
        :type maxcall: 
        :param maxbatch: 
        :type maxbatch: 
        :param n_effective: 
        :type n_effective: 
        :param stop_function: 
        :type stop_function: 
        :param stop_kwargs: 
        :type stop_kwargs: 
        :param use_stop: 
        :type use_stop: 
        :param save_bounds: 
        :type save_bounds: 
        :param print_func: 
        :type print_func: 
        :param live_points: 
        :type live_points: 
        :param bound: 
        :type bound: 
        :param sample:
        :type sample: 
        :param periodic: 
        :type periodic: 
        :param reflective: 
        :type reflective: 
        :param update_interval: 
        :type update_interval: 
        :param first_update: 
        :type first_update: 
        :param npdim: 
        :type npdim: 
        :param rstate: 
        :type rstate: 
        :param use_pool: 
        :type use_pool: 
        :param logl_args: 
        :type logl_args: 
        :param logl_kwargs: 
        :type logl_kwargs: 
        :param ptform_args: 
        :type ptform_args: 
        :param ptform_kwargs: 
        :type ptform_kwargs: 
        :param gradient: 
        :type gradient: 
        :param grad_args: 
        :type grad_args: 
        :param grad_kwargs: 
        :type grad_kwargs: 
        :param compute_jac: 
        :type compute_jac: 
        :param enlarge: 
        :type enlarge: 
        :param bootstrap: 
        :type bootstrap: 
        :param vol_dec: 
        :type vol_dec: 
        :param vol_check: 
        :type vol_check: 
        :param walks: 
        :type walks: 
        :param facc: 
        :type facc: 
        :param slices: 
        :type slices: 
        :param fmove: 
        :type fmove: 
        :param max_move: 
        :type max_move: 
        :param update_func: 
        :type update_func: 
        :returns: 

        """
        log.debug("Setup dynesty dynamic sampler")
        self._sampler_kwargs = {}
        self._sampler_kwargs["nlive_init"] = nlive_init
        self._sampler_kwargs["maxiter_init"] = maxiter_init
        self._sampler_kwargs["maxcall_init"] = maxcall_init
        self._sampler_kwargs["dlogz_init"] = dlogz_init
        self._sampler_kwargs["logl_max_init"] = logl_max_init
        self._sampler_kwargs["n_effective_init"] = n_effective_init
        self._sampler_kwargs["nlive_batch"] = nlive_batch
        self._sampler_kwargs["wt_function"] = wt_function
        self._sampler_kwargs["wt_kwargs"] = wt_kwargs
        self._sampler_kwargs["maxiter_batch"] = maxiter_batch
        self._sampler_kwargs["maxcall_batch"] = maxcall_batch
        self._sampler_kwargs["maxiter"] = maxiter
        self._sampler_kwargs["maxcall"] = maxcall
        self._sampler_kwargs["maxbatch"] = maxbatch
        self._sampler_kwargs["n_effective"] = n_effective
        self._sampler_kwargs["stop_function"] = stop_function
        self._sampler_kwargs["stop_kwargs"] = stop_kwargs
        self._sampler_kwargs["use_stop"] = use_stop
        self._sampler_kwargs["save_bounds"] = save_bounds
        self._sampler_kwargs["print_func"] = print_func
        self._sampler_kwargs["live_points"] = live_points

        self._kwargs = {}

        self._kwargs["bound"] = bound

        self._kwargs["sample"] = sample
        self._kwargs["periodic"] = periodic
        self._kwargs["reflective"] = reflective
        self._kwargs["update_interval"] = update_interval
        self._kwargs["first_update"] = first_update
        self._kwargs["npdim"] = npdim
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
        self._kwargs["gradient"] = gradient
        self._kwargs["grad_args"] = grad_args
        self._kwargs["grad_kwargs"] = grad_kwargs
        self._kwargs["compute_jac"] = compute_jac
        self._kwargs["enlarge"] = enlarge
        self._kwargs["bootstrap"] = bootstrap

        self._kwargs["walks"] = walks
        self._kwargs["facc"] = facc
        self._kwargs["slices"] = slices
        self._kwargs["fmove"] = fmove
        self._kwargs["max_move"] = max_move
        self._kwargs["update_func"] = update_func

        for k, v in kwargs.items():

            self._kwargs[k] = v

        self._is_setup = True

    def sample(self, quiet=False):
        """
        sample using the UltraNest numerical integration method
        :rtype: 

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
        idx = np.zeros(nsamples, dtype=np.int)
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
