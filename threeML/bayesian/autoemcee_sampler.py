import logging
import os
import time
from pathlib import Path

import numpy as np
from astromodels import ModelAssertionViolation, use_astromodels_memoization

from threeML.bayesian.sampler_base import UnitCubeSampler
from threeML.config.config import threeML_config
from threeML.io.logging import setup_logger

try:

    import autoemcee

except:

    has_autoemcee = False

else:

    has_autoemcee = True


try:

    # see if we have mpi and/or are using parallel

    from mpi4py import MPI

    if MPI.COMM_WORLD.Get_size() > 1:  # need parallel capabilities
        using_mpi = True

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

    else:

        using_mpi = False
except:

    using_mpi = False


# un_logger = logging.getLogger("ultranest")
# un_logger.propagate = False

log = setup_logger(__name__)


class AutoEmceeSampler(UnitCubeSampler):
    def __init__(self, likelihood_model=None, data_list=None, **kwargs):

        assert has_autoemcee, "You must install AutoEmcee to use this sampler"

        super(AutoEmceeSampler, self).__init__(
            likelihood_model, data_list, **kwargs)

    def setup(
        self,
            num_global_samples=10000,
            num_chains=4,
            num_walkers=None,
            max_ncalls=1000000,
            max_improvement_loops=4,
            num_initial_steps=100,
            min_autocorr_times=0
    ):
        """
        Sample until MCMC chains have converged.

        The steps are:

        1. Draw *num_global_samples* from prior. The highest *num_walkers* points are selected.
        2. Set *num_steps* to *num_initial_steps*
        3. Run *num_chains* MCMC ensembles for *num_steps* steps
        4. For each walker chain, compute auto-correlation length (Convergence requires *num_steps*/autocorrelation length > *min_autocorr_times*)
        5. For each parameter, compute geweke convergence diagnostic (Convergence requires \|z\| < 2)
        6. For each ensemble, compute gelman-rubin rank convergence diagnostic (Convergence requires rhat<1.2)
        7. If converged, stop and return results.
        8. Increase *num_steps* by 10, and repeat from (3) up to *max_improvement_loops* times.




        num_global_samples: int
            Number of samples to draw from the prior to
        num_chains: int
            Number of independent ensembles to run. If running with MPI,
            this is set to the number of MPI processes.
        num_walkers: int
            Ensemble size. If None, max(100, 4 * dim) is used
        max_ncalls: int
            Maximum number of likelihood function evaluations
        num_initial_steps: int
            Number of sampler steps to take in first iteration
        max_improvement_loops: int
            Number of times MCMC should be re-attempted (see above)
        min_autocorr_times: float
            if positive, additionally require for convergence that the
            number of samples is larger than the *min_autocorr_times*
            times the autocorrelation length.

        """

        # log.debug(f"Setup for UltraNest sampler: min_num_live_points:{min_num_live_points}, "\
        #           f"chain_name:{chain_name}, dlogz: {dlogz}, wrapped_params: {wrapped_params}. "\
        #           f"Other input: {kwargs}")

        self._num_global_samples = num_global_samples
        self._num_chains = num_chains
        self._num_walkers = num_walkers
        self._max_ncalls = max_ncalls
        self._max_improvement_loops = max_improvement_loops
        self._num_initial_steps = num_initial_steps
        self._min_autocorr_times = min_autocorr_times

        self._is_setup = True

    def sample(self, quiet=False):
        """
        sample using the UltraNest numerical integration method
        :rtype: 

        :returns: 

        """
        if not self._is_setup:

            log.error("You forgot to setup the sampler!")

            raise RuntimeError()

        loud = not quiet

        self._update_free_parameters()

        param_names = list(self._free_parameters.keys())

        n_dim = len(param_names)

        loglike, autoemcee_prior = self._construct_unitcube_posterior(
            return_copy=True)

        # We need to check if the MCMC
        # chains will have a place on
        # the disk to write and if not,
        # create one

        if threeML_config["parallel"]["use_parallel"]:

            log.error(
                "If you want to run ultranest in parallell you need to use an ad-hoc method")

            raise RuntimeError()

        else:

            sampler = autoemcee.ReactiveAffineInvariantSampler(
                param_names,
                loglike,
                transform=autoemcee_prior,
                vectorized=False,
                sampler="goodman-weare"
            )

            with use_astromodels_memoization(False):
                log.debug("Start autoemcee run")
                sampler.run(self._num_global_samples,
                            self._num_chains,
                            self._num_walkers,
                            self._max_ncalls,
                            self._max_improvement_loops,
                            self._num_initial_steps,
                            self._min_autocorr_times,
                            progress=threeML_config.interface.progress_bars


                            )
                log.debug("autoemcee run done")

        process_fit = False

        if using_mpi:

            # if we are running in parallel and this is not the
            # first engine, then we want to wait and let everything finish

            if rank != 0:

                # let these guys take a break
                time.sleep(1)

                # these engines do not need to read
                process_fit = False

            else:

                # wait for a moment to allow it all to turn off
                time.sleep(1)

                process_fit = True

        else:

            process_fit = True

        if process_fit:

            results = sampler.results

            self._sampler = sampler

            self._raw_samples = np.concatenate(
                [sampler.transform(s.get_chain(flat=True)) for s in self._sampler.samplers])

            # First we need the prior
            log_prior = [self._log_prior(x) for x in self._raw_samples]

            self._log_probability_values = np.concatenate(
                [s.get_log_prob(flat=True) for s in self._sampler.samplers])

            self._log_like_values = self._log_probability_values - log_prior

            self._marginal_likelihood = None
            
            self._build_samples_dictionary()

            self._build_results()

            # Display results
            if loud:
                self._results.display()

            # now get the marginal likelihood

            return self.samples
