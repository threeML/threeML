import logging
from importlib.util import find_spec
import multiprocessing as mp

import numpy as np
from astromodels import use_astromodels_memoization

from threeML.bayesian.sampler_base import MCMCSampler
from threeML.config.config import threeML_config

from threeML.parallel.parallel_client import ParallelClient

log = logging.getLogger(__name__)
has_zeus = False
if find_spec("zeus") is not None:
    import zeus

    has_zeus = True
has_mpi = False
if find_spec("mpi4py") is not None:
    from mpi4py.MPI import COMM_WORLD

    comm = COMM_WORLD
    if comm.Get_size() > 1:
        has_mpi = True


class ZeusSampler(MCMCSampler):
    def __init__(self, likelihood_model=None, data_list=None, **kwargs):
        assert has_zeus, "You must install zeus-mcmc to use this sampler"

        super(ZeusSampler, self).__init__(likelihood_model, data_list, **kwargs)

    def setup(self, n_iterations, n_burn_in=None, n_walkers=None, **kwargs):
        """Set up the zeus sampler.

        :param n_iterations:
        :type n_iterations:
        :param n_burn_in:
        :type n_burn_in:
        :param n_walkers:
        :type n_walkers:
        :returns:
        """
        log.debug(
            f"Setup for Zeus sampler: n_iterations:{n_iterations}, n_burn_in:"
            f"{n_burn_in}, n_walkers: {n_walkers}."
        )

        self._n_iterations = int(n_iterations)

        if n_burn_in is None:
            self._n_burn_in = int(np.floor(n_iterations / 4.0))

        else:
            self._n_burn_in = n_burn_in
        if n_walkers is None:
            n_walkers = int(len(self._likelihood_model.free_parameters.values())) * 2
            log.warning(
                "You did not provide 'n_walkers' for the zeus setup - will set it to a"
                " default of 2x the number of free parameters"
            )

        self._n_walkers = int(n_walkers)
        self._sampler_kwargs = kwargs

        self._is_setup = True

    def sample(self, quiet=False, n_chains=1, p0=None, **kwargs):
        if not self._is_setup:
            log.info("You forgot to setup the sampler!")
            return

        loud = not quiet

        self._update_free_parameters()

        n_dim = len(list(self._free_parameters.keys()))

        # Get starting point
        if p0 is None:
            p0 = np.array(self._get_starting_points(self._n_walkers, variance=0.3))

        # Deactivate memoization in astromodels, which is useless in this case since we
        # will never use twice the same set of parameters
        using_mpi = False
        with use_astromodels_memoization(False):
            if has_mpi:
                with zeus.ChainManager(n_chains, use_dill=False) as cm:
                    sampler = zeus.EnsembleSampler(
                        logprob_fn=self.get_posterior,
                        nwalkers=self._n_walkers,
                        ndim=n_dim,
                        pool=cm.get_pool,
                        verbose=loud,
                    )

                    # Run the true sampling
                    log.debug("Start zeus run")
                    using_mpi = True
                    _ = sampler.run_mcmc(
                        p0,
                        self._n_iterations + self._n_burn_in,
                        progress=loud,
                    )
                    log.debug("Zeus run done")
            elif threeML_config["parallel"]["use_parallel"]:
                c = ParallelClient()
                view = c[:]

                sampler = zeus.EnsembleSampler(
                    logprob_fn=self.get_posterior_proxy(),
                    nwalkers=self._n_walkers,
                    ndim=n_dim,
                    pool=view,
                    verbose=loud,
                )
            elif self._sampler_kwargs.get("pool", None) is not None:
                with mp.pool.Pool(int(self._sampler_kwargs.get("pool"))) as executor:
                    sampler = zeus.EnsembleSampler(
                        logprob_fn=self.get_posterior,
                        nwalkers=self._n_walkers,
                        ndim=n_dim,
                        pool=executor,
                        verbose=loud,
                    )
                    sampler.run_mcmc(
                        p0,
                        self._n_iterations + self._n_burn_in,
                        progress=loud,
                    )
                    using_mpi = True

            else:
                sampler = zeus.EnsembleSampler(
                    logprob_fn=self.get_posterior,
                    nwalkers=self._n_walkers,
                    ndim=n_dim,
                    verbose=loud,
                    **kwargs,
                )

            # Sample the burn-in
            if not using_mpi:
                log.debug("Start zeus run")
                _ = sampler.run_mcmc(
                    p0, self._n_iterations + self._n_burn_in, progress=loud
                )
                log.debug("Zeus run done")

        self._sampler = sampler
        self._raw_samples = sampler.get_chain(flat=True, discard=self._n_burn_in)

        # Compute the corresponding values of the likelihood

        # First we need the prior
        log_prior = np.array([self._log_prior(x) for x in self._raw_samples])
        self._log_probability_values = sampler.get_log_prob(
            flat=True, discard=self._n_burn_in
        )

        # np.array(
        #     [self.get_posterior(x) for x in self._raw_samples]
        # )

        # Now we get the log posterior and we remove the log prior

        self._log_like_values = self._log_probability_values - log_prior

        # we also want to store the log probability

        self._marginal_likelihood = None

        self._build_samples_dictionary()

        self._build_results()

        # Display results
        if loud:
            print(self._sampler.summary)
            self._results.display()

        return self.samples

    @property
    def n_walkers(self):
        return self._n_walkers
