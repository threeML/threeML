import emcee
import numpy as np

from threeML.io.logging import setup_logger
from threeML.bayesian.sampler_base import MCMCSampler
from threeML.config.config import threeML_config
from threeML.parallel.parallel_client import ParallelClient
from astromodels import ModelAssertionViolation, use_astromodels_memoization

log = setup_logger(__name__)

class EmceeSampler(MCMCSampler):
    def __init__(self, likelihood_model=None, data_list=None, **kwargs):
        """
        Sample using the emcee sampler. For details:
        https://emcee.readthedocs.io/en/stable/

        :param likelihood_model: 
        :param data_list: 
        :returns: 
        :rtype: 

        """

        super(EmceeSampler, self).__init__(likelihood_model, data_list, **kwargs)

    def setup(self, n_iterations, n_burn_in=None, n_walkers=20, seed=None):

        log.debug(f"Setup for Emcee sampler: n_iterations:{n_iterations}, n_burn_in:{n_burn_in},"\
                  f"n_walkers: {n_walkers}, seed: {seed}.")

        self._n_iterations = int(n_iterations)

        if n_burn_in is None:

            self._n_burn_in = int(np.floor(n_iterations / 4.0))

        else:

            self._n_burn_in = n_burn_in

        self._n_walkers = int(n_walkers)

        self._seed = seed

        self._is_setup = True

    def sample(self, quiet=False):

        if not self._is_setup:

            log.info("You forgot to setup the sampler!")
            return

        loud = not quiet

        self._update_free_parameters()

        n_dim = len(list(self._free_parameters.keys()))

        # Get starting point

        p0 = emcee.State(self._get_starting_points(self._n_walkers))

        # Deactivate memoization in astromodels, which is useless in this case since we will never use twice the
        # same set of parameters
        with use_astromodels_memoization(False):

            if threeML_config["parallel"]["use-parallel"]:

                c = ParallelClient()
                view = c[:]

                sampler = emcee.EnsembleSampler(
                    self._n_walkers, n_dim, self.get_posterior, pool=view
                )

            else:

                sampler = emcee.EnsembleSampler(
                    self._n_walkers, n_dim, self.get_posterior
                )

            # If a seed is provided, set the random number seed
            if self._seed is not None:

                sampler._random.seed(self._seed)

            log.debug("Start emcee run")
            # Sample the burn-in
            pos, prob, state = sampler.run_mcmc(
                initial_state=p0, nsteps=self._n_burn_in, progress=loud
            )
            log.debug("Emcee run done")

            # Reset sampler

            sampler.reset()

            state = emcee.State(pos, prob, random_state=state)

            # Run the true sampling

            _ = sampler.run_mcmc(
                initial_state=state, nsteps=self._n_iterations, progress=loud
            )

        acc = np.mean(sampler.acceptance_fraction)

        log.info("\nMean acceptance fraction: %s\n" % acc)

        self._sampler = sampler
        self._raw_samples = sampler.get_chain(flat=True)

        # Compute the corresponding values of the likelihood

        # First we need the prior
        log_prior = [self._log_prior(x) for x in self._raw_samples]

        # Now we get the log posterior and we remove the log prior

        self._log_like_values = sampler.get_log_prob(flat=True) - log_prior

        # we also want to store the log probability

        self._log_probability_values = sampler.get_log_prob(flat=True)

        self._marginal_likelihood = None

        self._build_samples_dictionary()

        self._build_results()

        # Display results
        if loud:
            self._results.display()

        return self.samples
