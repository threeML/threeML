import numpy as np

from threeML.io.logging import setup_logger
from threeML.bayesian.sampler_base import MCMCSampler
from threeML.config.config import threeML_config

from threeML.parallel.parallel_client import ParallelClient
from astromodels import use_astromodels_memoization


try:

    import zeus

except:

    has_zeus = False

else:

    has_zeus = True


try:

    # see if we have mpi and/or are using parallel

    from mpi4py import MPI

    if MPI.COMM_WORLD.Get_size() > 1:  # need parallel capabilities
        using_mpi = True

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        from mpi4py.futures import MPIPoolExecutor

    else:

        using_mpi = False
except:

    using_mpi = False

log = setup_logger(__name__)

class ZeusSampler(MCMCSampler):
    def __init__(self, likelihood_model=None, data_list=None, **kwargs):

        assert has_zeus, "You must install zeus-mcmc to use this sampler"

        super(ZeusSampler, self).__init__(likelihood_model, data_list, **kwargs)

    def setup(self, n_iterations, n_burn_in=None, n_walkers=20, seed=None):

        log.debug(f"Setup for Zeus sampler: n_iterations:{n_iterations}, n_burn_in:{n_burn_in},"\
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

        p0 = self._get_starting_points(self._n_walkers)

        # Deactivate memoization in astromodels, which is useless in this case since we will never use twice the
        # same set of parameters
        with use_astromodels_memoization(False):

            if using_mpi:

                with MPIPoolExecutor() as executor:

                    sampler = zeus.sampler(
                        logprob_fn=self.get_posterior,
                        nwalkers=self._n_walkers,
                        ndim=n_dim,
                        pool=executor,
                    )

                    # if self._seed is not None:

                    #     sampler._random.seed(self._seed)

                    # Run the true sampling
                    log.debug("Start zeus run")
                    _ = sampler.run(
                        p0, self._n_iterations + self._n_burn_in, progress=loud,
                    )
                    log.debug("Zeus run done")

            elif threeML_config["parallel"]["use-parallel"]:

                c = ParallelClient()
                view = c[:]

                sampler = zeus.sampler(
                    logprob_fn=self.get_posterior,
                    nwalkers=self._n_walkers,
                    ndim=n_dim,
                    pool=view,
                )

            else:

                sampler = zeus.sampler(
                    logprob_fn=self.get_posterior, nwalkers=self._n_walkers, ndim=n_dim
                )

            # If a seed is provided, set the random number seed
            # if self._seed is not None:

            #     sampler._random.seed(self._seed)

            # Sample the burn-in
            if not using_mpi:
                log.debug("Start zeus run")
                _ = sampler.run(p0, self._n_iterations + self._n_burn_in, progress=loud)
                log.debug("Zeus run done")

        self._sampler = sampler
        self._raw_samples = sampler.flatten(discard=self._n_burn_in)

        # Compute the corresponding values of the likelihood

        # First we need the prior
        log_prior = np.array([self._log_prior(x) for x in self._raw_samples])
        self._log_probability_values = sampler.get_log_prob(flat=True, discard=self._n_burn_in)



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
