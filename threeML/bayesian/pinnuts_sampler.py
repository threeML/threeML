import numpy as np
import numdifftools

from threeML.bayesian.sampler_base import MCMCSampler
from threeML.config.config import threeML_config
from threeML.parallel.parallel_client import ParallelClient
from astromodels import ModelAssertionViolation, use_astromodels_memoization


try:

    import nuts
    from nuts.helpers import numerical_grad
    
except:

    has_nuts = False

else:

    has_nuts = True


class NUTSSampler(MCMCSampler):
    def __init__(self, likelihood_model=None, data_list=None, **kwargs):
        """
        Sample using the emcee sampler. For details:
        https://emcee.readthedocs.io/en/stable/

        :param likelihood_model: 
        :param data_list: 
        :returns: 
        :rtype: 

        """

        super(NUTSSampler, self).__init__(likelihood_model, data_list, **kwargs)

    def setup(self, n_iterations, n_adapt=None, delta=0.6, seed=None):

        self._n_iterations = int(n_iterations)

        self._delta = delta

        if n_adapt is None:

            self._n_adapt = int(np.floor(n_iterations / 2.0))

        else:

            self._n_adapt = int(n_adapt)

        self._seed = seed

        self._is_setup = True

    def sample(self, quiet=False):

        assert self._is_setup, "You forgot to setup the sampler!"

        loud = not quiet

        self._update_free_parameters()

        n_dim = len(list(self._free_parameters.keys()))

        # Get starting point

        p0 = self._get_starting_points(1)[0]
        print(p0)
        
        # Deactivate memoization in astromodels, which is useless in this case since we will never use twice the
        # same set of parameters
        with use_astromodels_memoization(False):

            if threeML_config["parallel"]["use-parallel"]:

                c = ParallelClient()
                view = c[:]
                pool = view

            else:

                pool = None

                def logp(theta):

                    return self.get_posterior(theta)

                
                def grad(theta):

                    return numerical_grad(theta, self.get_posterior)
                
                nuts_fn = nuts.NutsSampler_fn_wrapper(self.get_posterior, grad)

                samples, lnprob, epsilon = nuts.nuts6(nuts_fn, self._n_iterations, self._n_adapt, p0)
                
#            sampler = nuts.NUTSSampler(n_dim, self.get_posterior, gradfn=None)  

            # # if a seed is provided, set the random number seed
            # if self._seed is not None:

            #     sampler._random.seed(self._seed)

            # # Run the true sampling

            # samples = sampler.run_mcmc(
            #     initial_state=p0,
            #     M=self._n_iterations,
            #     Madapt=self._n_adapt,
            #     delta=self._delta,
            #     progress=loud,
            # )


        self._sampler = None
        self._raw_samples = samples

        # Compute the corresponding values of the likelihood

        self._test = lnprob
        
        # First we need the prior
        log_prior = np.array([self._log_prior(x) for x in self._raw_samples])

        # Now we get the log posterior and we remove the log prior

        self._log_like_values = np.array([self._log_like(x) for x in self._raw_samples])

        # we also want to store the log probability

        self._log_probability_values = log_prior + self._log_like_values

        self._marginal_likelihood = None

        self._build_samples_dictionary()

        self._build_results()

        # Display results
        if loud:
            self._results.display()

        return self.samples
