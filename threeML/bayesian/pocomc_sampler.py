import logging
from importlib.util import find_spec

from astromodels import use_astromodels_memoization

from threeML.bayesian.sampler_base import MCMCSampler

log = logging.getLogger(__name__)
has_pocomc = False
if find_spec("pocomc") is not None:
    import pocomc as pc

    has_pocomc = True
has_mpi = False
if find_spec("mpi4py") is not None:
    from mpi4py.MPI import COMM_WORLD

    comm = COMM_WORLD
    if comm.Get_size() > 1:
        has_mpi = True


class PocoMCSampler(MCMCSampler):
    def __init__(self, likelihood_model=None, data_list=None, **kwargs):
        assert has_pocomc, "You must install pocomc to use this sampler"

        super(PocoMCSampler, self).__init__(likelihood_model, data_list, **kwargs)

    def get_posterior(self, trial_values) -> float:
        for i, (parameter_name, parameter) in enumerate(self._free_parameters.items()):
            parameter.value = trial_values[i]

        return self._log_like(trial_values)

    def setup(self, **kwargs):
        """Set up the pocomc sampler.

        :param n_iterations:
        :type n_iterations:
        :param n_burn_in:
        :type n_burn_in:
        :param n_walkers:
        :type n_walkers:
        :returns:
        """
        self._sampler_kwargs = kwargs
        self._is_setup = True

    def sample(self, **kwargs):
        kwargs.pop("quiet")
        if not self._is_setup:
            log.info("You forgot to setup the sampler!")
            return

        self._update_free_parameters()
        with use_astromodels_memoization(False):
            prior = pc.Prior(
                [
                    p.prior.scipy_dist
                    for p in list(self._likelihood_model.free_parameters.values())
                ]
            )

            sampler = pc.Sampler(
                prior=prior,
                likelihood=self.get_posterior_proxy(),
                **self._sampler_kwargs,
            )

            sampler.run(**kwargs)
        self._sampler = sampler

        samples, weights, logl, logp = sampler.posterior()

        self._raw_samples = samples
        self._log_like_values = logl
        self._log_probability_values = self._log_like_values + logp

        self._marginal_likelihood = None

        self._build_samples_dictionary()

        self._build_results()

        # Display results
        self._results.display()

        return self.samples

    @property
    def n_walkers(self):
        return self._n_walkers
