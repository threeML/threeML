import logging

import inspect

import numpy as np
from astromodels import use_astromodels_memoization

from threeML.bayesian.sampler_base import UnitCubeSampler
from threeML.config.config import threeML_config


try:
    import nautilus

except ModuleNotFoundError:
    has_nautilus: bool = False

else:
    has_nautilus: bool = True


try:
    # see if we have mpi and/or are using parallel
    from mpi4py import MPI

    if MPI.COMM_WORLD.Get_size() > 1:  # need parallel capabilities
        using_mpi: bool = True

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

    else:
        using_mpi: bool = False
except ModuleNotFoundError:
    using_mpi: bool = False

log = logging.getLogger(__name__)


class NautilusSampler(UnitCubeSampler):
    def __init__(self, likelihood_model=None, data_list=None, **kwargs):
        if not has_nautilus:
            log.error("You must install nautilus to use this sampler")

            raise AssertionError("You must install nautilus to use this sampler")

        super(NautilusSampler, self).__init__(likelihood_model, data_list, **kwargs)

    def setup(
        self,
        n_live: int = 3000,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        """Setup the nautilus sampler.
        See: https://nautilus-sampler.readthedocs.io/en/stable/index.html
        and https://doi.org/10.1093/mnras/stad2441

        :param n_live: Number of livepoints, default is 3000
        :type n_live: int
        :param verbose: Verbosity, default is false
        :type n_live: bool
        :param kwargs: Additional keyword arguments for the Nautilus Sampler.
            Please refer to the official documentation. Defaults to the Nautilus
            defaults.
        :type kwargs: dict
        """

        # get the allowed keys from the function definitions
        allowed_setup_keys = inspect.getfullargspec(nautilus.Sampler.__init__).args[5:]
        allowed_sampler_keys = inspect.getfullargspec(nautilus.Sampler.run).args[1:-1]

        sampler_dict = {}
        run_dict = {}

        for k, v in kwargs.items():
            if k in allowed_setup_keys:
                sampler_dict[k] = v
            elif k in allowed_sampler_keys:
                run_dict[k] = v
            else:
                log.warning(
                    f"You provided {k} with a value of {v} wich is not available"
                    + " for neither the Nautilus initalization nor the run function. "
                    + "We will skip it."
                )

        self._sampler_dict = {"n_live": n_live}
        self._sampler_dict.update(sampler_dict)

        self._run_dict = {}
        self._run_dict.update(run_dict)
        self._is_setup: bool = True

    def sample(self, quiet=False, **kwargs) -> np.array:
        """Sample using the Nautilus sampler

        :param quiet: Flag for anti-verbosity, defaults to false
        :type quiet: bool
        :param kwargs:
        :type kwargs: dict

        :rtype: np.array
        :returns: samples
        """
        if not self._is_setup:
            log.error("You forgot to setup the sampler!")

            return

        loud = not quiet

        self._update_free_parameters()

        # chain_name = self._kwargs.pop("log_dir")

        loglike, nautilus_prior = self._construct_unitcube_posterior(return_copy=True)

        sampler = nautilus.Sampler(
            nautilus_prior,
            loglike,
            n_dim=len(self._free_parameters),
            **self._sampler_dict,
        )

        if threeML_config["parallel"]["use_parallel"]:
            raise RuntimeError(
                "If you want to run ultranest in parallel you need to use an ad-hoc "
                "method"
            )

        else:
            with use_astromodels_memoization(False):
                log.debug("Start nautilus run")

                sampler.run(**self._run_dict)

                log.debug("nautilus run done")

        process_fit = False

        if using_mpi:
            # if we are running in parallel and this is not the
            # first engine, then we want to wait and let everything finish

            comm.Barrier()

            if rank != 0:
                # these engines do not need to read
                process_fit = False

            else:
                process_fit = True

        else:
            process_fit = True

        if process_fit:
            points, log_w, log_likelihood = sampler.posterior(equal_weight=True)

            self._sampler = sampler

            self._log_like_values = log_likelihood

            self._raw_samples = points

            # now get the log probability

            self._log_probability_values = self._log_like_values + np.array(
                [self._log_prior(samples) for samples in self._raw_samples]
            )

            self._build_samples_dictionary()

            self._marginal_likelihood = sampler.log_z

            self._build_results()

            # Display results
            if loud:
                self._results.display()

            # now get the marginal likelihood

            return self.samples
