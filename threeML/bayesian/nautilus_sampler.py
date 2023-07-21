import logging
import os
from typing import Any, Dict, Optional

import numpy as np
from astromodels import ModelAssertionViolation, use_astromodels_memoization
from threeML.bayesian.sampler_base import UnitCubeSampler
from threeML.config.config import threeML_config
from threeML.io.logging import setup_logger


import inspect


def capture_arguments(func, *args, **kwargs):
    # Get the function's signature
    signature = inspect.signature(func)

    # Bind the provided arguments to the function's parameters
    bound_args = signature.bind(*args, **kwargs)

    # Convert the bound arguments to a dictionary
    arg_dict = bound_args.arguments

    return arg_dict


try:

    import nautilus

except:

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
except:

    using_mpi: bool = False

log = setup_logger(__name__)


class NautilusSampler(UnitCubeSampler):
    def __init__(self, likelihood_model=None, data_list=None, **kwargs):

        if not has_nautilus:

            log.error("You must install nautilus to use this sampler")

            raise AssertionError(
                "You must install nautilus to use this sampler"
            )

        super(NautilusSampler, self).__init__(
            likelihood_model, data_list, **kwargs
        )

    def setup(
        self,
        n_live: int = 2000,
        n_update: Optional[int] = None,
        enlarge_per_dim: float = 1.1,
        n_points_min: Optional[int] = None,
        split_threshold: int = 100,
        n_networks: int = 4,
        neural_network_kwargs: Dict[Any] = dict(),
        prior_args: List[Any] = [],
        prior_kwargs: Dict[Any] = dict(),
        likelihood_args: List[Any] = [],
        likelihood_kwargs: Dict[Any] = dict(),
        n_batch: int = 100,
        n_like_new_bound: Optional[int] = None,
        vectorized: bool = False,
        pass_dict: Optional[bool] = None,
        pool: Optional[int] = None,
        seed: Optional[int] = None,
        filepath: Optional[str] = None,
        resume: bool = True,
        f_live: float = 0.01,
        n_shell: Optional[int] = None,
        n_eff: int = 10000,
        discard_exploration: bool = False,
        verbose: bool = False,
    ):

        """TODO describe function

        :param n_live:
        :type n_live: int
        :param n_update:
        :type n_update: Optional[int]
        :param enlarge_per_dim:
        :type enlarge_per_dim: float
        :param n_points_min:
        :type n_points_min: Optional[int]
        :param split_threshold:
        :type split_threshold: int
        :param n_networks:
        :type n_networks: int
        :param neural_network_kwargs:
        :type neural_network_kwargs: Dict[Any]
        :param prior_args:
        :type prior_args: List[Any]
        :param prior_kwargs:
        :type prior_kwargs: Dict[Any]
        :param likelihood_args:
        :type likelihood_args: List[Any]
        :param likelihood_kwargs:
        :type likelihood_kwargs: Dict[Any]
        :param n_batch:
        :type n_batch: int
        :param n_like_new_bound:
        :type n_like_new_bound: Optional[int]
        :param vectorized:
        :type vectorized: bool
        :param pass_dict:
        :type pass_dict: Optional[bool]
        :param pool:
        :type pool: Optional[int]
        :param seed:
        :type seed: Optional[int]
        :param filepath:
        :type filepath: Optional[str]
        :param resume:
        :type resume: bool
        :param f_live:
        :type f_live: float
        :param n_shell:
        :type n_shell: Optional[int]
        :param n_eff:
        :type n_eff: int
        :param discard_exploration:
        :type discard_exploration: bool
        :param verbose:
        :type verbose: bool
        :returns:

        """

        arg_dict = locals()

        # Remove the "self" key from the dictionary (if present) as it's not an argument
        arg_dict.pop("self", None)

        self._sampler_dict = dict(list(d.items())[:-5])
        self._run_dict = dict(list(d.items())[-5:])
        print(self._run_dict)

        self._is_setup: bool = True

    def sample(self, quiet=False):
        """
        sample using the UltraNest numerical integration method
        :rtype:

        :returns:

        """
        if not self._is_setup:

            log.error("You forgot to setup the sampler!")

            return

        loud = not quiet

        self._update_free_parameters()

        param_names = list(self._free_parameters.keys())

        chain_name = self._kwargs.pop("log_dir")

        loglike, nautilus_prior = self._construct_unitcube_posterior(
            return_copy=True
        )

        # Multinest must be run parallel via an external method

        # see the demo in the examples folder!!

        sampler = nautilus.Sampler(
            nautilus_prior, loglike, **self._sampler_dict
        )

        if threeML_config["parallel"]["use_parallel"]:

            raise RuntimeError(
                "If you want to run ultranest in parallel you need to use an ad-hoc method"
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

            self._marginal_likelihood = sampler.evidence()

            self._build_results()

            # Display results
            if loud:

                self._results.display()

            # now get the marginal likelihood

            return self.samples
