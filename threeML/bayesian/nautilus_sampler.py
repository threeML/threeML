import logging
import os
from typing import Any, Dict, Optional, List

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
        neural_network_kwargs: Dict[str, Any] = dict(),
        prior_args: List[Any] = [],
        prior_kwargs: Dict[str, Any] = dict(),
        likelihood_args: List[Any] = [],
        likelihood_kwargs: Dict[str, Any] = dict(),
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
        """

        setup the nautilus sampler.

        See: https://nautilus-sampler.readthedocs.io/en/stable/index.html

        :param n_live: Number of so-called live points. New bounds are constructed so that they encompass the live points. Default is 3000.
        :type n_live: int
        :param n_update: The maximum number of additions to the live set before a new bound is created. If None, use n_live. Default is None.
        :type n_update: Optional[int]
        :param enlarge_per_dim: Along each dimension, outer ellipsoidal bounds are enlarged by this factor. Default is 1.1.
        :type enlarge_per_dim: float
        :param n_points_min: The minimum number of points each ellipsoid should have. Effectively, ellipsoids with less than twice that number will not be split further. If None, uses n_points_min = n_dim + 50. Default is None.
        :type n_points_min: Optional[int]
        :param split_threshold: hreshold used for splitting the multi-ellipsoidal bound used for sampling. If the volume of the bound prior enlarging is larger than split_threshold times the target volume, the multi-ellipsiodal bound is split further, if possible. Default is 100.
        :type split_threshold: int
        :param n_networks: Number of networks used in the estimator. Default is 4.
        :type n_networks: int
        :param neural_network_kwargs: Non-default keyword arguments passed to the constructor of MLPRegressor.
        :type neural_network_kwargs: Dict[Any]
        :param prior_args: List of extra positional arguments for prior. Only used if prior is a function.
        :type prior_args: List[Any]
        :param prior_kwargs: Dictionary of extra keyword arguments for prior. Only used if prior is a function.
        :type prior_kwargs: Dict[Any]
        :param likelihood_args: List of extra positional arguments for likelihood.
        :type likelihood_args: List[Any]
        :param likelihood_kwargs: Dictionary of extra keyword arguments for likelihood.
        :type likelihood_kwargs: Dict[Any]
        :param n_batch: Number of likelihood evaluations that are performed at each step. If likelihood evaluations are parallelized, should be multiple of the number of parallel processes. Very large numbers can lead to new bounds being created long after n_update additions to the live set have been achieved. This will not cause any bias but could reduce efficiency. Default is 100.
        :type n_batch: int
        :param n_like_new_bound: The maximum number of likelihood calls before a new bounds is created. If None, use 10 times n_live. Default is None.
        :type n_like_new_bound: Optional[int]
        :param vectorized: If True, the likelihood function can receive multiple input sets at once. For example, if the likelihood function receives arrays, it should be able to take an array with shape (n_points, n_dim) and return an array with shape (n_points). Similarly, if the likelihood function accepts dictionaries, it should be able to process dictionaries where each value is an array with shape (n_points). Default is False.
        :type vectorized: bool
        :param pass_dict: If True, the likelihood function expects model parameters as dictionaries. If False, it expects regular numpy arrays. Default is to set it to True if prior was a nautilus.Prior instance and False otherwise
        :type pass_dict: Optional[bool]
        :param pool: Pool used for parallelization of likelihood calls and sampler calculations. If None, no parallelization is performed. If an integer, the sampler will use a multiprocessing.Pool object with the specified number of processes. Finally, if specifying a tuple, the first one specifies the pool used for likelihood calls and the second one the pool for sampler calculations. Default is None.
        :type pool: Optional[int]
        :param seed: Seed for random number generation used for reproducible results accross different runs. If None, results are not reproducible. Default is None.
        :type seed: Optional[int]
        :param filepath: ath to the file where results are saved. Must have a ‘.h5’ or ‘.hdf5’ extension. If None, no results are written. Default is None.
        :type filepath: Optional[str]
        :param resume: If True, resume from previous run if filepath exists. If False, start from scratch and overwrite any previous file. Default is True.
        :type resume: bool
        :param f_live: Maximum fraction of the evidence contained in the live set before building the initial shells terminates. Default is 0.01.
        :type f_live: float
        :param n_shell: Minimum number of points in each shell. The algorithm will sample from the shells until this is reached. Default is the batch size of the sampler which is 100 unless otherwise specified.
        :type n_shell: Optional[int]
        :param n_eff: Minimum effective sample size. The algorithm will sample from the shells until this is reached. Default is 10000.

        :type n_eff: int
        :param discard_exploration: Whether to discard points drawn in the exploration phase. This is required for a fully unbiased posterior and evidence estimate. Default is False.
        :type discard_exploration: bool
        :param verbose: If True, print additional information. Default is False.
        :type verbose: bool
        :returns:

        """

        arg_dict = locals()

        # Remove the "self" key from the dictionary (if present) as it's not an argument
        arg_dict.pop("self", None)

        self._sampler_dict = dict(list(arg_dict.items())[:-5])
        self._run_dict = dict(list(arg_dict.items())[-5:])

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

        # chain_name = self._kwargs.pop("log_dir")

        loglike, nautilus_prior = self._construct_unitcube_posterior(
            return_copy=True
        )

        sampler = nautilus.Sampler(
            nautilus_prior,
            loglike,
            n_dim=len(self._free_parameters),
            **self._sampler_dict
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
