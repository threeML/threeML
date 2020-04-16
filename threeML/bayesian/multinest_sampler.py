import os
import time

import numpy as np
import pymultinest

from threeML.bayesian.sampler_base import UnitCubeSampler
from threeML.config.config import threeML_config


try:

    import pymultinest

except:

    has_pymultinest = False

else:

    has_pymultinest = True


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


class MultiNestSampler(UnitCubeSampler):
    def __init__(self, likelihood_model=None, data_list=None, **kwargs):
        """
        Implements the MultiNest sampler of https://github.com/farhanferoz/MultiNest
        via the python wrapper of https://github.com/JohannesBuchner/PyMultiNest

        :param likelihood_model: 
        :param data_list: 
        :returns: 
        :rtype: 

        """

        assert has_pymultinest, "You must install MultiNest to use this sampler"

        super(MultiNestSampler, self).__init__(likelihood_model, data_list, **kwargs)

    def setup(
        self,
        n_live_points,
        chain_name="chains/fit-",
        resume=False,
        importance_nested_sampling=False,
        **kwargs
    ):
        """
        Setup the MultiNest Sampler. For details see:
        

        :param n_live_points: number of live points for the evaluation
        :param chain_name: the chain name
        :param importance_nested_sampling: use INS 
        :returns: 
        :rtype: 

        """

        self._kwargs = {}
        self._kwargs["n_live_points"] = n_live_points
        self._kwargs["outputfiles_basename"] = chain_name
        self._kwargs["importance_nested_sampling"] = importance_nested_sampling
        self._kwargs["chain_name"] = chain_name
        self._kwargs["resume"] = resume

        for k, v in kwargs.items():

            self._kwargs[k] = v

        self._is_setup = True

    def sample(self, quiet=False):
        """
        sample using the MultiNest numerical integration method

        :returns: 
        :rtype: 

        """
        if not self._is_setup:

            print("You forgot to setup the sampler!")
            return

        assert (
            has_pymultinest
        ), "You don't have pymultinest installed, so you cannot run the Multinest sampler"

        loud = not quiet

        self._update_free_parameters()

        n_dim = len(list(self._free_parameters.keys()))

        # MULTINEST uses a different call signiture for
        # sampling so we construct callbakcs
        loglike, multinest_prior = self._construct_unitcube_posterior()

        # We need to check if the MCMC
        # chains will have a place on
        # the disk to write and if not,
        # create one

        chain_name = self._kwargs.pop("chain_name")

        mcmc_chains_out_dir = ""
        tmp = chain_name.split("/")
        for s in tmp[:-1]:
            mcmc_chains_out_dir += s + "/"

        if using_mpi:

            # if we are running in parallel and this is not the
            # first engine, then we want to wait and let everything finish

            if rank != 0:

                # let these guys take a break
                time.sleep(1)

            else:

                # create mcmc chains directory only on first engine

                if not os.path.exists(mcmc_chains_out_dir):
                    os.makedirs(mcmc_chains_out_dir)

        else:

            if not os.path.exists(mcmc_chains_out_dir):
                os.makedirs(mcmc_chains_out_dir)

        # Multinest must be run parallel via an external method
        # see the demo in the examples folder!!

        if threeML_config["parallel"]["use-parallel"]:

            raise RuntimeError(
                "If you want to run multinest in parallell you need to use an ad-hoc method"
            )

        else:

            sampler = pymultinest.run(
                loglike, multinest_prior, n_dim, n_dim, **self._kwargs
            )

        # Use PyMULTINEST analyzer to gather parameter info

        process_fit = False

        if using_mpi:

            # if we are running in parallel and this is not the
            # first engine, then we want to wait and let everything finish

            if rank != 0:

                # let these guys take a break
                time.sleep(5)

                # these engines do not need to read
                process_fit = False

            else:

                # wait for a moment to allow it all to turn off
                time.sleep(5)

                process_fit = True

        else:

            process_fit = True

        if process_fit:

            multinest_analyzer = pymultinest.analyse.Analyzer(
                n_params=n_dim, outputfiles_basename=chain_name
            )

            # Get the log. likelihood values from the chain
            self._log_like_values = multinest_analyzer.get_equal_weighted_posterior()[
                :, -1
            ]

            self._sampler = sampler

            self._raw_samples = multinest_analyzer.get_equal_weighted_posterior()[
                :, :-1
            ]

            # now get the log probability

            self._log_probability_values = self._log_like_values + np.array(
                [self._log_prior(samples) for samples in self._raw_samples]
            )

            self._build_samples_dictionary()

            self._marginal_likelihood = multinest_analyzer.get_stats()[
                "global evidence"
            ] / np.log(10.0)

            self._build_results()

            # Display results
            if loud:
                self._results.display()

            # now get the marginal likelihood

            return self.samples
