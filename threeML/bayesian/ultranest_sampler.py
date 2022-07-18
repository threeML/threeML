import logging
import os
from typing import Optional

import numpy as np
from astromodels import ModelAssertionViolation, use_astromodels_memoization
from threeML.bayesian.sampler_base import UnitCubeSampler
from threeML.config.config import threeML_config
from threeML.io.logging import setup_logger

try:

    import ultranest

except:

    has_ultranest = False

else:

    has_ultranest = True


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


un_logger = logging.getLogger("ultranest")
un_logger.propagate = False

log = setup_logger(__name__)


class UltraNestSampler(UnitCubeSampler):
    def __init__(self, likelihood_model=None, data_list=None, **kwargs):

        assert has_ultranest, "You must install UltraNest to use this sampler"

        super(UltraNestSampler, self).__init__(likelihood_model, data_list, **kwargs)

    def setup(
        self,
        min_num_live_points: int = 400,
        dlogz: float = 0.5,
        chain_name: Optional[str] = None,
        resume: str = "overwrite",
        wrapped_params=None,
        stepsampler=None,
        use_mlfriends: bool = True,
        **kwargs,
    ):
        """
                set up the Ultranest sampler. Consult the documentation:

                https://johannesbuchner.github.io/UltraNest/ultranest.html?highlight=reactive#ultranest.integrator.ReactiveNestedSampler

                :param min_num_live_points: minimum number of live points throughout the run
                :type min_num_live_points: int
                :param dlogz: Target evidence uncertainty. This is the std between bootstrapped logz integrators.
                :type dlogz: float
                :param chain_name: where to store output files
                :type chain_name:
                :param resume:  ('resume', 'resume-similar', 'overwrite' or 'subfolder') –
        if ‘overwrite’, overwrite previous data.
        if ‘subfolder’, create a fresh subdirectory in log_dir.
        if ‘resume’ or True, continue previous run if available. Only works when dimensionality, transform or likelihood are consistent.
        if ‘resume-similar’, continue previous run if available. Only works when dimensionality and transform are consistent. If a likelihood difference is detected, the existing likelihoods are updated until the live point order differs. Otherwise, behaves like resume.
                :type resume: str
                :param wrapped_params:  (list of bools) – indicating whether this parameter wraps around (circular parameter).
                :type wrapped_params:
                :param stepsampler:
                :type stepsampler:
                :param use_mlfriends: Whether to use MLFriends+ellipsoidal+tellipsoidal region (better for multi-modal problems) or just ellipsoidal sampling (faster for high-dimensional, gaussian-like problems).
                :type use_mlfriends: bool
                :returns:

        """

        log.debug(
            f"Setup for UltraNest sampler: min_num_live_points:{min_num_live_points}, "
            f"chain_name:{chain_name}, dlogz: {dlogz}, wrapped_params: {wrapped_params}. "
            f"Other input: {kwargs}"
        )
        self._kwargs = {}
        self._kwargs["min_num_live_points"] = min_num_live_points
        self._kwargs["dlogz"] = dlogz
        self._kwargs["log_dir"] = chain_name
        self._kwargs["stepsampler"] = stepsampler
        self._kwargs["resume"] = resume

        self._wrapped_params = wrapped_params

        for k, v in kwargs.items():

            self._kwargs[k] = v

        self._use_mlfriends: bool = use_mlfriends

        self._is_setup: bool = True

    def sample(self, quiet=False):
        """
        sample using the UltraNest numerical integration method
        :rtype:

        :returns:

        """
        if not self._is_setup:

            log.info("You forgot to setup the sampler!")
            return

        loud = not quiet

        self._update_free_parameters()

        param_names = list(self._free_parameters.keys())

        loglike, ultranest_prior = self._construct_unitcube_posterior(return_copy=True)

        # We need to check if the MCMC
        # chains will have a place on
        # the disk to write and if not,
        # create one

        chain_name = self._kwargs.pop("log_dir")
        if chain_name is not None:
            mcmc_chains_out_dir = ""
            tmp = chain_name.split("/")
            for s in tmp[:-1]:
                mcmc_chains_out_dir += s + "/"

            if using_mpi:

                comm.Barrier()

                # if we are running in parallel and this is not the
                # first engine, then we want to wait and let everything finish

                if rank == 0:

                    # create mcmc chains directory only on first engine

                    if not os.path.exists(mcmc_chains_out_dir):
                        log.debug(f"Create {mcmc_chains_out_dir} for ultranest output")
                        os.makedirs(mcmc_chains_out_dir)



            else:

                if not os.path.exists(mcmc_chains_out_dir):
                    log.debug(f"Create {mcmc_chains_out_dir} for ultranest output")
                    os.makedirs(mcmc_chains_out_dir)

        # Multinest must be run parallel via an external method
        # see the demo in the examples folder!!

        if threeML_config["parallel"]["use_parallel"]:

            raise RuntimeError(
                "If you want to run ultranest in parallell you need to use an ad-hoc method"
            )

        else:

            resume = self._kwargs.pop("resume")

            sampler = ultranest.ReactiveNestedSampler(
                param_names,
                loglike,
                transform=ultranest_prior,
                log_dir=chain_name,
                vectorized=False,
                resume=resume,
                wrapped_params=self._wrapped_params,
            )

            if self._kwargs["stepsampler"] is not None:

                sampler.stepsampler = self._kwargs["stepsampler"]

            self._kwargs.pop("stepsampler")

            # use a different region class

            if not self._use_mlfriends:

                self._kwargs["region_class"] = ultranest.mlfriends.RobustEllipsoidRegion

            with use_astromodels_memoization(False):
                log.debug("Start ultranest run")
                sampler.run(show_status=loud, **self._kwargs)
                log.debug("Ultranest run done")

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

            results = sampler.results

            self._sampler = sampler

            ws = results["weighted_samples"]

            # Workaround to support older versions of ultranest
            try:
                wsamples = ws["v"]
                weights = ws["w"]
                logl = ws["L"]
            except KeyError:
                wsamples = ws["points"]
                weights = ws["weights"]
                logl = ws["logl"]

            # Get the log. likelihood values from the chain

            SQRTEPS = (float(np.finfo(np.float64).eps)) ** 0.5
            if abs(np.sum(weights) - 1.0) > SQRTEPS:  # same tol as in np.random.choice.
                raise ValueError("weights do not sum to 1")

            rstate = np.random

            N = len(weights)

            # make N subdivisions, and choose positions with a consistent random offset
            positions = (rstate.random() + np.arange(N)) / N

            idx = np.zeros(N, dtype=np.int)
            cumulative_sum = np.cumsum(weights)
            i, j = 0, 0
            while i < N:
                if positions[i] < cumulative_sum[j]:
                    idx[i] = j
                    i += 1
                else:
                    j += 1

            self._log_like_values = logl[idx]

            self._raw_samples = wsamples[idx]

            # now get the log probability

            self._log_probability_values = self._log_like_values + np.array(
                [self._log_prior(samples) for samples in self._raw_samples]
            )

            self._build_samples_dictionary()

            self._marginal_likelihood = sampler.results["logz"] / np.log(10.0)

            self._build_results()

            # Display results
            if loud:
                self._results.display()

            # now get the marginal likelihood

            return self.samples
