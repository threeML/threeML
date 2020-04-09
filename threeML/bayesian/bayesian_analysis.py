from __future__ import print_function
from __future__ import division
from builtins import range
from builtins import object
from past.utils import old_div
import emcee
import emcee.utils


try:

    import chainconsumer

except:

    has_chainconsumer = False

else:

    has_chainconsumer = True

try:

    # see if we have mpi and/or are using parallel

    from mpi4py import MPI
    if MPI.COMM_WORLD.Get_size() > 1: # need parallel capabilities
        using_mpi = True

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

    else:

        using_mpi = False
except:

    using_mpi = False








import numpy as np
import collections
import math
import os
import time

import matplotlib.pyplot as plt

from threeML.parallel.parallel_client import ParallelClient
from threeML.config.config import threeML_config
from threeML.io.progress_bar import progress_bar
from threeML.exceptions.custom_exceptions import LikelihoodIsInfinite, custom_warnings
from threeML.analysis_results import BayesianResults
from threeML.utils.statistics.stats_tools import aic, bic, dic

from astromodels import ModelAssertionViolation, use_astromodels_memoization


def sample_with_progress(title, p0, sampler, n_samples, **kwargs):
    # Loop collecting n_samples samples

    pos, prob, state = [None, None, None]

    # This is only for producing the progress bar

    with progress_bar(n_samples, title=title) as progress:
        for i, result in enumerate(sampler.sample(p0, iterations=n_samples, **kwargs)):
            # Show progress

            progress.animate((i + 1))

            # Get the vectors with the results

            pos, prob, state = result

    return pos, prob, state


def sample_without_progress(p0, sampler, n_samples, title=None, **kwargs):
    return sampler.run_mcmc(p0, n_samples, **kwargs)


class BayesianAnalysis(object):
    def __init__(self, likelihood_model, data_list, **kwargs):
        """
        Bayesian analysis.

        :param likelihood_model: the likelihood model
        :param data_list: the list of datasets to use (normally an instance of DataList)
        :param kwargs: use 'verbose=True' for verbose operation
        :return:
        """

        self._analysis_type = "bayesian"


        # # Make sure that the current model is used in all data sets
        #
        # for dataset in self.data_list.values():
        #     dataset.set_model(self._likelihood_model)

        # Init the samples to None

        self._samples = None
        self._raw_samples = None
        self._sampler = None
        self._log_like_values = None
        self._results = None

        # Get the initial list of free parameters, useful for debugging purposes

        self._update_free_parameters()

    @property
    def results(self):

        return self._results

    @property
    def analysis_type(self):
        return self._analysis_type

    @property
    def log_like_values(self):
        """
        Returns the value of the log_likelihood found by the bayesian sampler while sampling from the posterior. If
        you need to find the values of the parameters which generated a given value of the log. likelihood, remember
        that the samples accessible through the property .raw_samples are ordered in the same way as the vector
        returned by this method.

        :return: a vector of log. like values
        """
        return self._log_like_values

    @property
    def log_probability_values(self):
        """
        Returns the value of the log_probability (posterior) found by the bayesian sampler while sampling from the posterior. If
        you need to find the values of the parameters which generated a given value of the log. likelihood, remember
        that the samples accessible through the property .raw_samples are ordered in the same way as the vector
        returned by this method.

        :return: a vector of log probabilty values
        """

        return self._log_probability_values

    @property
    def log_marginal_likelihood(self):
        """
        Return the log marginal likelihood (evidence) if computed
        :return:
        """

        return self._marginal_likelihood

    def sample(self, n_walkers, burn_in, n_samples, quiet=False, seed=None):
        """
        Sample the posterior with the Goodman & Weare's Affine Invariant Markov chain Monte Carlo
        :param n_walkers:
        :param burn_in:
        :param n_samples:
        :param quiet: if False, do not print results
        :param seed: if provided, it is used to seed the random numbers generator before the MCMC

        :return: MCMC samples

        """

        self._update_free_parameters()

        n_dim = len(list(self._free_parameters.keys()))

        # Get starting point

        p0 = self._get_starting_points(n_walkers)

        sampling_procedure = sample_with_progress

        # Deactivate memoization in astromodels, which is useless in this case since we will never use twice the
        # same set of parameters
        with use_astromodels_memoization(False):

            if threeML_config['parallel']['use-parallel']:

                c = ParallelClient()
                view = c[:]

                sampler = emcee.EnsembleSampler(n_walkers, n_dim,
                                                self.get_posterior,
                                                pool=view)

                # Sampling with progress in parallel is super-slow, so let's
                # use the non-interactive one
                sampling_procedure = sample_without_progress

            else:

                sampler = emcee.EnsembleSampler(n_walkers, n_dim,
                                                self.get_posterior)

            # If a seed is provided, set the random number seed
            if seed is not None:

                sampler._random.seed(seed)

            # Sample the burn-in
            pos, prob, state = sampling_procedure(title="Burn-in", p0=p0, sampler=sampler, n_samples=burn_in)

            # Reset sampler

            sampler.reset()

            # Run the true sampling

            _ = sampling_procedure(title="Sampling", p0=pos, sampler=sampler, n_samples=n_samples, rstate0=state)

        acc = np.mean(sampler.acceptance_fraction)

        print("\nMean acceptance fraction: %s\n" % acc)

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
        if not quiet:
            self._results.display()

        return self.samples

    def sample_parallel_tempering(self, n_temps, n_walkers, burn_in, n_samples, quiet=False):
        """
        Sample with parallel tempering

        :param: n_temps
        :param: n_walkers
        :param: burn_in
        :param: n_samples

        :return: MCMC samples

        """

        free_parameters = self._likelihood_model.free_parameters

        n_dim = len(list(free_parameters.keys()))

        sampler = emcee.PTSampler(n_temps, n_walkers, n_dim, self._log_like, self._log_prior)

        # Get one starting point for each temperature

        p0 = np.empty((n_temps, n_walkers, n_dim))

        for i in range(n_temps):
            p0[i, :, :] = self._get_starting_points(n_walkers)

        print("Running burn-in of %s samples...\n" % burn_in)

        p, lnprob, lnlike = sample_with_progress("Burn-in", p0, sampler, burn_in)

        # Reset sampler

        sampler.reset()

        print("\nSampling\n")

        _ = sample_with_progress("Sampling", p, sampler, n_samples,
                                 lnprob0=lnprob, lnlike0=lnlike)

        self._sampler = sampler

        # Now build the _samples dictionary

        self._raw_samples = sampler.get_chain(flat=True).reshape(-1,
            sampler.get_chain(flat=True).shape[-1])

        self._log_probability_values = None

        self._log_like_values = None

        self._marginal_likelihood = None

        self._build_samples_dictionary()

        self._build_results()

        # Display results
        if not quiet:
            self._results.display()

        return self.samples




    @property
    def raw_samples(self):
        """
        Access the samples from the posterior distribution generated by the selected sampler in raw form (i.e.,
        in the format returned by the sampler)

        :return: the samples as returned by the sampler
        """

        return self._raw_samples

    @property
    def samples(self):
        """
        Access the samples from the posterior distribution generated by the selected sampler

        :return: a dictionary with the samples from the posterior distribution for each parameter
        """
        return self._samples

    @property
    def sampler(self):
        """
        Access the instance of the sampler used to sample the posterior distribution
        :return: an instance of the sampler
        """

        return self._sampler


    def plot_chains(self, thin=None):
        """
        Produce a plot of the series of samples for each parameter

        :parameter thin: use only one sample every 'thin' samples
        :return: a matplotlib.figure instance
        """

        return self._results.plot_chains( thin )

    @property
    def likelihood_model(self):
        """
        :return: likelihood model (a Model instance)
        """
        return self._likelihood_model

    @property
    def data_list(self):
        """
        :return: data list for this analysis
        """

        return self._data_list

    def convergence_plots(self, n_samples_in_each_subset, n_subsets):
        """
        Compute the mean and variance for subsets of the samples, and plot them. They should all be around the same
        values if the MCMC has converged to the posterior distribution.

        The subsamples are taken with two different strategies: the first is to slide a fixed-size window, the second
        is to take random samples from the chain (bootstrap)

        :param n_samples_in_each_subset: number of samples in each subset
        :param n_subsets: number of subsets to take for each strategy
        :return: a matplotlib.figure instance
        """

        return self._results.convergence_plots( n_samples_in_each_subset, n_subsets)
        

    def restore_median_fit(self):
        """
        Sets the model parameters to the mean of the marginal distributions
        """

        for i, (parameter_name, parameter) in enumerate(self._free_parameters.items()):
            # Add the samples for this parameter for this source

            mean_par = np.median(self._samples[parameter_name])
            parameter.value = mean_par

    def _update_free_parameters(self):
        """
        Update the dictionary of the current free parameters
        :return:
        """

        self._free_parameters = self._likelihood_model.free_parameters

    def get_posterior(self, trial_values):
        """Compute the posterior for the normal sampler"""

        # Assign this trial values to the parameters and
        # store the corresponding values for the priors

        # self._update_free_parameters()

        assert len(self._free_parameters) == len(trial_values), ("Something is wrong. Number of free parameters "
                                                                 "do not match the number of trial values.")

        log_prior = 0

        # with use_

        for i, (parameter_name, parameter) in enumerate(self._free_parameters.items()):

            prior_value = parameter.prior(trial_values[i])

            if prior_value == 0:
                # Outside allowed region of parameter space

                return -np.inf

            else:

                parameter.value = trial_values[i]

                log_prior += math.log10(prior_value)

        log_like = self._log_like(trial_values)

        # print("Log like is %s, log_prior is %s, for trial values %s" % (log_like, log_prior,trial_values))

        return log_like + log_prior



    def _log_prior(self, trial_values):
        """Compute the sum of log-priors, used in the parallel tempering sampling"""

        # Compute the sum of the log-priors

        log_prior = 0

        for i, (parameter_name, parameter) in enumerate(self._free_parameters.items()):

            prior_value = parameter.prior(trial_values[i])

            if prior_value == 0:
                # Outside allowed region of parameter space

                return -np.inf

            else:

                parameter.value = trial_values[i]

                log_prior += math.log10(prior_value)

        return log_prior

    def _log_like(self, trial_values):
        """Compute the log-likelihood"""

        # Get the value of the log-likelihood for this parameters

        try:

            # Loop over each dataset and get the likelihood values for each set

            log_like_values = [dataset.get_log_like() for dataset in list(self._data_list.values())]

        except ModelAssertionViolation:

            # Fit engine or sampler outside of allowed zone

            return -np.inf

        except:

            # We don't want to catch more serious issues

            raise

        # Sum the values of the log-like

        log_like = np.sum(log_like_values)

        if not np.isfinite(log_like):
            # Issue warning

            custom_warnings.warn("Likelihood value is infinite for parameters %s" % trial_values,
                                 LikelihoodIsInfinite)

            return -np.inf

        return log_like

    @staticmethod
    def _calc_min_interval(x, alpha):
        """
        Internal method to determine the minimum interval of a given width
        Assumes that x is sorted numpy array.
        :param a: a numpy array containing samples
        :param alpha: probability of type I error

        :returns: list containing min and max HDI

        """

        n = len(x)
        cred_mass = 1.0 - alpha

        interval_idx_inc = int(np.floor(cred_mass * n))
        n_intervals = n - interval_idx_inc
        interval_width = x[interval_idx_inc:] - x[:n_intervals]

        if len(interval_width) == 0:
            raise ValueError('Too few elements for interval calculation')

        min_idx = np.argmin(interval_width)
        hdi_min = x[min_idx]
        hdi_max = x[min_idx + interval_idx_inc]
        return hdi_min, hdi_max

    def _hpd(self, x, alpha=0.05):
        """Calculate highest posterior density (HPD) of array for given alpha.
        The HPD is the minimum width Bayesian credible interval (BCI).

        :param x: array containing MCMC samples
        :param alpha : Desired probability of type I error (defaults to 0.05)
        """

        # Currently only 1D available.
        # future addition will fix this

        # Make a copy of trace
        # x = x.copy()
        # For multivariate node
        # if x.ndim > 1:
        # Transpose first, then sort
        #    tx = np.transpose(x, list(range(x.ndim))[1:] + [0])
        #    dims = np.shape(tx)
        # Container list for intervals
        #    intervals = np.resize(0.0, dims[:-1] + (2,))

        #    sx = np.sort(tx[index])
        # Append to list
        #    intervals[index] = self._calc_min_interval(sx, alpha)
        # Transpose back before returning
        #    return np.array(intervals)
        # else:
        # Sort univariate node
        sx = np.sort(x)
        return np.array(self._calc_min_interval(sx, alpha))
