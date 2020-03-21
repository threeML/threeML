from __future__ import print_function
from __future__ import division
from builtins import range
from builtins import object
from past.utils import old_div
import emcee
import emcee.utils

try:

    import pymultinest

except:

    has_pymultinest = False

else:

    has_pymultinest = True

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

        # Verify that all the free parameters have priors
        for parameter_name, parameter in likelihood_model.free_parameters.items():

            if not parameter.has_prior():
                raise RuntimeError("You need to define priors for all free parameters before instancing a "
                                   "Bayesian analysis")

        # Process optional keyword parameters

        self.verbose = False

        for k, v in kwargs.items():

            if k.lower() == "verbose":
                self.verbose = bool(kwargs["verbose"])

        self._likelihood_model = likelihood_model

        self._data_list = data_list

        for dataset in list(self._data_list.values()):

            dataset.set_model(self._likelihood_model)

            # Now get the nuisance parameters from the data and add them to the model
            # NOTE: it is important that this is *after* the setting of the model, as some
            # plugins might need to adjust the number of nuisance parameters depending on the
            # likelihood model

            for parameter_name, parameter in list(dataset.nuisance_parameters.items()):
                # Enforce that the nuisance parameter contains the instance name, because otherwise multiple instance
                # of the same plugin will overwrite each other's nuisance parameters

                assert dataset.name in parameter_name, "This is a bug of the plugin for %s: nuisance parameters " \
                                                       "must contain the instance name" % type(dataset)

                self._likelihood_model.add_external_parameter(parameter)

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

    def sample_multinest(self, n_live_points, chain_name="chains/fit-", quiet=False, **kwargs):
        """
        Sample the posterior with MULTINEST nested sampling (Feroz & Hobson)

        :param: n_live_points: number of MULTINEST livepoints
        :param: chain_names: where to stor the multinest incremental output
        :param: quiet: Whether or not to should results
        :param: **kwargs (pyMULTINEST kwords)

        :return: MCMC samples

        """

        assert has_pymultinest, "You don't have pymultinest installed, so you cannot run the Multinest sampler"

        self._update_free_parameters()

        n_dim = len(list(self._free_parameters.keys()))

        # MULTINEST has a convergence criteria and therefore, there is no way
        # to determine progress

        sampling_procedure = sample_without_progress

        # MULTINEST uses a different call signiture for
        # sampling so we construct callbakcs
        loglike, multinest_prior = self._construct_multinest_posterior()

        # We need to check if the MCMC
        # chains will have a place on
        # the disk to write and if not,
        # create one

        mcmc_chains_out_dir = ""
        tmp = chain_name.split('/')
        for s in tmp[:-1]:
            mcmc_chains_out_dir += s + '/'

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


        print("\nSampling\n")
        print("MULTINEST has its own convergence criteria... you will have to wait blindly for it to finish")
        print("If INS is enabled, one can monitor the likelihood in the terminal for completion information")

        # Multinest must be run parallel via an external method
        # see the demo in the examples folder!!

        if threeML_config['parallel']['use-parallel']:

            raise RuntimeError("If you want to run multinest in parallell you need to use an ad-hoc method")

        else:

            sampler = pymultinest.run(loglike,
                                      multinest_prior,
                                      n_dim,
                                      n_dim,
                                      outputfiles_basename=chain_name,
                                      n_live_points=n_live_points,
                                      **kwargs)

        # Use PyMULTINEST analyzer to gather parameter info

        process_fit = False

        if using_mpi:

            # if we are running in parallel and this is not the
            # first engine, then we want to wait and let everything finish

            if rank !=0:

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

            multinest_analyzer = pymultinest.analyse.Analyzer(n_params=n_dim,
                                                              outputfiles_basename=chain_name)

            # Get the log. likelihood values from the chain
            self._log_like_values = multinest_analyzer.get_equal_weighted_posterior()[:, -1]

            self._sampler = sampler

            self._raw_samples = multinest_analyzer.get_equal_weighted_posterior()[:, :-1]

            # now get the log probability

            self._log_probability_values = self._log_like_values +  np.array([self._log_prior(samples) for samples in self._raw_samples])

            self._build_samples_dictionary()

            self._marginal_likelihood = old_div(multinest_analyzer.get_stats()['global evidence'], np.log(10.))

            self._build_results()

            # Display results
            if not quiet:
                self._results.display()

            # now get the marginal likelihood



            return self.samples

    def _build_samples_dictionary(self):
        """
        Build the dictionary to access easily the samples by parameter

        :return: none
        """

        self._samples = collections.OrderedDict()

        for i, (parameter_name, parameter) in enumerate(self._free_parameters.items()):
            # Add the samples for this parameter for this source

            self._samples[parameter_name] = self._raw_samples[:, i]

    def _build_results(self):

        # Find maximum of the log posterior
        idx = self._log_probability_values.argmax()

        # Get parameter values at the maximum
        approximate_MAP_point = self._raw_samples[idx, :]

        # Sets the values of the parameters to their MAP values
        for i, parameter in enumerate(self._free_parameters):

            self._free_parameters[parameter].value = approximate_MAP_point[i]

        # Get the value of the posterior for each dataset at the MAP
        log_posteriors = collections.OrderedDict()

        log_prior = self._log_prior(approximate_MAP_point)

        # keep track of the total number of data points
        # and the total posterior

        total_n_data_points = 0

        total_log_posterior = 0

        for dataset in list(self._data_list.values()):


            log_posterior = dataset.get_log_like() + log_prior

            log_posteriors[dataset.name] = log_posterior

            total_n_data_points += dataset.get_number_of_data_points()

            total_log_posterior += log_posterior


        # compute the statistical measures

        statistical_measures = collections.OrderedDict()

        # compute the point estimates

        statistical_measures['AIC'] = aic(total_log_posterior,len(self._free_parameters),total_n_data_points)
        statistical_measures['BIC'] = bic(total_log_posterior,len(self._free_parameters),total_n_data_points)

        this_dic, pdic = dic(self)

        # compute the posterior estimates

        statistical_measures['DIC'] = this_dic
        statistical_measures['PDIC'] = pdic

        if self._marginal_likelihood is not None:

            statistical_measures['log(Z)'] = self._marginal_likelihood


        #TODO: add WAIC


        # Instance the result

        self._results = BayesianResults(self._likelihood_model, self._raw_samples, log_posteriors,statistical_measures=statistical_measures)

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

    def _construct_multinest_posterior(self):
        """
        pymultinest becomes confused with the self pointer. We therefore ceate callbacks
        that pymultinest can understand.

        Here, we construct the prior and log. likelihood for multinest on the unit cube
        """

        # First update the free parameters (in case the user changed them after the construction of the class)
        self._update_free_parameters()

        def loglike(trial_values, ndim, params):

            # NOTE: the _log_like function DOES NOT assign trial_values to the parameters

            for i, parameter in enumerate(self._free_parameters.values()):
                parameter.value = trial_values[i]

            log_like = self._log_like(trial_values)

            if self.verbose:
                n_par = len(self._free_parameters)

                print(
                "Trial values %s gave a log_like of %s" % (["%.2g" % trial_values[i] for i in range(n_par)],
                                                           log_like))

            return log_like

        # Now construct the prior
        # MULTINEST priors are defined on the unit cube
        # and should return the value in the bounds... not the
        # probability. Therefore, we must make some transforms

        def prior(params, ndim, nparams):

            for i, (parameter_name, parameter) in enumerate(self._free_parameters.items()):

                try:

                    params[i] = parameter.prior.from_unit_cube(params[i])

                except AttributeError:

                    raise RuntimeError("The prior you are trying to use for parameter %s is "
                                       "not compatible with multinest" % parameter_name)

        # Give a test run to the prior to check that it is working. If it crashes while multinest is going
        # it will not stop multinest from running and generate thousands of exceptions (argh!)
        n_dim = len(self._free_parameters)

        _ = prior([0.5] * n_dim, n_dim, [])

        return loglike, prior

    def _get_starting_points(self, n_walkers, variance=0.1):

        # Generate the starting points for the walkers by getting random
        # values for the parameters close to the current value

        # Fractional variance for randomization
        # (0.1 means var = 0.1 * value )

        p0 = []

        for i in range(n_walkers):
            this_p0 = [x.get_randomized_value(variance) for x in list(self._free_parameters.values())]

            p0.append(this_p0)

        return p0

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
