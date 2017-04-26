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

import numpy as np
import collections
import math
import os

import matplotlib.pyplot as plt

from threeML.parallel.parallel_client import ParallelClient
from threeML.config.config import threeML_config
from threeML.io.progress_bar import progress_bar
from threeML.exceptions.custom_exceptions import LikelihoodIsInfinite, custom_warnings
from threeML.analysis_results import BayesianResults
from threeML.utils.stats_tools import aic, bic, dic

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
        for parameter_name, parameter in likelihood_model.free_parameters.iteritems():

            if not parameter.has_prior():
                raise RuntimeError("You need to define priors for all free parameters before instancing a "
                                   "Bayesian analysis")

        # Process optional keyword parameters

        self.verbose = False

        for k, v in kwargs.iteritems():

            if k.lower() == "verbose":
                self.verbose = bool(kwargs["verbose"])

        self._likelihood_model = likelihood_model

        self._data_list = data_list

        for dataset in self._data_list.values():

            dataset.set_model(self._likelihood_model)

            # Now get the nuisance parameters from the data and add them to the model
            # NOTE: it is important that this is *after* the setting of the model, as some
            # plugins might need to adjust the number of nuisance parameters depending on the
            # likelihood model

            for parameter_name, parameter in dataset.nuisance_parameters.items():
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

    def sample(self, n_walkers, burn_in, n_samples, quiet=False):
        """
        Sample the posterior with the Goodman & Weare's Affine Invariant Markov chain Monte Carlo
        :param n_walkers:
        :param burn_in:
        :param n_samples:
        :param quiet: if False, do not print results

        :return: MCMC samples

        """

        self._update_free_parameters()

        n_dim = len(self._free_parameters.keys())

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

            # Sample the burn-in
            pos, prob, state = sampling_procedure(title="Burn-in", p0=p0, sampler=sampler, n_samples=burn_in)

            # Reset sampler

            sampler.reset()

            # Run the true sampling

            _ = sampling_procedure(title="Sampling", p0=pos, sampler=sampler, n_samples=n_samples, rstate0=state)

        acc = np.mean(sampler.acceptance_fraction)

        print("\nMean acceptance fraction: %s\n" % acc)

        self._sampler = sampler
        self._raw_samples = sampler.flatchain

        # Compute the corresponding values of the likelihood

        # First we need the prior
        log_prior = map(lambda x: self._log_prior(x), self._raw_samples)

        # Now we get the log posterior and we remove the log prior

        self._log_like_values = sampler.flatlnprobability - log_prior

        # we also want to store the log probability

        self._log_probability_values = sampler.flatlnprobability

        self._marginal_likelihood = None

        self._build_samples_dictionary()

        self._build_results()

        # Display results
        if not quiet:
            self._results.display()

        return self.samples

    def sample_parallel_tempering(self, n_temps, n_walkers, burn_in, n_samples):
        """
        Sample with parallel tempering

        :param: n_temps
        :param: n_walkers
        :param: burn_in
        :param: n_samples

        :return: MCMC samples

        """

        free_parameters = self._likelihood_model.getFreeParameters()

        n_dim = len(free_parameters.keys())

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

        self._raw_samples = sampler.flatchain.reshape(-1, sampler.flatchain.shape[-1])

        self._log_probability_values = None

        self._log_like_values = None

        self._marginal_likelihood = None

        self._build_samples_dictionary()

        return self.samples

    def sample_multinest(self, n_live_points, chain_name="chains/fit-", **kwargs):
        """
        Sample the posterior with MULTINEST nested sampling (Feroz & Hobson)

        :param: n_live_points
        :param: chain_names
        :param: **kwargs (pyMULTINEST kwords)

        :return: MCMC samples

        """

        assert has_pymultinest, "You don't have pymultinest installed, so you cannot run the Multinest sampler"

        self._update_free_parameters()

        n_dim = len(self._free_parameters.keys())

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
        multinest_analyzer = pymultinest.analyse.Analyzer(n_params=n_dim,
                                                          outputfiles_basename=chain_name)

        # Get the log. likelihood values from the chain
        self._log_like_values = multinest_analyzer.get_equal_weighted_posterior()[:, -1]

        self._sampler = sampler

        self._raw_samples = multinest_analyzer.get_equal_weighted_posterior()[:, :-1]

        # now get the log probability

        self._log_probability_values = map(lambda samples: self.get_posterior(samples), self._raw_samples)

        self._build_samples_dictionary()

        # now get the marginal likelihood

        self._marginal_likelihood = multinest_analyzer.get_stats()['global evidence'] / np.log(10.)

        return self.samples

    def _build_samples_dictionary(self):
        """
        Build the dictionary to access easily the samples by parameter

        :return: none
        """

        self._samples = collections.OrderedDict()

        for i, (parameter_name, parameter) in enumerate(self._free_parameters.iteritems()):
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

        for dataset in self._data_list.values():


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


    # def get_highest_density_interval(self, probability=95):
    #     """
    #     Print and returns the (non-equal-tail) highest density credible intervals for all free parameters in the model
    #
    #     :param probability: the probability for this credible interval (default: 95, corresponding to 95%)
    #     :return: a dictionary with the lower bound and upper bound of the credible intervals, as well as the median
    #     """
    #     # Gather the credible intervals (percentiles of the posterior)
    #
    #     credible_intervals = collections.OrderedDict()
    #
    #     for i, (parameter_name, parameter) in enumerate(self._free_parameters.iteritems()):
    #         # Get the percentiles from the posterior samples
    #
    #         lower_bound, upper_bound = self._hpd(self.samples[parameter_name], 1 - (float(probability) / 100.))
    #         median = np.median(self.samples[parameter_name])
    #
    #         # Save them in the dictionary
    #
    #         credible_intervals[parameter_name] = {'lower bound': lower_bound,
    #                                               'median': median,
    #                                               'upper bound': upper_bound}
    #
    #     # Print a table with the errors
    #
    #     data = []
    #     name_length = 0
    #
    #     for i, (parameter_name, parameter) in enumerate(self._free_parameters.iteritems()):
    #
    #         # Format the value and the error with sensible significant
    #         # numbers
    #
    #         lower_bound, median, upper_bound = [credible_intervals[parameter_name][key] for key in ('lower bound',
    #                                                                                                 'median',
    #                                                                                                 'upper bound')
    #                                             ]
    #
    #         # Process the negative "error"
    #
    #         x = uncertainties.ufloat(median, abs(lower_bound - median))
    #
    #         # Split the uncertainty in number, negative error, and exponent (if any)
    #
    #         number, unc_lower_bound, exponent = get_uncertainty_tokens(x)
    #
    #         # Process the positive "error"
    #
    #         x = uncertainties.ufloat(median, abs(upper_bound - median))
    #
    #         # Split the uncertainty in number, positive error, and exponent (if any)
    #
    #         _, unc_upper_bound, _ = get_uncertainty_tokens(x)
    #
    #         if exponent is None:
    #
    #             # Number without exponent
    #
    #             pretty_string = "%s -%s +%s" % (number, unc_lower_bound, unc_upper_bound)
    #
    #         else:
    #
    #             # Number with exponent
    #
    #             pretty_string = "(%s -%s +%s)%s" % (number, unc_lower_bound, unc_upper_bound, exponent)
    #
    #         unit = self._free_parameters[parameter_name].unit
    #
    #         data.append([parameter_name, pretty_string, unit])
    #
    #         if len(parameter_name) > name_length:
    #             name_length = len(parameter_name)
    #
    #     # Create and display the table
    #
    #     table = Table(rows=data,
    #                   names=["Name", "Value", "Unit"],
    #                   dtype=('S%i' % name_length, str, 'S15'))
    #
    #     display(table)
    #
    #     return credible_intervals

    # def get_credible_intervals(self, probability=68):
    #     """
    #     Print and returns the (equal-tail) credible intervals for all free parameters in the model
    #
    #     :param probability: the probability for this credible interval (default: 68, corresponding to 68%)
    #     :return: a dictionary with the lower bound and upper bound of the credible intervals, as well as the median
    #     """
    #
    #     # Gather the credible intervals (percentiles of the posterior)
    #
    #     credible_intervals = collections.OrderedDict()
    #
    #     for i, (parameter_name, parameter) in enumerate(self._free_parameters.iteritems()):
    #         # Get the percentiles from the posterior samples
    #
    #         lower_bound, median, upper_bound = np.percentile(self.samples[parameter_name],
    #                                                          (100 - probability, 50, probability))
    #
    #         # Save them in the dictionary
    #
    #         credible_intervals[parameter_name] = {'lower bound': lower_bound,
    #                                               'median': median,
    #                                               'upper bound': upper_bound}
    #
    #     # Print a table with the errors
    #
    #     data = []
    #     name_length = 0
    #
    #     for i, (parameter_name, parameter) in enumerate(self._free_parameters.iteritems()):
    #
    #         # Format the value and the error with sensible significant
    #         # numbers
    #
    #         lower_bound, median, upper_bound = [credible_intervals[parameter_name][key] for key in ('lower bound',
    #                                                                                                 'median',
    #                                                                                                 'upper bound')
    #                                             ]
    #
    #         # Process the negative "error"
    #
    #         x = uncertainties.ufloat(median, abs(lower_bound - median))
    #
    #         # Split the uncertainty in number, negative error, and exponent (if any)
    #
    #         number, unc_lower_bound, exponent = get_uncertainty_tokens(x)
    #
    #         # Process the positive "error"
    #
    #         x = uncertainties.ufloat(median, abs(upper_bound - median))
    #
    #         # Split the uncertainty in number, positive error, and exponent (if any)
    #
    #         _, unc_upper_bound, _ = get_uncertainty_tokens(x)
    #
    #         if exponent is None:
    #
    #             # Number without exponent
    #
    #             pretty_string = "%s -%s +%s" % (number, unc_lower_bound, unc_upper_bound)
    #
    #         else:
    #
    #             # Number with exponent
    #
    #             pretty_string = "(%s -%s +%s)%s" % (number, unc_lower_bound, unc_upper_bound, exponent)
    #
    #         unit = self._free_parameters[parameter_name].unit
    #
    #         data.append([parameter_name, pretty_string, unit])
    #
    #         if len(parameter_name) > name_length:
    #             name_length = len(parameter_name)
    #
    #     # Create and display the table
    #
    #     table = Table(rows=data,
    #                   names=["Name", "Value", "Unit"],
    #                   dtype=('S%i' % name_length, str, 'S15'))
    #
    #     display(table)
    #     print("\n(probability %s)" % probability)
    #
    #     return credible_intervals

    def corner_plot(self, renamed_parameters=None, **kwargs):
        """
        Produce the corner plot showing the marginal distributions in one and two directions.

        :param renamed_parameters: a python dictionary of parameters to rename.
             Useful when e.g. spectral indices in models have different names but you wish to compare them. Format is
             {'old label': 'new label'}
        :param kwargs: arguments to be passed to the corner function
        :return: a matplotlib.figure instance
        """


        DeprecationWarning('Please use <bayesian_analysis>.results.corner_plot. This feature will be removed in the future.')

        if self.samples is not None:

            return self._results.corner_plot(renamed_parameters, **kwargs)

        else:

            raise RuntimeError("You have to run the sampler first, using the sample() method")

    def corner_plot_cc(self, parameters=None, renamed_parameters=None, figsize='PAGE', **cc_kwargs):
        """
        Corner plots using chainconsumer which allows for nicer plotting of
        marginals
        see: https://samreay.github.io/ChainConsumer/chain_api.html#chainconsumer.ChainConsumer.configure
        for all options
        :param parameters: list of parameters to plot
        :param renamed_parameters: a python dictionary of parameters to rename.
             Useful when e.g. spectral indices in models have different names but you wish to compare them. Format is
             {'old label': 'new label'}
        :param **cc_kwargs: chainconsumer general keyword arguments
        :return fig:
        """

        DeprecationWarning(
            'Please use <bayesian_analysis>.results.corner_plot_cc. This feature will be removed in the future.')


        if self.samples is not None:

            return self._results.corner_plot_cc(parameters,renamed_parameters,figsize, **cc_kwargs)

        else:

            raise RuntimeError("You have to run the sampler first, using the sample() method")



    def compare_posterior(self, *other_fits, **kwargs):
        """
        Create a corner plot from many different bayesian fits which allow for co-plotting of parameters marginals.

        :param other_fits: other fitted Bayesian analysis objects
        :param parameters: parameters to plot
        :param renamed_parameters: a python dictionary of parameters to rename.
             Useful when e.g. spectral indices in models have different names but you wish to compare them. Format is
             {'old label': 'new label'}
        :param kwargs: chain consumer kwargs
        :return:

        Returns:

        """



        if self.samples is not None:
            return self._results.comparison_corner_plot(self, *other_fits, **kwargs)

        else:

            raise RuntimeError("You have to run the sampler first, using the sample() method")



    def plot_chains(self, thin=None):
        """
        Produce a plot of the series of samples for each parameter

        :parameter thin: use only one sample every 'thin' samples
        :return: a matplotlib.figure instance
        """

        figures = []

        for parameter_name in self._free_parameters.keys():

            figure, subplot = plt.subplots(1, 1)

            if thin is None:

                # Use all samples

                subplot.plot(self.samples[parameter_name])

            else:

                assert isinstance(thin, int), "Thin must be a integer number"

                subplot.plot(self.samples[parameter_name][::thin])

            subplot.set_ylabel(parameter_name.replace(".", "\n"))
            subplot.set_xlabel("sample #")

            figures.append(figure)

        return figures

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

        # Compute all the quantities

        averages = {}
        bootstrap_averages = {}

        variances = {}
        bootstrap_variances = {}

        n_samples = self._raw_samples[:, 0].shape[0]

        stepsize = n_samples // n_subsets

        assert stepsize > 10, "Too few samples for this method to be effective"

        print("Stepsize for sliding window is %s" % stepsize)

        for parameter_name in self._free_parameters.keys():

            this_samples = self.samples[parameter_name]

            # First compute averages and variances using the sliding window

            this_averages = []
            this_variances = []

            for i in range(n_subsets):

                idx1 = i * stepsize
                idx2 = idx1 + n_samples_in_each_subset

                if idx2 > n_samples - 1:
                    break

                this_averages.append(np.average(this_samples[idx1: idx2]))
                this_variances.append(np.std(this_samples[idx1: idx2]))

            averages[parameter_name] = this_averages

            variances[parameter_name] = this_variances

            # Now choose random samples and do the same

            this_bootstrap_averages = []
            this_bootstrap_variances = []

            for i in range(n_subsets):
                samples = np.random.choice(self.samples[parameter_name], n_samples)

                this_bootstrap_averages.append(np.average(samples))
                this_bootstrap_variances.append(np.std(samples))

            bootstrap_averages[parameter_name] = this_bootstrap_averages
            bootstrap_variances[parameter_name] = this_bootstrap_variances

        # Now plot all these things

        def plot_one_histogram(subplot, data, label):

            nbins = int(self.freedman_diaconis_rule(data))

            subplot.hist(data, nbins, label=label)

            subplot.locator_params(nbins=4)

        figures = []

        for i, parameter_name in enumerate(self._free_parameters.keys()):
            fig, subs = plt.subplots(1, 2, sharey=True)

            fig.suptitle(parameter_name)

            plot_one_histogram(subs[0], averages[parameter_name], 'sliding window')
            plot_one_histogram(subs[0], bootstrap_averages[parameter_name], 'bootstrap')

            subs[0].set_ylabel("N subsets")
            subs[0].set_xlabel("Average")

            plot_one_histogram(subs[1], variances[parameter_name], 'sliding window')
            plot_one_histogram(subs[1], bootstrap_variances[parameter_name], 'bootstrap')

            subs[1].set_xlabel("Std. deviation")

            figures.append(fig)

        return figures

    @staticmethod
    def freedman_diaconis_rule(data):
        """
        Returns the number of bins from the Freedman-Diaconis rule for a histogram of the given data

        :param data: an array of data
        :return: the optimal number of bins
        """

        q25, q75 = np.percentile(data, [25.0, 75.0])
        iqr = abs(q75 - q25)

        binsize = 2 * iqr * pow(len(data), -1 / 3.0)

        nbins = np.ceil((max(data) - min(data)) / binsize)

        return nbins

    def restore_median_fit(self):
        """
        Sets the model parameters to the mean of the marginal distributions
        """

        for i, (parameter_name, parameter) in enumerate(self._free_parameters.iteritems()):
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

        for i, (parameter_name, parameter) in enumerate(self._free_parameters.iteritems()):

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
                "Trial values %s gave a log_like of %s" % (map(lambda i: "%.2g" % trial_values[i], range(n_par)),
                                                           log_like))

            return log_like

        # Now construct the prior
        # MULTINEST priors are defined on the unit cube
        # and should return the value in the bounds... not the
        # probability. Therefore, we must make some transforms

        def prior(params, ndim, nparams):

            for i, (parameter_name, parameter) in enumerate(self._free_parameters.iteritems()):

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
            this_p0 = map(lambda x: x.get_randomized_value(variance), self._free_parameters.values())

            p0.append(this_p0)

        return p0

    def _log_prior(self, trial_values):
        """Compute the sum of log-priors, used in the parallel tempering sampling"""

        # Compute the sum of the log-priors

        log_prior = 0

        for i, (parameter_name, parameter) in enumerate(self._free_parameters.iteritems()):

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

            log_like_values = map(lambda dataset: dataset.get_log_like(), self._data_list.values())

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
