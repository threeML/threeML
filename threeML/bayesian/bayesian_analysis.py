import emcee
import emcee.utils

try:

    import pymultinest

except:

    has_pymultinest = False

else:

    has_pymultinest = True

import numpy as np
import collections
import math
import os

import matplotlib.pyplot as plt

import uncertainties

from threeML.io.table import Table
from threeML.parallel.parallel_client import ParallelClient
from threeML.config.config import threeML_config
from threeML.io.progress_bar import ProgressBar
from corner import corner
from threeML.exceptions.custom_exceptions import LikelihoodIsInfinite, custom_warnings
from threeML.io.rich_display import display
from threeML.utils.uncertainties_regexpr import get_uncertainty_tokens

from astromodels import ModelAssertionViolation


def sample_with_progress(p0, sampler, n_samples, **kwargs):
    # Create progress bar

    progress = ProgressBar(n_samples)

    # Loop collecting n_samples samples

    pos, prob, state = [None, None, None]

    # This is only for producing the progress bar
    progress_bar_iter = max(int(n_samples / 100), 1)

    for i, result in enumerate(sampler.sample(p0, iterations=n_samples, **kwargs)):
        # Show progress

        if i % progress_bar_iter == 0:
            progress.animate((i + 1))

        # Get the vectors with the results

        pos, prob, state = result

    # Make sure we show 100% completion

    progress.animate(n_samples)

    # Go to new line

    print("")

    return pos, prob, state


def sample_without_progress(p0, sampler, n_samples, **kwargs):
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

        self.data_list = data_list

        # Make sure that the current model is used in all data sets

        for dataset in self.data_list.values():
            dataset.set_model(self._likelihood_model)

        # Init the samples to None

        self._samples = None
        self._raw_samples = None
        self._sampler = None
        self._log_like_values = None

        # Get the initial list of free parameters, useful for debugging purposes

        self._update_free_parameters()

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

    def sample(self, n_walkers, burn_in, n_samples):
        """
        Sample the posterior with the Goodman & Weare's Affine Invariant Markov chain Monte Carlo
        :param: n_walkers
        :param: burn_in
        :param: n_samples

        :return: MCMC samples

        """

        self._update_free_parameters()

        n_dim = len(self._free_parameters.keys())

        # Get starting point

        p0 = self._get_starting_points(n_walkers)

        sampling_procedure = sample_with_progress

        if threeML_config['parallel']['use-parallel']:

            c = ParallelClient()
            view = c[:]

            sampler = emcee.EnsembleSampler(n_walkers, n_dim,
                                            self._get_posterior,
                                            pool=view)

            # Sampling with progress in parallel is super-slow, so let's
            # use the non-interactive one
            sampling_procedure = sample_without_progress

        else:

            sampler = emcee.EnsembleSampler(n_walkers, n_dim,
                                            self._get_posterior)

        print("Running burn-in of %s samples...\n" % burn_in)

        # Prepare the list of likelihood values
        self._log_like_values = []

        # Sample the burn-in
        pos, prob, state = sampling_procedure(p0, sampler, burn_in)

        # Reset sampler

        sampler.reset()

        # Run the true sampling

        print("\nSampling...\n")

        # Reset also the list of likelihood values
        self._log_like_values = []

        _ = sampling_procedure(pos, sampler, n_samples, rstate0=state)

        acc = np.mean(sampler.acceptance_fraction)

        print("Mean acceptance fraction: %s" % acc)

        self._sampler = sampler
        self._raw_samples = sampler.flatchain

        self._build_samples_dictionary()

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

        sampler = emcee.PTSampler(n_temps, n_walkers, n_dim, self._log_like, self._logp)

        # Get one starting point for each temperature

        p0 = np.empty((n_temps, n_walkers, n_dim))

        for i in range(n_temps):
            p0[i, :, :] = self._get_starting_points(n_walkers)

        print("Running burn-in of %s samples...\n" % burn_in)

        p, lnprob, lnlike = sample_with_progress(p0, sampler, burn_in)

        # Reset sampler

        sampler.reset()

        print("\nSampling...\n")

        _ = sample_with_progress(p, sampler, n_samples,
                                 lnprob0=lnprob, lnlike0=lnlike)

        self._sampler = sampler

        # Now build the _samples dictionary

        self._raw_samples = sampler.flatchain.reshape(-1, sampler.flatchain.shape[-1])

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

        print("\nSampling...\n")
        print("MULTINEST has its own convergence criteria... you will have to wait blindly for it to finish")
        print("If INS is enabled, one can monitor the likelihood in the terminal for completion information")

        # Multinest must be run parallel via an external method
        # see the demo in the examples folder!!

        # Reset the likelihood values
        self._log_like_values = []

        if threeML_config['parallel']['use-parallel']:

            raise RuntimeError("If you want to run multinest in paralell you need to use an ad-hoc method")

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

        self._build_samples_dictionary()

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

    def get_effective_free_parameters(self):
        """
        Calculates the effective number of free parameters from the posterior
         -2.*(mean(posterior)-max(log. likelihood))
        :return: Effective number of free parameters
        """

        return -2. * (np.mean(self._log_like_values) - np.max(self._log_like_values))  # need to check math!

    def get_highest_density_interval(self, probability=95):
        """
        Print and returns the (non-equal-tail) highest density credible intervals for all free parameters in the model

        :param probability: the probability for this credible interval (default: 95, corresponding to 95%)
        :return: a dictionary with the lower bound and upper bound of the credible intervals, as well as the median
        """
        # Gather the credible intervals (percentiles of the posterior)

        credible_intervals = collections.OrderedDict()

        for i, (parameter_name, parameter) in enumerate(self._free_parameters.iteritems()):
            # Get the percentiles from the posterior samples

            lower_bound, upper_bound = self._hpd(self.samples[parameter_name], 1 - (float(probability) / 100.))
            median = np.median(self.samples[parameter_name])

            # Save them in the dictionary

            credible_intervals[parameter_name] = {'lower bound': lower_bound,
                                                  'median': median,
                                                  'upper bound': upper_bound}

        # Print a table with the errors

        data = []
        name_length = 0

        for i, (parameter_name, parameter) in enumerate(self._free_parameters.iteritems()):

            # Format the value and the error with sensible significant
            # numbers

            lower_bound, median, upper_bound = [credible_intervals[parameter_name][key] for key in ('lower bound',
                                                                                                    'median',
                                                                                                    'upper bound')
                                                ]

            # Process the negative "error"

            x = uncertainties.ufloat(median, abs(lower_bound - median))

            # Split the uncertainty in number, negative error, and exponent (if any)

            number, unc_lower_bound, exponent = get_uncertainty_tokens(x)

            # Process the positive "error"

            x = uncertainties.ufloat(median, abs(upper_bound - median))

            # Split the uncertainty in number, positive error, and exponent (if any)

            _, unc_upper_bound, _ = get_uncertainty_tokens(x)

            if exponent is None:

                # Number without exponent

                pretty_string = "%s -%s +%s" % (number, unc_lower_bound, unc_upper_bound)

            else:

                # Number with exponent

                pretty_string = "(%s -%s +%s)%s" % (number, unc_lower_bound, unc_upper_bound, exponent)

            unit = self._free_parameters[parameter_name].unit

            data.append([parameter_name, pretty_string, unit])

            if len(parameter_name) > name_length:
                name_length = len(parameter_name)

        # Create and display the table

        table = Table(rows=data,
                      names=["Name", "Value", "Unit"],
                      dtype=('S%i' % name_length, str, 'S15'))

        display(table)

        return credible_intervals

    def get_credible_intervals(self, probability=68):
        """
        Print and returns the (equal-tail) credible intervals for all free parameters in the model

        :param probability: the probability for this credible interval (default: 68, corresponding to 68%)
        :return: a dictionary with the lower bound and upper bound of the credible intervals, as well as the median
        """

        # Gather the credible intervals (percentiles of the posterior)

        credible_intervals = collections.OrderedDict()

        for i, (parameter_name, parameter) in enumerate(self._free_parameters.iteritems()):
            # Get the percentiles from the posterior samples

            lower_bound, median, upper_bound = np.percentile(self.samples[parameter_name],
                                                             (100 - probability, 50, probability))

            # Save them in the dictionary

            credible_intervals[parameter_name] = {'lower bound': lower_bound,
                                                  'median': median,
                                                  'upper bound': upper_bound}

        # Print a table with the errors

        data = []
        name_length = 0

        for i, (parameter_name, parameter) in enumerate(self._free_parameters.iteritems()):

            # Format the value and the error with sensible significant
            # numbers

            lower_bound, median, upper_bound = [credible_intervals[parameter_name][key] for key in ('lower bound',
                                                                                                    'median',
                                                                                                    'upper bound')
                                                ]

            # Process the negative "error"

            x = uncertainties.ufloat(median, abs(lower_bound - median))

            # Split the uncertainty in number, negative error, and exponent (if any)

            number, unc_lower_bound, exponent = get_uncertainty_tokens(x)

            # Process the positive "error"

            x = uncertainties.ufloat(median, abs(upper_bound - median))

            # Split the uncertainty in number, positive error, and exponent (if any)

            _, unc_upper_bound, _ = get_uncertainty_tokens(x)

            if exponent is None:

                # Number without exponent

                pretty_string = "%s -%s +%s" % (number, unc_lower_bound, unc_upper_bound)

            else:

                # Number with exponent

                pretty_string = "(%s -%s +%s)%s" % (number, unc_lower_bound, unc_upper_bound, exponent)

            unit = self._free_parameters[parameter_name].unit

            data.append([parameter_name, pretty_string, unit])

            if len(parameter_name) > name_length:
                name_length = len(parameter_name)

        # Create and display the table

        table = Table(rows=data,
                      names=["Name", "Value", "Unit"],
                      dtype=('S%i' % name_length, str, 'S15'))

        display(table)
        print("\n(probability %s)" % probability)

        return credible_intervals

    def corner_plot(self, **kwargs):
        """
        Produce the corner plot showing the marginal distributions in one and two directions.

        :param kwargs: arguments to be passed to the corner function
        :return: a matplotlib.figure instance
        """

        if self.samples is not None:

            assert len(self._free_parameters.keys()) == self.raw_samples[0].shape[0], ("Mismatch between sample"
                                                                                       " dimensions and number of free"
                                                                                       " parameters")

            labels = []
            priors = []

            for i, (parameter_name, parameter) in enumerate(self._free_parameters.iteritems()):
                short_name = parameter_name.split(".")[-1]

                labels.append(short_name)

                priors.append(self._likelihood_model.parameters[parameter_name].prior)

            # default arguments
            default_args = {'show_titles': True, 'title_fmt': ".2g", 'labels': labels,
                            'quantiles': [0.16, 0.50, 0.84]}

            # Update the default arguents with the one provided (if any). Note that .update also adds new keywords,
            # if they weren't present in the original dictionary, so you can use any option in kwargs, not just
            # the one in default_args
            default_args.update(kwargs)

            fig = corner(self.raw_samples, **default_args)

            return fig

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

            nbins = self.freedman_diaconis_rule(data)

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

    def _update_free_parameters(self):
        """
        Update the dictionary of the current free parameters
        :return:
        """

        self._free_parameters = self._likelihood_model.free_parameters

    def _get_posterior(self, trial_values):
        """Compute the posterior for the normal sampler"""

        # Assign this trial values to the parameters and
        # store the corresponding values for the priors

        self._update_free_parameters()

        assert len(self._free_parameters) == len(trial_values), ("Something is wrong. Number of free parameters "
                                                                 "do not match the number of trial values.")

        log_prior = 0

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

            return self._log_like(trial_values)

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

    def _logp(self, trial_values):
        """Compute the sum of log-priors, used in the parallel tempering sampling"""

        # Compute the sum of the log-priors

        logp = 0

        for i, (src_name, param_name) in enumerate(self._free_parameters.keys()):
            this_param = self._likelihood_model.parameters[src_name][param_name]

            logp += this_param.prior(trial_values[i])

        return logp

    def _log_like(self, trial_values):
        """Compute the log-likelihood"""

        # Get the value of the log-likelihood for this parameters

        try:

            # Loop over each dataset and get the likelihood values for each set

            log_like_values = map(lambda dataset: dataset.get_log_like(), self.data_list.values())

        except ModelAssertionViolation:

            # Fit engine or sampler outside of allowed zone

            return -np.inf

        except:

            # We don't want to catch more serious issues

            raise

        # Sum the values of the log-like

        log_like = np.sum(log_like_values)

        self._log_like_values.append(log_like)

        if not np.isfinite(log_like):
            # Issue warning

            custom_warnings.warn("Likelihood value is infinite for parameters %s" % trial_values, LikelihoodIsInfinite)

            return -np.inf

        return log_like

    @staticmethod
    def _calc_min_interval(x, alpha):
        """
        Internal method to determine the minimum interval of a given width
        Assumes that x is sorted numpy array.
        :argument: a: a numpy array containing samples
        :argument: alpha: probability of type I error

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
        :Arguments:
        x : Numpy array
        An array containing MCMC samples
        alpha : float
        Desired probability of type I error (defaults to 0.05)
        """


        # Currently only 1D available.
        # future addition will fix this

        # Make a copy of trace
        #x = x.copy()
        # For multivariate node
        #if x.ndim > 1:
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
        #else:
            # Sort univariate node
        sx = np.sort(x)
        return np.array(self._calc_min_interval(sx, alpha))
