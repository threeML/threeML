import emcee
import emcee.utils
import numpy
import collections
import re
import math

import matplotlib.pyplot as plt

import uncertainties

from threeML.io.table import Table
from threeML.parallel.parallel_client import ParallelClient
from threeML.config.config import threeML_config
from threeML.io.progress_bar import ProgressBar
from threeML.io.triangle import corner
from threeML.exceptions.custom_exceptions import ModelAssertionViolation, LikelihoodIsInfinite, custom_warnings
from threeML.io.rich_display import display

from astromodels import uniform_prior, log_uniform_prior

def sample_with_progress(p0, sampler, n_samples, **kwargs):
    # Create progress bar

    progress = ProgressBar(n_samples)

    # Loop collecting n_samples samples

    pos, prob, state = [None, None, None]

    for i, result in enumerate(sampler.sample(p0, iterations=n_samples, **kwargs)):
        # Show progress

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

        # Get the initial list of free parameters, useful for debugging purposes

        self._update_free_parameters()

    def set_uniform_priors(self):
        """
        Automatically set all parameters to uniform or log-uniform priors. The latter is used when the range spanned
        by the parameter is larger than 2 orders of magnitude.

        :return: (none)
        """

        for parameter_name, parameter in self._free_parameters.iteritems():

            if parameter.min_value is None or parameter.max_value is None:

                custom_warnings.warn("Cannot decide the prior for parameter %s, since it has no "
                                     "minimum or no maximum value (or both)")

                continue

            n_orders_of_magnitude = numpy.log10(parameter.max_value - parameter.min_value)

            if n_orders_of_magnitude > 2:

                print("Using log-uniform prior for %s" % parameter_name)

                parameter.prior = log_uniform_prior(lower_bound=parameter.min_value,
                                                    upper_bound=parameter.max_value)

            else:

                print("Using uniform prior for %s" % parameter_name)

                parameter.prior = uniform_prior(lower_bound=parameter.min_value,
                                                upper_bound=parameter.max_value,
                                                value=1.0)

    def sample(self, n_walkers, burn_in, n_samples):
        """
        Sample the posterior with the Goodman & Weare's Affine Invariant Markov chain Monte Carlo
        """

        self._update_free_parameters()

        n_dim = len(self._free_parameters.keys())

        # Get starting point

        p0 = self._get_starting_points(n_walkers)

        if threeML_config['parallel']['use-parallel']:

            c = ParallelClient()
            view = c[:]

            sampler = emcee.EnsembleSampler(n_walkers, n_dim,
                                            self._get_posterior,
                                            pool=view)

        else:

            sampler = emcee.EnsembleSampler(n_walkers, n_dim,
                                            self._get_posterior)

        print("Running burn-in of %s samples...\n" % burn_in)

        pos, prob, state = sample_with_progress(p0, sampler, burn_in)

        # Reset sampler

        sampler.reset()

        # Run the true sampling

        print("\nSampling...\n")

        _ = sample_with_progress(pos, sampler, n_samples, rstate0=state)

        acc = numpy.mean(sampler.acceptance_fraction)

        print("Mean acceptance fraction: %s" % acc)

        self._sampler = sampler
        self._raw_samples = sampler.flatchain

        self._build_samples_dictionary()

        return self.samples

    def sample_parallel_tempering(self, n_temps, n_walkers, burn_in, n_samples):
        """
        Sample with parallel tempering
        """

        free_parameters = self._likelihood_model.getFreeParameters()

        n_dim = len(free_parameters.keys())

        sampler = emcee.PTSampler(n_temps, n_walkers, n_dim, self._log_like, self._logp)

        # Get one starting point for each temperature

        p0 = numpy.empty((n_temps, n_walkers, n_dim))

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

    def _build_samples_dictionary(self):
        """
        Build the dictionary to access easily the samples by parameter

        :return: none
        """

        self._samples = collections.OrderedDict()

        for i, (parameter_name, parameter) in enumerate(self._free_parameters.iteritems()):

            # Add the samples for this parameter for this source

            self._samples[parameter_name] = self._raw_samples[:,i]

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

    def get_credible_intervals(self, probability=95):
        """
        Print and returns the (equal-tail) credible intervals for all free parameters in the model

        :param probability: the probability for this credible interval (default: 95, corresponding to 95%)
        :return: a dictionary with the lower bound and upper bound of the credible intervals, as well as the median
        """

        # Gather the credible intervals (percentiles of the posterior)

        credible_intervals = collections.OrderedDict()

        for i, (parameter_name, parameter) in enumerate(self._free_parameters.iteritems()):

            # Get the percentiles from the posterior samples

            lower_bound,median,upper_bound = numpy.percentile(self.samples[parameter_name],
                                                              (100-probability,50,probability))

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

            number, unc_lower_bound, exponent = re.match('\(?(\-?[0-9]+\.?[0-9]+) ([0-9]+\.[0-9]+)\)?(e[\+|\-][0-9]+)?',
                                           x.__str__().replace("+/-", " ")).groups()

            # Process the positive "error"

            x = uncertainties.ufloat(median, abs(upper_bound - median))

            # Split the uncertainty in number, positive error, and exponent (if any)

            _, unc_upper_bound, _ = re.match('\(?(\-?[0-9]+\.?[0-9]+) ([0-9]+\.[0-9]+)\)?(e[\+|\-][0-9]+)?',
                                  x.__str__().replace("+/-", " ")).groups()

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

            fig = corner(self.raw_samples, labels=labels,
                         quantiles=[0.16, 0.50, 0.84],
                         priors=priors, **kwargs)

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

            subplot.set_ylabel(parameter_name.replace(".","\n"))
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

                idx1 = i*stepsize
                idx2 = idx1 + n_samples_in_each_subset

                if idx2 > n_samples - 1:

                    break

                this_averages.append(numpy.average(this_samples[idx1 : idx2]))
                this_variances.append(numpy.std(this_samples[idx1 : idx2]))

            averages[parameter_name] = this_averages

            variances[parameter_name] = this_variances

            # Now choose random samples and do the same

            this_bootstrap_averages = []
            this_bootstrap_variances = []

            for i in range(n_subsets):

                samples = numpy.random.choice(self.samples[parameter_name], n_samples)

                this_bootstrap_averages.append(numpy.average(samples))
                this_bootstrap_variances.append(numpy.std(samples))

            bootstrap_averages[parameter_name] = this_bootstrap_averages
            bootstrap_variances[parameter_name] = this_bootstrap_variances

        # Now plot all these things

        def plot_one_histogram(subplot, data, label):

            nbins = self.freedman_diaconis_rule(data)

            subplot.hist(data, nbins, label=label)

            subplot.locator_params(nbins=4)

        figures = []

        for i, parameter_name in enumerate(self._free_parameters.keys()):

            fig, subs = plt.subplots(1,2,sharey=True)

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

        q25, q75 = numpy.percentile(data, [25.0, 75.0])
        iqr = abs(q75 - q25)

        binsize = 2 * iqr * pow(len(data), -1/3.0)

        nbins = numpy.ceil((max(data)-min(data)) / binsize)

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

                return -numpy.inf

            else:

                parameter.value = trial_values[i]

                log_prior += math.log10(prior_value)

        # Get the value of the log-likelihood for this parameters

        try:

            # Loop over each dataset and get the likelihood values for each set

            log_like_values = map(lambda dataset: dataset.get_log_like(), self.data_list.values())

        except ModelAssertionViolation:

            # Fit engine or sampler outside of allowed zone

            return -numpy.inf

        except:

            # We don't want to catch more serious issues

            raise

        # Sum the values of the log-like

        log_like = numpy.sum(log_like_values)

        if not numpy.isfinite(log_like):
            # Issue warning

            custom_warnings.warn("Likelihood value is infinite for parameters %s" % trial_values, LikelihoodIsInfinite)

            return -numpy.inf

        #print("Log like is %s, log_prior is %s, for trial values %s" % (log_like, log_prior,trial_values))

        return log_like + log_prior

    def _get_starting_points(self, n_walkers, variance=0.1):

        # Generate the starting points for the walkers by getting random
        # values for the parameters close to the current value

        # Fractional variance for randomization
        # (0.1 means var = 0.1 * value )

        p0 = []

        for i in range(n_walkers):

            this_p0 = map(lambda x:x.get_randomized_value(variance), self._free_parameters.values())

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
        """Compute the log-likelihood, used in the parallel tempering sampling"""

        # Compute the log-likelihood

        # Set the parameters to their trial values

        for i, (src_name, param_name) in enumerate(self._free_parameters.keys()):
            this_param = self._likelihood_model.parameters[src_name][param_name]

            this_param.setValue(trial_values[i])

        # Get the value of the log-likelihood for this parameters

        try:

            # Loop over each dataset and get the likelihood values for each set

            log_like_values = map(lambda dataset: dataset.get_log_like(), self.data_list.values())

        except ModelAssertionViolation:

            # Fit engine or sampler outside of allowed zone

            return -numpy.inf

        except:

            # We don't want to catch more serious issues

            raise

        # Sum the values of the log-like

        log_like = numpy.sum(log_like_values)

        if not numpy.isfinite(log_like):
            # Issue warning

            custom_warnings.warn("Likelihood value is infinite for parameters %s" % trial_values, LikelihoodIsInfinite)

            return -numpy.inf

        return log_like
