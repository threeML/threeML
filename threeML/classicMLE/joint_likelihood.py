import collections
import time
import numpy
import scipy.optimize
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import BoundaryNorm

from threeML.io.rich_display import display
import numpy as np
import pandas as pd
import uncertainties


from threeML.minimizer import minimization
from threeML.exceptions import custom_exceptions
from threeML.io.table import Table, NumericMatrix
from threeML.parallel.parallel_client import ParallelClient
from threeML.io.progress_bar import ProgressBar
from threeML.config.config import threeML_config
from threeML.exceptions.custom_exceptions import custom_warnings, FitFailed
from threeML.utils.uncertainties_regexpr import get_uncertainty_tokens


from astromodels import ModelAssertionViolation

class ReducingNumberOfThreads(Warning):
    pass


class ReducingNumberOfSteps(Warning):
    pass


class NotANumberInLikelihood(Warning):
    pass


class JointLikelihood(object):
    def __init__(self, likelihood_model, data_list, verbose=False):
        """
        Implement a joint likelihood analysis.

        :param likelihood_model: the model for the likelihood analysis
        :param data_list: the list of data sets (plugin instances) to be used in this analysis
        :param verbose: (True or False) print every step in the -log likelihood minimization
        :return:
        """

        # Process optional keyword parameters
        self.verbose = verbose

        self._likelihood_model = likelihood_model

        self._data_list = data_list

        for dataset in self._data_list.values():

            dataset.set_model(self._likelihood_model)

        # This is to keep track of the number of calls to the likelihood
        # function
        self.ncalls = 0

        # Pre-defined minimizer is Minuit

        self.set_minimizer("MINUIT")

        # Initial set of free parameters

        self._free_parameters = self._likelihood_model.free_parameters

        # Initially set the value of _current_minimum to None, it will be change by the fit() method

        self._current_minimum = None

        self._minimizer = None

    @property
    def likelihood_model(self):
        """
        :return: likelihood model for this analysis
        """

        return self._likelihood_model

    @property
    def data_list(self):
        """
        :return: data list for this analysis
        """

        return self._data_list

    @property
    def current_minimum(self):
        """
        :return: current minimum of the joint likelihood (available only after the fit() method)
        """
        return self._current_minimum

    @property
    def minimizer(self):
        """
        :return: an instance of the minimizer used in the fit (available only after the fit() method)
        """
        return self._minimizer

    @property
    def covariance_matrix(self):
        """
        :return: covariance matrix from the last fit
        """
        try:

            return self._minimizer.covariance_matrix

        except AttributeError:

            raise RuntimeError("You need to run a fit before accessing the covariance matrix")

    @property
    def correlation_matrix(self):
        """
        :return: correlation matrix from the last fit
        """

        try:

            return self._minimizer.correlation_matrix

        except AttributeError:

            raise RuntimeError("You need to run a fit before accessing the correlation matrix")

    def _update_free_parameters(self):

        """Update the dictionary of free parameters"""

        self._free_parameters = self._likelihood_model.free_parameters

    def fit(self):
        """
        Perform a fit of the current likelihood model on the datasets

        :param pre_fit: (True or False) If True, perform a pre-fit with only normalizations free (experimental)
        :return: a dictionary with the results on the parameters, and the values of the likelihood at the minimum
                 for each dataset and the total one.
        """

        # Update the list of free parameters, to be safe against changes the user might do between
        # the creation of this class and the calling of this method

        self._update_free_parameters()

        # Now check and fix if needed all the deltas of the parameters
        # to 20% of their value (otherwise the fit will be super-slow)

        for k, v in self._free_parameters.iteritems():

            if abs(v.delta) < abs(v.value) * 0.2:

                v.delta = abs(v.value) * 0.2

        # Instance the minimizer

        self._minimizer = self._get_minimizer(self.minus_log_like_profile,
                                              self._free_parameters)

        # Perform the fit

        xs, log_likelihood_minimum = self._minimizer.minimize()

        if log_likelihood_minimum == minimization.FIT_FAILED:

            raise FitFailed("The fit failed to converge.")

        # Store the current minimum for the -log likelihood

        self._current_minimum = float(log_likelihood_minimum)

        # Print results

        # Get the results from the minimizer (a panda container)

        fit_results = self._minimizer.fit_results

        # Now produce an ad-hoc display. We don't use the pandas display methods because
        # we want to display uncertainties with the right number of significant numbers

        data = []

        # Also store the maximum length to decide the length for the line

        name_length = 0

        for i, parameter_name in enumerate(fit_results.index.values):

            value = fit_results.at[parameter_name, 'value']

            error = fit_results.at[parameter_name, 'error']

            # Format the value and the error with sensible significant
            # numbers
            x = uncertainties.ufloat(value, error)

            # Add some space around the +/- sign

            rep = x.__str__().replace("+/-", " +/- ")

            data.append([i, parameter_name, rep, self._free_parameters[parameter_name].unit])

            if len(parameter_name) > name_length:

                name_length = len(parameter_name)

        best_fit_table = Table(rows=data,
                      names=["#", "Name", "Best fit value", "Unit"],
                      dtype=(str, 'S%i' % name_length, str, str))

        print("Best fit values:\n")

        display(best_fit_table)

        print("\nNOTE: errors on parameters are approximate. Use get_errors().\n")

        # Now display the nuisance parameters

        nuisance_parameters = collections.OrderedDict()

        for dataset in self._data_list.values():

            for pName in dataset.get_nuisance_parameters():

                nuisance_parameters[(dataset.get_name(), pName)] = dataset.nuisanceParameters[pName]

        nuisance_parameters_table = self._get_table_of_parameters(nuisance_parameters)

        print("Nuisance parameters:\n")

        display(nuisance_parameters_table)

        print("\nCorrelation matrix:\n")

        best_fit_table = NumericMatrix(self._minimizer.correlation_matrix)

        for col in best_fit_table.colnames:

            best_fit_table[col].format = '2.2f'

        display(best_fit_table)

        # Now collect the values for the likelihood for the various datasets

        # First restore best fit (to make sure we compute the likelihood at the right point)
        self._minimizer.restore_best_fit()

        # Fill the dictionary with the values of the -log likelihood (total, and dataset by dataset)

        minus_log_likelihood_values = collections.OrderedDict()

        minus_log_likelihood_values['all'] = self._current_minimum

        total = 0

        for dataset in self._data_list.values():

            ml = dataset.inner_fit() * (-1)

            minus_log_likelihood_values[dataset.get_name()] = ml

            total += ml

        assert total == self._current_minimum, "Current minimum stored after fit and current do not correspond!"

        print("\nValues of -log(likelihood) at the minimum:\n")

        loglike_dataframe = pd.DataFrame(minus_log_likelihood_values.items(), columns=['instrument','-log(likelihood)'])

        display(loglike_dataframe)

        return fit_results, loglike_dataframe

    def get_errors(self):
        """
        Compute the errors on the parameters using the profile likelihood method.

        :return: a dictionary containing the asymmetric errors for each parameter.
        """

        # Check that the user performed a fit first

        assert self._current_minimum is not None, "You have to run the .fit method before calling errors."

        errors = self._minimizer.get_errors()

        # Set the parameters back to the best fit value
        self.restore_best_fit()

        # Print a table with the errors

        data = []
        name_length = 0

        for k, v in self._free_parameters.iteritems():

            # Format the value and the error with sensible significant
            # numbers

            # Process the negative error

            neg_error = abs(errors[k][0])

            if np.isnan(neg_error):

                x = uncertainties.ufloat(v.value, 0)

            else:

                x = uncertainties.ufloat(v.value, neg_error)

            # Split the uncertainty in number, negative error, and exponent (if any)

            num, uncm, exponent = get_uncertainty_tokens(x)

            if np.isnan(neg_error):

                uncm = np.nan

            # Process the positive error

            pos_error = abs(errors[k][1])

            if np.isnan(pos_error):

                x = uncertainties.ufloat(v.value, 0)

            else:

                x = uncertainties.ufloat(v.value, pos_error)

            # Split the uncertainty in number, positive error, and exponent (if any)

            _, uncp, _ = get_uncertainty_tokens(x)

            if np.isnan(pos_error):

                uncp = np.nan

            if exponent is None:

                # Number without exponent

                pretty_string = "%s -%s +%s" % (num, uncm, uncp)

            else:

                # Number with exponent

                pretty_string = "(%s -%s +%s)%s" % (num, uncm, uncp, exponent)

            data.append([k, pretty_string, v.unit])

            if len(k) > name_length:
                name_length = len(k)

        # Create and display the table

        table = Table(rows=data,
                      names=["Name", "Value", "Unit"],
                      dtype=('S%i' % name_length, str, str))

        display(table)

    def get_contours(self, param_1, param_1_minimum, param_1_maximum, param_1_n_steps,
                     param_2=None, param_2_minimum=None, param_2_maximum=None, param_2_n_steps=None,
                     progress=True, **options):
        """
        Generate confidence contours for the given parameters by stepping for the given number of steps between
        the given boundaries. Call it specifying only source_1, param_1, param_1_minimum and param_1_maximum to
        generate the profile of the likelihood for parameter 1. Specify all parameters to obtain instead a 2d
        contour of param_1 vs param_2.

        NOTE: if using parallel computation, param_1_n_steps must be an integer multiple of the number of running
        engines. If that is not the case, the code will reduce the number of steps to match that requirement, and
        issue a warning

        :param param_1: fully qualified name of the first parameter or parameter instance
        :param param_1_minimum: lower bound for the range for the first parameter
        :param param_1_maximum: upper bound for the range for the first parameter
        :param param_1_n_steps: number of steps for the first parameter
        :param param_2: fully qualified name of the second parameter or parameter instance
        :param param_2_minimum: lower bound for the range for the second parameter
        :param param_2_maximum: upper bound for the range for the second parameter
        :param param_2_n_steps: number of steps for the second parameter
        :param progress: (True or False) whether to display progress or not
        :param log: by default the steps are taken linearly. With this optional parameter you can provide a tuple of
                    booleans which specify whether the steps are to be taken logarithmically. For example,
                    'log=(True,False)' specify that the steps for the first parameter are to be taken logarithmically,
                    while they are linear for the second parameter. If you are generating the profile for only one
                    parameter, you can specify 'log=(True,)' or 'log=(False,)' (optional)
        :return: a tuple containing an array corresponding to the steps for the first parameter, an array corresponding
                 to the steps for the second parameter (or None if stepping only in one direction), a matrix of size
                 param_1_steps x param_2_steps containing the value of the function at the corresponding points in the
                 grid. If param_2_steps is None (only one parameter), then this reduces to an array of
                 size param_1_steps.
        """

        if hasattr(param_1,"value"):

            # Substitute with the name
            param_1 = param_1.path

        if hasattr(param_2,'value'):

            param_2 = param_2.path

        # Check that the parameters exist
        assert param_1 in self._likelihood_model.free_parameters, "Parameter %s is not a free parameters of the " \
                                                                 "current model" % param_1

        if param_2 is not None:
            assert param_2 in self._likelihood_model.free_parameters, "Parameter %s is not a free parameters of the " \
                                                                      "current model" % param_2


        # Check that we have a valid fit

        assert self._current_minimum is not None, "You have to run the .fit method before calling get_contours."

        # Then restore the best fit

        self._minimizer.restore_best_fit()

        # Check minimal assumptions about the procedure

        assert not (param_1 == param_2), "You have to specify two different parameters"

        assert param_1_minimum < param_1_maximum, "Minimum larger than maximum for parameter 1"

        if param_2 is not None:
            assert param_2_minimum < param_2_maximum, "Minimum larger than maximum for parameter 2"

        # Check whether we are parallelizing or not

        if not threeML_config['parallel']['use-parallel']:

            a, b, cc = self.minimizer.contours(param_1, param_1_minimum, param_1_maximum, param_1_n_steps,
                                               param_2, param_2_minimum, param_2_maximum, param_2_n_steps,
                                               progress, **options)

            # Collapse the second dimension of the results if we are doing a 1d contour

            if param_2 is None:
                cc = cc[:, 0]

        else:

            # With parallel computation

            # In order to distribute fairly the computation, the strategy is to parallelize the computation
            # by assigning to the engines one "line" of the grid at the time

            # Connect to the engines

            client = ParallelClient(**options)

            # Get the number of engines

            n_engines = client.get_number_of_engines()

            # Check whether the number of threads is larger than the number of steps in the first direction

            if n_engines > param_1_n_steps:

                n_engines = int(param_1_n_steps)

                custom_warnings.warn("The number of engines is larger than the number of steps. Using only %s engines."
                                     % n_engines, ReducingNumberOfThreads)

            # Check if the number of steps is divisible by the number
            # of threads, otherwise issue a warning and make it so

            if float(param_1_n_steps) % n_engines != 0:
                # Set the number of steps to an integer multiple of the engines
                # (note that // is the floor division, also called integer division)

                param_1_n_steps = (param_1_n_steps // n_engines) * n_engines

                custom_warnings.warn("Number of steps is not a multiple of the number of threads. Reducing steps to %s"
                                     % param_1_n_steps, ReducingNumberOfSteps)

            # Compute the number of splits, i.e., how many lines in the grid for each engine.
            # (note that this is guaranteed to be an integer number after the previous checks)

            p1_split_steps = param_1_n_steps // n_engines

            # Prepare arrays for results

            if param_2 is None:

                # One array
                pcc = numpy.zeros(param_1_n_steps)

                pa = numpy.linspace(param_1_minimum, param_1_maximum, param_1_n_steps)
                pb = None

            else:

                pcc = numpy.zeros((param_1_n_steps, param_2_n_steps))

                # Prepare the two axes of the parameter space
                pa = numpy.linspace(param_1_minimum, param_1_maximum, param_1_n_steps)
                pb = numpy.linspace(param_2_minimum, param_2_maximum, param_2_n_steps)

            # Define the parallel worker which will go through the computation

            # NOTE: I only divide
            # on the first parameter axis so that the different
            # threads are more or less well mixed for points close and
            # far from the best fit

            def worker(start_index):

                # Re-create the minimizer

                backup_freeParameters = map(lambda x:x.value, self._likelihood_model.free_parameters.values())

                this_minimizer = self._get_minimizer(self.minus_log_like_profile,
                                                     self._free_parameters)

                this_p1min = pa[start_index * p1_split_steps]
                this_p1max = pa[(start_index + 1) * p1_split_steps - 1]

                # print("From %s to %s" % (this_p1min, this_p1max))

                aa, bb, ccc = this_minimizer.contours(param_1, this_p1min, this_p1max, p1_split_steps,
                                                      param_2, param_2_minimum, param_2_maximum,
                                                      param_2_n_steps,
                                                      False, **options)

                # Restore best fit values

                for val, par in zip(backup_freeParameters, self._likelihood_model.free_parameters.values()):

                    par.value = val

                return ccc

            # Get a balanced view of the engines

            lview = client.load_balanced_view()
            # lview.block = True

            # Distribute the work among the engines and start it, but return immediately the control
            # to the main thread

            amr = lview.map_async(worker, range(n_engines))

            # print progress

            progress = ProgressBar(n_engines)

            # This loop will check from time to time the status of the computation, which is happening on
            # different threads, and update the progress bar

            while not amr.ready():
                # Check and report the status of the computation every second

                time.sleep(1 + np.random.uniform(0, 1))

                # if (debug):
                #     stdouts = amr.stdout
                #
                #     # clear_output doesn't do much in terminal environments
                #     for stdout, stderr in zip(amr.stdout, amr.stderr):
                #         if stdout:
                #             print "%s" % (stdout[-1000:])
                #         if stderr:
                #             print "%s" % (stderr[-1000:])
                #     sys.stdout.flush()

                progress.animate(amr.progress - 1)

            # Always display 100% at the end

            progress.animate(n_engines - 1)

            # Add a new line after the progress bar
            print("\n")

            # print("Serial time: %1.f (speed-up: %.1f)" %(amr.serial_time, float(amr.serial_time) / amr.wall_time))

            # Get the results. This will raise exceptions if something wrong happened during the computation.
            # We don't catch it so that the user will be aware of that

            res = amr.get()

            # Now re-assemble the vector of results taking the different parts from the engines

            for i in range(n_engines):

                if param_2 is None:

                    pcc[i * p1_split_steps: (i + 1) * p1_split_steps] = res[i][:, 0]

                else:

                    pcc[i * p1_split_steps: (i + 1) * p1_split_steps, :] = res[i]

            # Give the results the names that the following code expect. These are kept separate for debugging
            # purposes

            cc = pcc
            a = pa
            b = pb

        # Here we have done the computation, in parallel computation or not. Let's make the plot
        # with the contour

        if param_2 is not None:

            # 2d contour

            fig = self._plot_contours("%s" % (param_1), a, "%s" % (param_2,), b, cc)

        else:

            # 1d contour (i.e., a profile)

            fig = self._plot_profile("%s" % (param_1), a, cc)

        # Check if we found a better minimum. This shouldn't happen, but in case of very difficult fit
        # it might.

        if self._current_minimum - cc.min() > 0.1:

            if param_2 is not None:

                idx = cc.argmin()

                aidx, bidx = numpy.unravel_index(idx, cc.shape)

                print("\nFound a better minimum: %s with %s = %s and %s = %s. Run again your fit starting from here."
                      % (cc.min(), param_1, a[aidx], param_2, b[bidx]))

            else:

                idx = cc.argmin()

                print("Found a better minimum: %s with %s = %s. Run again your fit starting from here."
                      % (cc.min(), param_1, a[idx]))

        return a, b, cc, fig

    def minus_log_like_profile(self, *trial_values):
        """
        Return the minus log likelihood for a given set of trial values

        :param trial_values: the trial values. Must be in the same number as the free parameters in the model
        :return: minus log likelihood
        """

        # Keep track of the number of calls
        self.ncalls += 1

        # Transform the trial values in a numpy array

        trial_values = numpy.array(trial_values)

        # Check that there are no nans within the trial values

        # This is the fastest way to check for any nan
        # (try other methods if you don't believe me)

        if not numpy.isfinite(numpy.dot(trial_values, trial_values.T)):
            # There are nans, something weird is going on. Return FIT_FAILED so the engine
            # stays away from this (or fail)

            return minimization.FIT_FAILED

        # Assign the new values to the parameters

        for i, parameter in enumerate(self._free_parameters.values()):

            parameter.value = trial_values[i]

        # Now profile out nuisance parameters and compute the new value
        # for the likelihood

        summed_log_likelihood = 0

        for dataset in self._data_list.values():

            try:

                this_log_like = dataset.inner_fit()

            except ModelAssertionViolation:

                # This is a zone of the parameter space which is not allowed. Return
                # a big number for the likelihood so that the fit engine will avoid it

                custom_warnings.warn("Fitting engine in forbidden space: %s" % (trial_values,),
                                     custom_exceptions.ForbiddenRegionOfParameterSpace)

                return minimization.FIT_FAILED

            except:

                # Do not intercept other errors

                raise

            # if this_log_like > 0:

            #    raise RuntimeError("LogLike cannot be larger than 0")

            summed_log_likelihood += this_log_like

            # dataset.getLogLike()

        # Check that the global like is not NaN
        # I use this weird check because it is not guaranteed that the plugins return np.nan,
        # especially if they are written in something other than python

        if "%s" % summed_log_likelihood == 'nan':
            custom_warnings.warn("These parameters returned a logLike = Nan: %s" % (trial_values,),
                                 NotANumberInLikelihood)

            return minimization.FIT_FAILED

        if self.verbose:
            print("Trying with parameters %s, resulting in logL = %s" % (trial_values, summed_log_likelihood))

        # Return the minus log likelihood

        return summed_log_likelihood * (-1)

    def set_minimizer(self, minimizer, algorithm=None):
        """
        Set the minimizer to be used, among those available. At the moment these are supported:

        * ROOT
        * MINUIT (which means iminuit, default)
        * PYOPT

        :param minimizer: the name of the new minimizer.
        :param algorithm: (optional) for the PYOPT minimizer, specify the algorithm among those available. See a list at
        http://www.pyopt.org/reference/optimizers.html . For the other minimizers this is ignored, if provided.
        :return: (none)
        """

        if minimizer.upper() == "MINUIT":

            self._minimizer_name = "MINUIT"
            self._minimizer_algorithm = None

        elif minimizer.upper() == "ROOT":

            assert minimization.has_ROOT, "pyROOT not available on this system. Cannot use ROOT minimizer."

            self._minimizer_name = "ROOT"
            self._minimizer_algorithm = None

        elif minimizer.upper() == "PYOPT":

            assert minimization.has_pyOpt, "pyOpt not available on this system. Cannot use PYOPT minimizer."

            assert algorithm is not None, "You have to provide an algorithm for the PYOPT minimizer. " \
                                          "See http://www.pyopt.org/reference/optimizers.html"

            self._minimizer_name = "PYOPT"

            assert algorithm.upper() in minimization._pyopt_algorithms, "Provided algorithm not available. " \
                                                                        "Available algorithms are: " \
                                                                        "%s" % minimization._pyopt_algorithms.keys()
            self._minimizer_algorithm = algorithm.upper()

        else:

            raise ValueError("Do not know minimizer %s" % minimizer)

        # Get the type

        self._minimizer_type = minimization.get_minimizer(self._minimizer_name)

    def _get_minimizer(self, *args, **kwargs):

        minimizer_instance = self._minimizer_type(*args, **kwargs)

        if self._minimizer_algorithm:

            minimizer_instance.set_algorithm(self._minimizer_algorithm)

        return minimizer_instance

    def restore_best_fit(self):
        """
        Restore the model to its best fit

        :return: (none)
        """

        if self._minimizer:

            self._minimizer.restore_best_fit()

        else:

            custom_warnings.warn("Cannot restore best fit, since fit has not been executed.")

    def _get_table_of_parameters(self, parameters):

        data = []
        max_length_of_name = 0

        for k, v in parameters.iteritems():

            current_name = "%s_of_%s" % (k[1], k[0])

            data.append([current_name, "%s" % v.value, v.unit])

            if len(v.name) > max_length_of_name:
                max_length_of_name = len(current_name)

        table = Table(rows=data,
                      names=["Name", "Value", "Unit"],
                      dtype=('S%i' % max_length_of_name, str, 'S15'))

        return table

    def _plot_profile(self, name1, a, cc):
        """
        Plot the likelihood profile.

        :param name1: Name of parameter
        :param a: grid for the parameter
        :param cc: log. likelihood values for the parameter
        :return: a figure containing the likelihood profile
        """

        # plot 1,2 and 3 sigma horizontal lines

        sigmas = [1, 2, 3]

        # Compute the corresponding probability. We do not
        # pre-compute them because we will introduce at
        # some point the possibility to personalize the
        # levels

        probabilities = []

        for s in sigmas:
            # One-sided probability
            # It is one-sided because we consider one side at the time
            # when computing the error

            probabilities.append(1 - (scipy.stats.norm.sf(s) * 2))

        # Compute the corresponding delta chisq. (chisq has 1 d.o.f.)

        # noinspection PyTypeChecker
        delta_chi2 = np.array(scipy.stats.chi2.ppf(probabilities, 1) / 2.0)  # two-sided!

        fig = plt.figure()
        sub = fig.add_subplot(111)

        # Neutralize values of the loglike too high
        # (fit failed)
        idx = (cc == minimization.FIT_FAILED)

        sub.plot(a[~idx], cc[~idx], lw=2)

        # Now plot the failed fits as "x"

        sub.plot(a[idx], [cc.min()] * a[idx].shape[0], 'x', c='red', markersize=2)

        # Decide colors
        colors = ['blue', 'cyan', 'red']

        for s, d, c in zip(sigmas, delta_chi2, colors):
            sub.axhline(self._current_minimum + d, linestyle='--',
                        color=c, label=r'%s $\sigma$' % s, lw=2)

        # Fix the axis to cover from the minimum to the 3 sigma line
        sub.set_ylim([self._current_minimum - delta_chi2[0],
                      self._current_minimum + (delta_chi2[-1] * 2)])

        plt.legend(loc=0, frameon=True)

        sub.set_xlabel(name1)
        sub.set_ylabel("-log( likelihood )")

        return fig

    def _plot_contours(self, name1, a, name2, b, cc):
        """
        Make a contour plot.

        :param name1: Name of the first parameter
        :param a: Grid for the first parameter (dimension N)
        :param name2: Name of the second parameter
        :param b: grid for the second parameter (dimension M)
        :param cc: N x M matrix containing the value of the log.likelihood for each point in the grid
        :return: figure containing the contour
        """

        # Check that we have something to plot

        delta = cc.max() - cc.min()

        if delta < 0.5:
            print("\n\nThe maximum difference in statistic is %s among all the points in the grid." % delta)
            print(" This is too small. Enlarge the search region to display a contour plot")

            return None

        # plot 1,2 and 3 sigma contours
        sigmas = [1, 2, 3]

        # Compute the corresponding probability. We do not
        # pre-compute them because we will introduce at
        # some point the possibility to personalize the
        # levels
        probabilities = []

        for s in sigmas:
            # One-sided probability
            # It is one-sided because we consider one side at the time
            # when computing the error

            probabilities.append(1 - (scipy.stats.norm.sf(s) * 2))

        # Compute the corresponding delta chisq. (chisq has 2 d.o.f.)
        delta_chi2 = scipy.stats.chi2.ppf(probabilities, 2) / 2.0  # two-sided

        # Boundaries for the colormap
        bounds = [self._current_minimum]
        # noinspection PyTypeChecker
        bounds.extend(self._current_minimum + delta_chi2)
        bounds.append(cc.max())

        # Define the color palette
        palette = cm.Pastel1
        palette.set_over('white')
        palette.set_under('white')
        palette.set_bad('white')

        fig = plt.figure()
        sub = fig.add_subplot(111)

        # Show the contours with square axis

        # NOTE: suppress the UnicodeWarning, which is due to a small problem in matplotlib

        with custom_warnings.catch_warnings():

            # Cause all warnings to always be triggered.
            custom_warnings.simplefilter("ignore", UnicodeWarning)

            im = sub.imshow(cc,
                            cmap=palette,
                            extent=[b.min(), b.max(), a.min(), a.max()],
                            aspect=float(b.max() - b.min()) / (a.max() - a.min()),
                            origin='lower',
                            norm=BoundaryNorm(bounds, 256),
                            interpolation='bicubic',
                            vmax=(self._current_minimum + delta_chi2).max())

        # Plot the color bar with the sigmas
        cb = fig.colorbar(im, boundaries=bounds[:-1])
        lbounds = [0]
        lbounds.extend(bounds[:-1])
        cb.set_ticks(lbounds)
        ll = ['']
        ll.extend(map(lambda x: r'%i $\sigma$' % x, sigmas))
        cb.set_ticklabels(ll)

        # Align the labels to the end of the color level
        for t in cb.ax.get_yticklabels():
            t.set_verticalalignment('baseline')

        # Draw the line contours
        sub.contour(b, a, cc, self._current_minimum + delta_chi2)

        # Set the axes labels

        sub.set_xlabel(name2)
        sub.set_ylabel(name1)

        return fig
