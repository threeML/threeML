import collections
import numpy
import scipy.optimize
import scipy.stats
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm

from threeML.io.rich_display import display
import numpy as np
import pandas as pd
import uncertainties
import astromodels.core.model


from threeML.minimizer import minimization
from threeML.exceptions import custom_exceptions
from threeML.io.table import Table
from threeML.parallel.parallel_client import ParallelClient
from threeML.config.config import threeML_config
from threeML.exceptions.custom_exceptions import custom_warnings, FitFailed
from threeML.config.config import threeML_config
from threeML.io.uncertainty_formatter import uncertainty_formatter
from threeML.analysis_results import MLEResults


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

        self._analysis_type = "mle"

        # Process optional keyword parameters
        self.verbose = verbose

        self._likelihood_model = likelihood_model  # type: astromodels.core.model.Model

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

        # This is to keep track of the number of calls to the likelihood
        # function
        self.ncalls = 0

        # Pre-defined minimizer is Minuit

        self.set_minimizer(threeML_config['mle']['default minimizer'],
                           threeML_config['mle']['default minimizer algorithm'],
                           threeML_config['mle']['default minimizer callback'])

        # Initial set of free parameters

        self._free_parameters = self._likelihood_model.free_parameters

        # Initially set the value of _current_minimum to None, it will be change by the fit() method

        self._current_minimum = None

        # Null setup for minimizer

        self._minimizer = None

        self._minimizer_callback = None

        self._analysis_results = None

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

    @property
    def analysis_type(self):
        return self._analysis_type

    def _update_free_parameters(self):

        """Update the dictionary of free parameters"""

        self._free_parameters = self._likelihood_model.free_parameters

    def fit(self, quiet=False, compute_covariance=True):
        """
        Perform a fit of the current likelihood model on the datasets

        :param quiet: If True, print the results (default), otherwise do not print anything
        :param compute_covariance:If True (default), compute and display the errors and the correlation matrix.
        :return: a dictionary with the results on the parameters, and the values of the likelihood at the minimum
                 for each dataset and the total one.
        """

        # Update the list of free parameters, to be safe against changes the user might do between
        # the creation of this class and the calling of this method

        self._update_free_parameters()

        # Check if we have free parameters, otherwise simply return the value of the log like
        if len(self._free_parameters) == 0:

            custom_warnings.warn("There is no free parameter in the current model", RuntimeWarning)

            # Generate an empty data frame, which will be returned

            fit_results = pd.DataFrame.from_dict({})

            # Store the "minimum", which is just the current value
            self._current_minimum = float(self.minus_log_like_profile([]))

        else:

            # Now check and fix if needed all the deltas of the parameters
            # to 20% of their value (otherwise the fit will be super-slow)

            for k, v in self._free_parameters.items():

                if abs(v.delta) < abs(v.value) * 0.2:

                    v.delta = abs(v.value) * 0.2

            # Instance the minimizer

            self._minimizer = self._get_minimizer(self.minus_log_like_profile,
                                                  self._free_parameters)

            # Perform the fit

            xs, log_likelihood_minimum = self._minimizer.minimize(compute_covar=compute_covariance)

            if log_likelihood_minimum == minimization.FIT_FAILED:

                raise FitFailed("The fit failed to converge.")

            # Store the current minimum for the -log likelihood

            self._current_minimum = float(log_likelihood_minimum)

        # Now collect the values for the likelihood for the various datasets

        # First restore best fit (to make sure we compute the likelihood at the right point)
        self._minimizer.restore_best_fit()

        # Fill the dictionary with the values of the -log likelihood (dataset by dataset)

        minus_log_likelihood_values = collections.OrderedDict()

        # Keep track of the total for a double check

        total = 0

        for dataset in self._data_list.values():

            ml = dataset.inner_fit() * (-1)

            minus_log_likelihood_values[dataset.get_name()] = ml

            total += ml

        assert total == self._current_minimum, "Current minimum stored after fit and current do not correspond!"

        # Now instance an analysis results class
        self._analysis_results = MLEResults(self.likelihood_model, self._minimizer.covariance_matrix,
                                            minus_log_likelihood_values)

        # Show the results

        if not quiet:

            self._analysis_results.display()

        return self._analysis_results.get_data_frame(), self._analysis_results.get_statistic_frame()

    @property
    def results(self):

        return self._analysis_results

    def get_errors(self, quiet=False):
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

        for k, v in self._free_parameters.items():

            # Format the value and the error with sensible significant
            # numbers

            neg_error = errors[k][0]

            pos_error = errors[k][1]

            pretty_string = uncertainty_formatter(v.value, v.value + neg_error, v.value + pos_error)

            data.append([k, pretty_string, v.unit])

            if len(k) > name_length:
                name_length = len(k)

        # Create and display the table

        table = Table(rows=data,
                      names=["Name", "Value", "Unit"],
                      dtype=('S%i' % name_length, str, str))

        if not quiet:

            display(table)

        return table.to_pandas()

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

        min1, max1 = self.likelihood_model[param_1].bounds

        if min1 is not None:

            assert param_1_minimum >= min1, "Requested low range for parameter %s (%s) " \
                                            "is below parameter minimum (%s)" % (param_1, param_1_minimum, min1)

        if max1 is not None:

            assert param_1_maximum <= max1, "Requested hi range for parameter %s (%s) " \
                                            "is above parameter maximum (%s)" % (param_1, param_1_maximum, max1)

        if param_2 is not None:

            min2, max2 = self.likelihood_model[param_2].bounds

            if min2 is not None:

                assert param_2_minimum >= min2, "Requested low range for parameter %s (%s) " \
                                                "is below parameter minimum (%s)" % (param_2, param_2_minimum, min2)

            if max2 is not None:

                assert param_2_maximum <= max2, "Requested hi range for parameter %s (%s) " \
                                                "is above parameter maximum (%s)" % (param_2, param_2_maximum, max2)

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
                                                      progress=True, **options)

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

            client.wait_watching_progress(amr, 10)

            # print progress

            # progress = ProgressBar(n_engines)
            #
            # # This loop will check from time to time the status of the computation, which is happening on
            # # different threads, and update the progress bar
            #
            # while not amr.ready():
            #     # Check and report the status of the computation every second
            #
            #     time.sleep(1 + np.random.uniform(0, 1))
            #
            #     # if (debug):
            #     #     stdouts = amr.stdout
            #     #
            #     #     # clear_output doesn't do much in terminal environments
            #     #     for stdout, stderr in zip(amr.stdout, amr.stderr):
            #     #         if stdout:
            #     #             print "%s" % (stdout[-1000:])
            #     #         if stderr:
            #     #             print "%s" % (stderr[-1000:])
            #     #     sys.stdout.flush()
            #
            #     progress.animate(amr.progress - 1)
            #
            # # Always display 100% at the end
            #
            # progress.animate(n_engines - 1)

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

            summed_log_likelihood += this_log_like

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

    def set_minimizer(self, minimizer, algorithm=None, callback=None):
        """
        Set the minimizer to be used, among those available. At the moment these are supported:

        * ROOT
        * MINUIT (which means iminuit, default)
        * PYOPT
        * PAGMO (http://esa.github.io/pygmo/documentation)

        :param minimizer: the name of the new minimizer.
        :param algorithm: (optional) for the PYOPT minimizer, specify the algorithm among those available. See a list at
        http://www.pyopt.org/reference/optimizers.html . For the PAGMO algorithm, this is an instance of an algorithm
        (see http://esa.github.io/pygmo/documentation/algorithms.html). For the other minimizers this is ignored.
        :param callback: (optional) a function which receives the minimizer instance as argument, which perform further
        setup before the minimizer is used for the fit. This is for example needed for the GRID minimizer to setup the
        grid
        :return: (none)
        """

        assert minimizer.upper() in minimization._minimizers, "Minimizer %s is not available on this system. Available " \
                                                              "minimizers: " \
                                                              "%s" % (minimizer,
                                                                      ",".join(minimization._minimizers.keys()))

        if minimizer.upper() == "MINUIT":

            self._minimizer_name = "MINUIT"
            self._minimizer_algorithm = None

        elif minimizer.upper() == "ROOT":

            self._minimizer_name = "ROOT"
            self._minimizer_algorithm = None

        elif minimizer.upper() == "PYOPT":

            assert algorithm is not None, "You have to provide an algorithm for the PYOPT minimizer. " \
                                          "See http://www.pyopt.org/reference/optimizers.html"

            self._minimizer_name = "PYOPT"

            assert algorithm.upper() in minimization._pyopt_algorithms, "Provided algorithm not available. " \
                                                                        "Available algorithms are: " \
                                                                        "%s" % minimization._pyopt_algorithms.keys()
            self._minimizer_algorithm = algorithm.upper()

        elif minimizer.upper() == "MULTINEST":

            self._minimizer_name = "MULTINEST"
            self._minimizer_algorithm = None

        elif minimizer.upper() == "GRID":

            self._minimizer_name = "GRID"
            self._minimizer_algorithm = None

        elif minimizer.upper() == "PAGMO":

            assert algorithm is not None, "You have to provide an instance of an algorithm for the PAGMO minimizer. " \
                                          "See http://esa.github.io/pygmo/documentation/algorithms.html"

            self._minimizer_name = "PAGMO"
            self._minimizer_algorithm = algorithm

        else:

            raise ValueError("Do not know minimizer %s. "
                             "Available minimizers are: %s" % (minimizer, ",".join(minimization._minimizers.keys())))

        # Get the type

        self._minimizer_type = minimization.get_minimizer(self._minimizer_name)

        # Set the callback (default is None)
        self._minimizer_callback = callback

    def _get_minimizer(self, *args, **kwargs):

        # Get an instance of the minimizer

        minimizer_instance = self._minimizer_type(*args, **kwargs)

        # Set up the algorithm, if appropriate

        if self._minimizer_algorithm:

            minimizer_instance.set_algorithm(self._minimizer_algorithm)

        # Call the callback if one is set

        if self._minimizer_callback is not None:

            self._minimizer_callback(minimizer_instance, self._likelihood_model)

        return minimizer_instance

    @property
    def minimizer_in_use(self):
        return self._minimizer_name, self._minimizer_algorithm, self._minimizer_callback

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

        sub.plot(a[~idx], cc[~idx], lw=2, color=threeML_config['mle']['profile color'])

        # Now plot the failed fits as "x"

        sub.plot(a[idx], [cc.min()] * a[idx].shape[0], 'x', c='red', markersize=2)

        # Decide colors
        colors = [threeML_config['mle']['profile level 1'],
                  threeML_config['mle']['profile level 2'],
                  threeML_config['mle']['profile level 3']]

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
        palette = plt.get_cmap(threeML_config['mle']['contour cmap'])  # cm.Pastel1
        palette.set_over(threeML_config['mle']['contour background'])
        palette.set_under(threeML_config['mle']['contour background'])
        palette.set_bad(threeML_config['mle']['contour background'])

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
        sub.contour(b, a, cc, self._current_minimum + delta_chi2,
                    colors=(threeML_config['mle']['contour level 1'], threeML_config['mle']['contour level 2'],
                            threeML_config['mle']['contour level 3']))

        # Set the axes labels

        sub.set_xlabel(name2)
        sub.set_ylabel(name1)

        return fig
