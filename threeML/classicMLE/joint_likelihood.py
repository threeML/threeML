from __future__ import division, print_function

import collections
import sys
from builtins import object, range, zip

import astromodels.core.model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats
from astromodels import Model, ModelAssertionViolation, clone_model
from past.utils import old_div

from threeML.analysis_results import MLEResults
from threeML.config.config import threeML_config
from threeML.data_list import DataList
from threeML.exceptions import custom_exceptions
from threeML.exceptions.custom_exceptions import FitFailed, custom_warnings
from threeML.io.logging import setup_logger
from threeML.io.results_table import ResultsTable
from threeML.io.table import Table
from threeML.minimizer import minimization
from threeML.parallel.parallel_client import ParallelClient
from threeML.utils.statistics.stats_tools import aic, bic

log = setup_logger(__name__)


class ReducingNumberOfThreads(Warning):
    pass


class ReducingNumberOfSteps(Warning):
    pass


class NotANumberInLikelihood(Warning):
    pass


class JointLikelihood(object):
    def __init__(
        self,
        likelihood_model: Model,
        data_list: DataList,
        verbose: bool = False,
        record: bool = True,
    ):
        """
        Implement a joint likelihood analysis.

        :param likelihood_model: the model for the likelihood analysis
        :param data_list: the list of data sets (plugin instances) to be used in this analysis
        :param verbose: (True or False) print every step in the -log likelihood minimization
        :param record: it records every call to the log likelihood function during minimization. The recorded values
        can be retrieved as a pandas DataFrame using the .fit_trace property
        :return:
        """

        log.debug("creating new MLE analysis")

        self._analysis_type: str = "mle"

        # Process optional keyword parameters
        self.verbose: bool = verbose

        self._likelihood_model: Model = likelihood_model

        self._data_list: DataList = data_list

        self._assign_model_to_data(self._likelihood_model)

        # This is to keep track of the number of calls to the likelihood
        # function
        self._record: bool = bool(record)
        self._ncalls: int = 0
        self._record_calls: dict = {}

        # Pre-defined minimizer
        default_minimizer = minimization.LocalMinimization(
            threeML_config["mle"]["default minimizer"]
        )

        if threeML_config["mle"]["default minimizer algorithm"] is not None:

            default_minimizer.set_algorithm(
                threeML_config["mle"]["default minimizer algorithm"]
            )

        self.set_minimizer(default_minimizer)

        # Initial set of free parameters

        self._free_parameters = self._likelihood_model.free_parameters

        # Initially set the value of _current_minimum to None, it will be change by the fit() method

        self._current_minimum = None

        # Null setup for minimizer

        self._minimizer = None

        self._minimizer_callback = None

        self._analysis_results = None

    def _assign_model_to_data(self, model) -> None:

        log.debug("REGISTERING MODEL")

        for dataset in list(self._data_list.values()):

            dataset.set_model(model)

            # Now get the nuisance parameters from the data and add them to the model
            # NOTE: it is important that this is *after* the setting of the model, as some
            # plugins might need to adjust the number of nuisance parameters depending on the
            # likelihood model

            for parameter_name, parameter in dataset.nuisance_parameters.items():

                # Enforce that the nuisance parameter contains the instance name, because otherwise multiple instance
                # of the same plugin will overwrite each other's nuisance parameters

                assert dataset.name in parameter_name, (
                    "This is a bug of the plugin for %s: nuisance parameters "
                    "must contain the instance name" % type(dataset)
                )

                self._likelihood_model.add_external_parameter(parameter)

        log.debug("MODEL REGISTERED!")

    @property
    def likelihood_model(self) -> Model:
        """
        :return: likelihood model for this analysis
        """

        return self._likelihood_model

    @property
    def data_list(self) -> DataList:
        """
        :return: data list for this analysis
        """

        return self._data_list

    @property
    def current_minimum(self) -> float:
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

            raise RuntimeError(
                "You need to run a fit before accessing the covariance matrix"
            )

    @property
    def correlation_matrix(self):
        """
        :return: correlation matrix from the last fit
        """

        try:

            return self._minimizer.correlation_matrix

        except AttributeError:

            raise RuntimeError(
                "You need to run a fit before accessing the correlation matrix"
            )

    @property
    def analysis_type(self) -> str:
        return self._analysis_type

    def _update_free_parameters(self):
        """Update the dictionary of free parameters"""

        self._free_parameters = self._likelihood_model.free_parameters

    def fit(
        self,
        quiet: bool = False,
        compute_covariance: bool = True,
        n_samples: int = 5000,
    ):
        """
        Perform a fit of the current likelihood model on the datasets

        :param quiet: If True, print the results (default), otherwise do not print anything
        :param compute_covariance:If True (default), compute and display the errors and the correlation matrix.
        :return: a dictionary with the results on the parameters, and the values of the likelihood at the minimum
                 for each dataset and the total one.
        """

        # Update the list of free parameters, to be safe against changes the user might do between
        # the creation of this class and the calling of this method

        log.debug("beginning the fit!")
        self._update_free_parameters()

        # Empty the call recorder
        self._record_calls = {}
        self._ncalls = 0

        # Check if we have free parameters, otherwise simply return the value of the log like
        if len(self._free_parameters) == 0:

            log.warning("There is no free parameter in the current model")

            # Create the minimizer anyway because it will be needed by the following code

            self._minimizer = self._get_minimizer(
                self.minus_log_like_profile, self._free_parameters
            )

            # Store the "minimum", which is just the current value
            self._current_minimum = float(self.minus_log_like_profile())

        else:

            # Instance the minimizer

            # If we have a global minimizer, use that first (with no covariance)
            if isinstance(self._minimizer_type, minimization.GlobalMinimization):

                # Do global minimization first
                log.debug(f"starting global optimization")

                if quiet:

                    verbosity = 0

                else:

                    verbosity = 1

                global_minimizer = self._get_minimizer(
                    self.minus_log_like_profile, self._free_parameters, verbosity=verbosity
                )

                xs, global_log_likelihood_minimum = global_minimizer.minimize(
                    compute_covar=False
                )

                # Gather global results
                paths = []
                values = []
                errors = []
                units = []

                for par in list(self._free_parameters.values()):

                    paths.append(par.path)
                    values.append(par.value)
                    errors.append(0)
                    units.append(par.unit)

                global_results = ResultsTable(
                    paths, values, errors, errors, units)

                if not quiet:

                    log.info(
                        "\n\nResults after global minimizer (before secondary optimization):"
                    )

                    global_results.display()

                    log.info(
                        "\nTotal log-likelihood minimum: %.3f\n"
                        % global_log_likelihood_minimum
                    )

                # Now set up secondary minimizer
                self._minimizer = self._minimizer_type.get_second_minimization_instance(
                    self.minus_log_like_profile, self._free_parameters
                )

            else:

                # Only local minimization to be performed

                log.debug("starting local optimization")

                self._minimizer = self._get_minimizer(
                    self.minus_log_like_profile, self._free_parameters
                )

            # Perform the fit, but first flush stdout (so if we have verbose=True the messages there will follow
            # what is already in the buffer)
            sys.stdout.flush()

            xs, log_likelihood_minimum = self._minimizer.minimize(
                compute_covar=compute_covariance
            )

            if log_likelihood_minimum == minimization.FIT_FAILED:
                log.error("The fit failed to converge.")
                raise FitFailed()

            # Store the current minimum for the -log likelihood

            self._current_minimum = float(log_likelihood_minimum)

            # First restore best fit (to make sure we compute the likelihood at the right point in the following)
            self._minimizer.restore_best_fit()

        # Now collect the values for the likelihood for the various datasets

        # Fill the dictionary with the values of the -log likelihood (dataset by dataset)

        minus_log_likelihood_values = collections.OrderedDict()

        # Keep track of the total for a double check

        total = 0

        # sum up the total number of data points

        total_number_of_data_points = 0

        for dataset in list(self._data_list.values()):

            ml = dataset.inner_fit() * (-1)

            minus_log_likelihood_values[dataset.name] = ml

            total += ml

            total_number_of_data_points += dataset.get_number_of_data_points()

        assert (
            total == self._current_minimum
        ), "Current minimum stored after fit and current do not correspond!"

        # compute additional statistics measures

        statistical_measures = collections.OrderedDict()

        # for MLE we can only compute the AIC and BIC as they
        # are point estimates

        statistical_measures["AIC"] = aic(
            -total, len(self._free_parameters), total_number_of_data_points
        )
        statistical_measures["BIC"] = bic(
            -total, len(self._free_parameters), total_number_of_data_points
        )

        # Now instance an analysis results class
        self._analysis_results = MLEResults(
            self.likelihood_model,
            self._minimizer.covariance_matrix,
            minus_log_likelihood_values,
            statistical_measures=statistical_measures,
            n_samples=n_samples,
        )

        # Show the results

        if not quiet:

            self._analysis_results.display()

        return (
            self._analysis_results.get_data_frame(),
            self._analysis_results.get_statistic_frame(),
        )

    @property
    def results(self) -> MLEResults:

        return self._analysis_results

    def get_errors(self, quiet=False):
        """
        Compute the errors on the parameters using the profile likelihood method.

        :return: a dictionary containing the asymmetric errors for each parameter.
        """

        # Check that the user performed a fit first

        assert (
            self._current_minimum is not None
        ), "You have to run the .fit method before calling errors."

        errors = self._minimizer.get_errors()

        # Set the parameters back to the best fit value
        self.restore_best_fit()

        # Print a table with the errors

        parameter_names = list(self._free_parameters.keys())
        best_fit_values = [x.value for x in list(
            self._free_parameters.values())]
        negative_errors = [errors[k][0] for k in parameter_names]
        positive_errors = [errors[k][1] for k in parameter_names]
        units = [par.unit for par in list(self._free_parameters.values())]

        results_table = ResultsTable(
            parameter_names, best_fit_values, negative_errors, positive_errors, units
        )

        if not quiet:

            results_table.display()

        return results_table.frame

    def get_contours(
        self,
        param_1,
        param_1_minimum,
        param_1_maximum,
        param_1_n_steps,
        param_2=None,
        param_2_minimum=None,
        param_2_maximum=None,
        param_2_n_steps=None,
        progress=True,
        **options,
    ):
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

        if hasattr(param_1, "value"):

            # Substitute with the name
            param_1 = param_1.path

        if hasattr(param_2, "value"):

            param_2 = param_2.path

        # Check that the parameters exist
        assert param_1 in self._likelihood_model.free_parameters, (
            "Parameter %s is not a free parameters of the " "current model" % param_1
        )

        if param_2 is not None:
            assert param_2 in self._likelihood_model.free_parameters, (
                "Parameter %s is not a free parameters of the "
                "current model" % param_2
            )

        # Check that we have a valid fit

        assert (
            self._current_minimum is not None
        ), "You have to run the .fit method before calling get_contours."

        # Then restore the best fit

        self._minimizer.restore_best_fit()

        # Check minimal assumptions about the procedure

        assert not (
            param_1 == param_2), "You have to specify two different parameters"

        assert (
            param_1_minimum < param_1_maximum
        ), "Minimum larger than maximum for parameter 1"

        min1, max1 = self.likelihood_model[param_1].bounds

        if min1 is not None:

            assert param_1_minimum >= min1, (
                "Requested low range for parameter %s (%s) "
                "is below parameter minimum (%s)" % (
                    param_1, param_1_minimum, min1)
            )

        if max1 is not None:

            assert param_1_maximum <= max1, (
                "Requested hi range for parameter %s (%s) "
                "is above parameter maximum (%s)" % (
                    param_1, param_1_maximum, max1)
            )

        if param_2 is not None:

            min2, max2 = self.likelihood_model[param_2].bounds

            if min2 is not None:

                assert param_2_minimum >= min2, (
                    "Requested low range for parameter %s (%s) "
                    "is below parameter minimum (%s)" % (
                        param_2, param_2_minimum, min2)
                )

            if max2 is not None:

                assert param_2_maximum <= max2, (
                    "Requested hi range for parameter %s (%s) "
                    "is above parameter maximum (%s)" % (
                        param_2, param_2_maximum, max2)
                )

        # Check whether we are parallelizing or not

        if not threeML_config["parallel"]["use-parallel"]:

            a, b, cc = self.minimizer.contours(
                param_1,
                param_1_minimum,
                param_1_maximum,
                param_1_n_steps,
                param_2,
                param_2_minimum,
                param_2_maximum,
                param_2_n_steps,
                progress,
                **options,
            )

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

                log.warning(
                    "The number of engines is larger than the number of steps. Using only %s engines."
                    % n_engines,
                )

            # Check if the number of steps is divisible by the number
            # of threads, otherwise issue a warning and make it so

            if float(param_1_n_steps) % n_engines != 0:
                # Set the number of steps to an integer multiple of the engines
                # (note that // is the floor division, also called integer division)

                param_1_n_steps = (param_1_n_steps // n_engines) * n_engines

                log.warning(
                    "Number of steps is not a multiple of the number of threads. Reducing steps to %s"
                    % param_1_n_steps,
                )

            # Compute the number of splits, i.e., how many lines in the grid for each engine.
            # (note that this is guaranteed to be an integer number after the previous checks)

            p1_split_steps = param_1_n_steps // n_engines

            # Prepare arrays for results

            if param_2 is None:

                # One array
                pcc = np.zeros(param_1_n_steps)

                pa = np.linspace(
                    param_1_minimum, param_1_maximum, param_1_n_steps)
                pb = None

            else:

                pcc = np.zeros((param_1_n_steps, param_2_n_steps))

                # Prepare the two axes of the parameter space
                pa = np.linspace(
                    param_1_minimum, param_1_maximum, param_1_n_steps)
                pb = np.linspace(
                    param_2_minimum, param_2_maximum, param_2_n_steps)

            # Define the parallel worker which will go through the computation

            # NOTE: I only divide
            # on the first parameter axis so that the different
            # threads are more or less well mixed for points close and
            # far from the best fit

            def worker(start_index):

                # Re-create the minimizer

                backup_freeParameters = [
                    x.value
                    for x in list(self._likelihood_model.free_parameters.values())
                ]

                this_minimizer = self._get_minimizer(
                    self.minus_log_like_profile, self._free_parameters
                )

                this_p1min = pa[start_index * p1_split_steps]
                this_p1max = pa[(start_index + 1) * p1_split_steps - 1]

                # print("From %s to %s" % (this_p1min, this_p1max))

                aa, bb, ccc = this_minimizer.contours(
                    param_1,
                    this_p1min,
                    this_p1max,
                    p1_split_steps,
                    param_2,
                    param_2_minimum,
                    param_2_maximum,
                    param_2_n_steps,
                    progress=True,
                    **options,
                )

                # Restore best fit values

                for val, par in zip(
                    backup_freeParameters,
                    list(self._likelihood_model.free_parameters.values()),
                ):

                    par.value = val

                return ccc

            # Now re-assemble the vector of results taking the different parts from the engines

            all_results = client.execute_with_progress_bar(
                worker, list(range(n_engines)), chunk_size=1
            )

            for i, these_results in enumerate(all_results):

                if param_2 is None:

                    pcc[i * p1_split_steps: (i + 1) * p1_split_steps] = these_results[
                        :, 0
                    ]

                else:

                    pcc[
                        i * p1_split_steps: (i + 1) * p1_split_steps, :
                    ] = these_results

            # Give the results the names that the following code expect. These are kept separate for debugging
            # purposes

            cc = pcc
            a = pa
            b = pb

        # Here we have done the computation, in parallel computation or not. Let's make the plot
        # with the contour

        if param_2 is not None:

            # 2d contour

            fig = self._plot_contours(
                "%s" % (param_1), a, "%s" % (param_2,), b, cc)

        else:

            # 1d contour (i.e., a profile)

            fig = self._plot_profile("%s" % (param_1), a, cc)

        # Check if we found a better minimum. This shouldn't happen, but in case of very difficult fit
        # it might.

        if self._current_minimum - cc.min() > 0.1:

            if param_2 is not None:

                idx = cc.argmin()

                aidx, bidx = np.unravel_index(idx, cc.shape)

                print(
                    "\nFound a better minimum: %s with %s = %s and %s = %s. Run again your fit starting from here."
                    % (cc.min(), param_1, a[aidx], param_2, b[bidx])
                )

            else:

                idx = cc.argmin()

                print(
                    "Found a better minimum: %s with %s = %s. Run again your fit starting from here."
                    % (cc.min(), param_1, a[idx])
                )

        return a, b, cc, fig

    def plot_all_contours(self, nsteps_1d, nsteps_2d=0, n_sigma=5, log_norm=True):

        figs = []
        names = []

        res = self.get_errors(False)

        if nsteps_1d >= 0:
            for param in self._likelihood_model.free_parameters:

                center = res["value"][param]
                do_log = (False,)
                lower = center + res["negative_error"][param] * n_sigma
                upper = center + res["positive_error"][param] * n_sigma

                if (
                    log_norm
                    and self._likelihood_model.free_parameters[param].is_normalization
                ):
                    do_log = (True,)
                    lower = (
                        center
                        * (1.0 + old_div(res["negative_error"][param], center))
                        ** n_sigma
                    )
                    upper = (
                        center
                        * (1.0 + old_div(res["positive_error"][param], center))
                        ** n_sigma
                    )

                lower = max(self.likelihood_model[param].bounds[0], lower)
                upper = min(self.likelihood_model[param].bounds[1], upper)

                # print param, lower, center, upper

                try:
                    a, b, cc, fig = self.get_contours(
                        param, lower, upper, nsteps_1d, log=do_log
                    )
                    figs.append(fig)
                    names.append(param)
                except Exception as e:
                    print(e)

        if nsteps_2d >= 0:

            for param_1 in self._likelihood_model.free_parameters:

                do_log = (False, False)
                center_1 = res["value"][param_1]
                lower_1 = center_1 + res["negative_error"][param_1] * n_sigma
                upper_1 = center_1 + res["positive_error"][param_1] * n_sigma

                if (
                    log_norm
                    and self._likelihood_model.free_parameters[param_1].is_normalization
                ):
                    do_log = (True, False)
                    lower_1 = (
                        center_1
                        * (1.0 + old_div(res["negative_error"][param_1], center_1))
                        ** n_sigma
                    )
                    upper_1 = (
                        center_1
                        * (1.0 + old_div(res["positive_error"][param_1], center_1))
                        ** n_sigma
                    )

                lower_1 = max(
                    self.likelihood_model[param_1].bounds[0], lower_1)
                upper_1 = min(
                    self.likelihood_model[param_1].bounds[1], upper_1)

                for param_2 in self._likelihood_model.free_parameters:

                    if param_2 <= param_1:
                        continue

                    center_2 = res["value"][param_2]
                    lower_2 = center_2 + \
                        res["negative_error"][param_2] * n_sigma
                    upper_2 = center_2 + \
                        res["positive_error"][param_2] * n_sigma

                    if (
                        log_norm
                        and self._likelihood_model.free_parameters[
                            param_2
                        ].is_normalization
                    ):
                        do_log = (do_log[0], True)
                        lower_2 = (
                            center_2
                            * (1.0 + old_div(res["negative_error"][param_2], center_2))
                            ** n_sigma
                        )
                        upper_2 = (
                            center_2
                            * (1.0 + old_div(res["positive_error"][param_2], center_2))
                            ** n_sigma
                        )

                    lower_2 = max(
                        self.likelihood_model[param_2].bounds[0], lower_2)
                    upper_2 = min(
                        self.likelihood_model[param_2].bounds[1], upper_2)

                    try:
                        a, b, cc, fig = self.get_contours(
                            param_1,
                            lower_1,
                            upper_1,
                            nsteps_2d,
                            param_2,
                            lower_2,
                            upper_2,
                            nsteps_2d,
                            log=do_log,
                        )
                        figs.append(fig)
                        names.append("%s-%s" % (param_1, param_2))
                    except Exception as e:
                        print(e)
        return figs, names

    def minus_log_like_profile(self, *trial_values):
        """
        Return the minus log likelihood for a given set of trial values

        :param trial_values: the trial values. Must be in the same number as the free parameters in the model
        :return: minus log likelihood
        """

        # Keep track of the number of calls
        self._ncalls += 1

        # Transform the trial values in a numpy array

        trial_values = np.array(trial_values)

        # Check that there are no nans within the trial values

        # This is the fastest way to check for any nan
        # (try other methods if you don't believe me)

        if not np.isfinite(np.dot(trial_values, trial_values.T)):
            # There are nans, something weird is going on. Return FIT_FAILED so the engine
            # stays away from this (or fail)

            return minimization.FIT_FAILED

        # Assign the new values to the parameters

        for i, parameter in enumerate(self._free_parameters.values()):

            # Use the internal representation (see the Parameter class)

            parameter._set_internal_value(trial_values[i])

        # Now profile out nuisance parameters and compute the new value
        # for the likelihood

        summed_log_likelihood = 0

        for dataset in list(self._data_list.values()):

            try:

                this_log_like = dataset.inner_fit()

            except ModelAssertionViolation:

                # This is a zone of the parameter space which is not allowed. Return
                # a big number for the likelihood so that the fit engine will avoid it

                log.warning(
                    "Fitting engine in forbidden space: %s" % (trial_values,),
                )

                return minimization.FIT_FAILED

            except:

                # Do not intercept other errors

                raise

            summed_log_likelihood += this_log_like

        # Check that the global like is not NaN
        # I use this weird check because it is not guaranteed that the plugins return np.nan,
        # especially if they are written in something other than python

        if "%s" % summed_log_likelihood == "nan":
            log.warning(
                "These parameters returned a logLike = Nan: %s" % (trial_values,),
            )

            return minimization.FIT_FAILED

        if self.verbose:
            sys.stderr.write(
                "trial values: %s -> logL = %.3f\n"
                % (",".join(["%.5g" % x for x in trial_values]), summed_log_likelihood)
            )

        # Record this call
        if self._record:

            self._record_calls[tuple(trial_values)] = summed_log_likelihood

        # Return the minus log likelihood

        return summed_log_likelihood * (-1)

    @property
    def fit_trace(self):
        return pd.DataFrame(self._record_calls)

    def set_minimizer(self, minimizer):
        """
        Set the minimizer to be used, among those available.

        :param minimizer: the name of the new minimizer or an instance of a LocalMinimization or a GlobalMinimization
        class. Using the latter two classes allows for more choices and a better control of the details of the
        minimization, like the choice of algorithms (if supported by the used minimizer)
        :return: (none)
        """

        if isinstance(minimizer, minimization._Minimization):

            self._minimizer_type = minimizer

        else:

            assert minimizer.upper() in minimization._minimizers, (
                "Minimizer %s is not available on this system. "
                "Available minimizers: %s"
                % (minimizer, ",".join(list(minimization._minimizers.keys())))
            )

            # The string can only specify a local minimization. This will return an error if that is not the case.
            # In order to setup global optimization the user needs to use the GlobalMinimization factory directly

            self._minimizer_type = minimization.LocalMinimization(minimizer)

        log.info(f"set the minimizer to {minimizer}")

    def _get_minimizer(self, *args, **kwargs):

        # Get an instance of the minimizer

        minimizer_instance = self._minimizer_type.get_instance(*args, **kwargs)

        # Call the callback if one is set

        if self._minimizer_callback is not None:

            self._minimizer_callback(
                minimizer_instance, self._likelihood_model)

        return minimizer_instance

    @property
    def minimizer_in_use(self):

        return self._minimizer_type

    def restore_best_fit(self):
        """
        Restore the model to its best fit

        :return: (none)
        """

        if self._minimizer:

            self._minimizer.restore_best_fit()

        else:

            log.warning("Cannot restore best fit, since fit has not been executed.")

    def _get_table_of_parameters(self, parameters):

        data = []
        max_length_of_name = 0

        for k, v in parameters.items():

            current_name = "%s_of_%s" % (k[1], k[0])

            data.append([current_name, "%s" % v.value, v.unit])

            if len(v.name) > max_length_of_name:
                max_length_of_name = len(current_name)

        table = Table(
            rows=data,
            names=["Name", "Value", "Unit"],
            dtype=("S%i" % max_length_of_name, str, "S15"),
        )

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
        delta_chi2 = np.array(
            scipy.stats.chi2.ppf(probabilities, 1) / 2.0
        )  # two-sided!

        fig = plt.figure()
        sub = fig.add_subplot(111)

        # Neutralize values of the loglike too high
        # (fit failed)
        idx = cc == minimization.FIT_FAILED

        sub.plot(a[~idx], cc[~idx], lw=2,
                 color=threeML_config["mle"]["profile color"])

        # Now plot the failed fits as "x"

        sub.plot(a[idx], [cc.min()] * a[idx].shape[0],
                 "x", c="red", markersize=2)

        # Decide colors
        colors = [
            threeML_config["mle"]["profile level 1"],
            threeML_config["mle"]["profile level 2"],
            threeML_config["mle"]["profile level 3"],
        ]

        for s, d, c in zip(sigmas, delta_chi2, colors):
            sub.axhline(
                self._current_minimum + d,
                linestyle="--",
                color=c,
                label=r"%s $\sigma$" % s,
                lw=2,
            )

        # Fix the axis to cover from the minimum to the 3 sigma line
        sub.set_ylim(
            [
                self._current_minimum - delta_chi2[0],
                self._current_minimum + (delta_chi2[-1] * 2),
            ]
        )

        plt.legend(loc=0, frameon=True)

        sub.set_xlabel(name1)
        sub.set_ylabel("-log( likelihood )")

        plt.tight_layout()

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
            print(
                "\n\nThe maximum difference in statistic is %s among all the points in the grid."
                % delta
            )
            print(
                " This is too small. Enlarge the search region to display a contour plot"
            )

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
        palette = plt.get_cmap(
            threeML_config["mle"]["contour cmap"])  # cm.Pastel1
        palette.set_over(threeML_config["mle"]["contour background"])
        palette.set_under(threeML_config["mle"]["contour background"])
        palette.set_bad(threeML_config["mle"]["contour background"])

        fig = plt.figure()
        sub = fig.add_subplot(111)

        # Draw the line contours
        sub.contour(
            b,
            a,
            cc,
            self._current_minimum + delta_chi2,
            colors=(
                threeML_config["mle"]["contour level 1"],
                threeML_config["mle"]["contour level 2"],
                threeML_config["mle"]["contour level 3"],
            ),
        )

        # Set the axes labels

        sub.set_xlabel(name2)
        sub.set_ylabel(name1)

        plt.tight_layout()

        return fig

    def compute_TS(self, source_name, alt_hyp_mlike_df):
        """
        Computes the Likelihood Ratio Test statistic (TS) for the provided source

        :param source_name: name for the source
        :param alt_hyp_mlike_df: likelihood dataframe (it is the second output of the .fit() method)
        :return: a DataFrame containing the null hypothesis and the alternative hypothesis -log(likelihood) values and
        the value for TS for the source for each loaded dataset
        """

        assert source_name in self._likelihood_model, (
            "Source %s is not in the current model" % source_name
        )

        # Clone model
        model_clone = clone_model(self._likelihood_model)

        # Remove this source from the model
        _ = model_clone.remove_source(source_name)

        # Fit
        another_jl = JointLikelihood(model_clone, self._data_list)

        # Use the same minimizer as the parent object
        another_jl.set_minimizer(self.minimizer_in_use)

        # We do not need the covariance matrix, just the likelihood value
        _, null_hyp_mlike_df = another_jl.fit(
            quiet=True, compute_covariance=False, n_samples=1
        )

        # Compute TS for all datasets
        TSs = []
        alt_hyp_mlikes = []
        null_hyp_mlikes = []

        for dataset in list(self._data_list.values()):

            this_name = dataset.name

            null_hyp_mlike = null_hyp_mlike_df.loc[this_name,
                                                   "-log(likelihood)"]
            alt_hyp_mlike = alt_hyp_mlike_df.loc[this_name, "-log(likelihood)"]

            this_TS = 2 * (null_hyp_mlike - alt_hyp_mlike)

            TSs.append(this_TS)
            alt_hyp_mlikes.append(alt_hyp_mlike)
            null_hyp_mlikes.append(null_hyp_mlike)

        TS_df = pd.DataFrame(index=list(self._data_list.keys()))

        TS_df["Null hyp."] = null_hyp_mlikes
        TS_df["Alt. hyp."] = alt_hyp_mlikes
        TS_df["TS"] = TSs

        # Reassign the original likelihood model to the datasets
        self._assign_model_to_data(self._likelihood_model)

        return TS_df
