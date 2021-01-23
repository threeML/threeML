from __future__ import division

import collections
import math
from builtins import object, range, str, zip

import numpy as np
import pandas as pd
import scipy.optimize
from past.utils import old_div
from threeML.utils.progress_bar import tqdm

from threeML.exceptions.custom_exceptions import custom_warnings
from threeML.io.logging import setup_logger
from threeML.config.config import threeML_config
from threeML.utils.differentiation import ParameterOnBoundary, get_hessian

# Set the warnings to be issued always for this module

custom_warnings.simplefilter("always", RuntimeWarning)

log = setup_logger(__name__)

# Special constants
FIT_FAILED = 1e12


# Define a bunch of custom exceptions relevant for what is being accomplished here


class CannotComputeCovariance(RuntimeWarning):
    pass


class CannotComputeErrors(RuntimeWarning):
    pass


class ParameterIsNotFree(Exception):
    pass


class FitFailed(Exception):
    pass


class MinimizerNotAvailable(Exception):
    pass


class BetterMinimumDuringProfiling(RuntimeWarning):
    pass


# This will contain the available minimizers

_minimizers = {}


def get_minimizer(minimizer_type):
    """
    Return the requested minimizer *class* (not instance)

    :param minimizer_type: MINUIT, ROOT, PYOPT...
    :return: the class (i.e., the type) for the requested minimizer
    """

    try:

        return _minimizers[minimizer_type.upper()]

    except KeyError:

        log.error("Minimizer %s is not available on your system" %
                  minimizer_type)

        raise MinimizerNotAvailable()


class FunctionWrapper(object):
    def __init__(self, function, all_parameters, fixed_parameters):
        """

        :param function:
        :param all_parameters:
        :param fixed_parameters: list of fixed parameters
        """
        self._function = function

        self._all_parameters = all_parameters

        self._fixed_parameters_values = np.zeros(len(fixed_parameters))
        self._fixed_parameters_names = fixed_parameters

        self._indexes_of_fixed_par = np.zeros(len(self._all_parameters), bool)

        for i, parameter_name in enumerate(self._fixed_parameters_names):

            this_index = list(self._all_parameters.keys()
                              ).index(parameter_name)

            self._indexes_of_fixed_par[this_index] = True

        self._all_values = np.zeros(len(self._all_parameters))

    def set_fixed_values(self, new_fixed_values):

        # Note that this will receive the fixed values in internal reference (after the transformations, if any)

        # A use [:] so there is an implicit check on the right size of new_fixed_values

        self._fixed_parameters_values[:] = new_fixed_values

    def __call__(self, *trial_values):

        # Note that this function will receive the trial values in internal reference (after the transformations,
        # if any)

        self._all_values[self._indexes_of_fixed_par] = self._fixed_parameters_values
        self._all_values[~self._indexes_of_fixed_par] = trial_values

        return self._function(*self._all_values)


class ProfileLikelihood(object):
    def __init__(self, minimizer_instance, fixed_parameters):

        self._fixed_parameters = fixed_parameters

        assert (
            len(self._fixed_parameters) <= 2
        ), "Can handle only one or two fixed parameters"

        # Get some info from the original minimizer

        self._function = minimizer_instance.function

        # Note that here we have to use the original parameters (not the internal parameters)

        self._all_parameters = minimizer_instance.parameters

        # Create a copy of the dictionary of parameters

        free_parameters = collections.OrderedDict(self._all_parameters)

        # Remove the fixed ones

        for parameter_name in fixed_parameters:

            free_parameters.pop(parameter_name)

        # Now compute how many free parameters we have

        self._n_free_parameters = len(free_parameters)

        if self._n_free_parameters > 0:

            self._wrapper = FunctionWrapper(
                self._function, self._all_parameters, self._fixed_parameters
            )

            # Create a copy of the optimizer with the new parameters (i.e., one or two
            # parameters fixed to their current values)

            self._optimizer = type(minimizer_instance)(
                self._wrapper, free_parameters, verbosity=0
            )

            if minimizer_instance.algorithm_name is not None:

                self._optimizer.set_algorithm(
                    minimizer_instance.algorithm_name)

        else:

            # Special case when there are no free parameters after fixing the requested ones
            # There is no profiling necessary here

            self._wrapper = None
            self._optimizer = None

    def _transform_steps(self, parameter_name, steps):
        """
        If the parameter has a transformation, use it for the steps and return the transformed steps

        :return: transformed steps
        """

        if self._all_parameters[parameter_name].has_transformation():

            new_steps = self._all_parameters[parameter_name].transformation.forward(
                steps
            )

            return new_steps

        else:

            # Nothing to do

            return steps

    def step(self, steps1, steps2=None):

        if steps2 is not None:

            assert (
                len(self._fixed_parameters) == 2
            ), "Cannot step in 2d if you fix only one parameter"

            # Find out if the user is giving flipped steps (i.e. param_1 is after param_2 in the
            # parameters dictionary)

            param_1_name = self._fixed_parameters[0]
            param_1_idx = list(self._all_parameters.keys()).index(param_1_name)

            param_2_name = self._fixed_parameters[1]
            param_2_idx = list(self._all_parameters.keys()).index(param_2_name)

            # Fix steps if needed
            steps1 = self._transform_steps(param_1_name, steps1)

            if steps2 is not None:

                steps2 = self._transform_steps(param_2_name, steps2)

            if param_1_idx > param_2_idx:

                # Switch steps

                swap = steps1
                steps1 = steps2
                steps2 = swap

                results = self._step2d(steps1, steps2).T

            else:

                results = self._step2d(steps1, steps2)

            return results

        else:

            assert (
                len(self._fixed_parameters) == 1
            ), "You cannot step in 1d if you fix 2 parameters"

            param_1_name = self._fixed_parameters[0]

            # Fix steps if needed.
            steps1 = self._transform_steps(param_1_name, steps1)

            return self._step1d(steps1)

    def __call__(self, values):

        self._wrapper.set_fixed_values(values)

        _, this_log_like = self._optimizer.minimize(compute_covar=False)

        return this_log_like

    def _step1d(self, steps1):

        log_likes = np.zeros_like(steps1)

        

        for i, step in enumerate(tqdm(steps1, desc="Profiling likelihood")):

            if self._n_free_parameters > 0:

                # Profile out the free parameters

                self._wrapper.set_fixed_values(step)

                _, this_log_like = self._optimizer.minimize(
                    compute_covar=False)

            else:

                # No free parameters, just compute the likelihood

                this_log_like = self._function(step)

            log_likes[i] = this_log_like

            

        return log_likes

    def _step2d(self, steps1, steps2):

        log_likes = np.zeros((len(steps1), len(steps2)))

        if threeML_config["interface"]["show_progress_bars"]:
        
            p = tqdm(total=len(steps1) * len(steps2), desc="Profiling likelihood")

        for i, step1 in enumerate(steps1):

            for j, step2 in enumerate(steps2):

                if self._n_free_parameters > 0:

                    # Profile out the free parameters

                    self._wrapper.set_fixed_values([step1, step2])

                    try:

                        _, this_log_like = self._optimizer.minimize(
                            compute_covar=False
                        )

                    except FitFailed:

                        # If the user is stepping too far it might be that the fit fails. It is usually not a
                        # problem

                        this_log_like = np.nan

                else:

                    # No free parameters, just compute the likelihood

                    this_log_like = self._function(step1, step2)

                log_likes[i, j] = this_log_like

                if threeML_config["interface"]["show_progress_bars"]:
                    p.update(1)

        return log_likes


# This classes are used directly by the user to have better control on the minimizers.
# They are actually factories


class _Minimization(object):
    def __init__(self, minimizer_type):

        self._minimizer_type = get_minimizer(minimizer_type=minimizer_type)

        self._algorithm = None
        self._setup_dict = {}

    def setup(self, **setup_dict):

        valid_setup_keys = self._minimizer_type.valid_setup_keys

        # Check that the setup has been specified well
        for key in list(setup_dict.keys()):

            assert key in valid_setup_keys, (
                "%s is not a valid setup parameter for this minimizer" % key
            )

        self._setup_dict = setup_dict

    def set_algorithm(self, algorithm):

        # Note that algorithm might be None

        self._algorithm = algorithm


class LocalMinimization(_Minimization):
    def __init__(self, minimizer_type):

        super(LocalMinimization, self).__init__(minimizer_type)

        assert issubclass(self._minimizer_type, LocalMinimizer), (
            "Minimizer %s is not a local minimizer" % minimizer_type
        )

    def get_instance(self, *args, **kwargs):

        instance = self._minimizer_type(*args, **kwargs)

        if self._algorithm is not None:

            instance.set_algorithm(self._algorithm)

        # Set up the minimizer
        instance._setup(self._setup_dict)

        return instance


class GlobalMinimization(_Minimization):
    def __init__(self, minimizer_type):

        super(GlobalMinimization, self).__init__(minimizer_type)

        assert issubclass(self._minimizer_type, GlobalMinimizer), (
            "Minimizer %s is not a local minimizer" % minimizer_type
        )

        self._2nd_minimization = None

    def setup(self, **setup_dict):

        assert "second_minimization" in setup_dict, (
            "You have to provide a secondary minimizer during setup, "
            "using the second_minimization keyword"
        )

        self._2nd_minimization = setup_dict["second_minimization"]

        super(GlobalMinimization, self).setup(**setup_dict)

    def get_second_minimization_instance(self, *args, **kwargs):

        return self._2nd_minimization.get_instance(*args, **kwargs)

    def get_instance(self, *args, **kwargs):

        instance = self._minimizer_type(*args, **kwargs)

        if self._algorithm is not None:

            instance.set_algorithm(self._algorithm)

        # Set up the minimizer
        instance._setup(self._setup_dict)

        return instance


class Minimizer(object):
    def __init__(self, function, parameters, verbosity=1, setup_dict=None):
        """

        :param function: function to be minimized
        :param parameters: ordered dictionary of the FREE parameters in the fit. The order must be the same as
               in the calling sequence of the function to be minimized.
        :param verbosity: control the verbosity of the output
        :param type: type of the optimizer (use the enums LOCAL_OPTIMIZER or GLOBAL_OPTIMIZER)
        :return:
        """

        self._function = function
        self._external_parameters = parameters
        self._internal_parameters = self._update_internal_parameter_dictionary()
        self._Npar = len(list(self.parameters.keys()))
        self._verbosity = verbosity

        self._setup(setup_dict)

        self._fit_results = None
        self._covariance_matrix = None
        self._correlation_matrix = None

        self._algorithm_name = None
        self._m_log_like_minimum = None

        self._optimizer_type = str(type)

    def _update_internal_parameter_dictionary(self):
        """
        Returns a dictionary parameter_name -> (current value, delta, minimum, maximum) in the internal frame
        (if the parameter has a transformation set).

        This should be used by the implementation of the minimizers to get the parameters to optimize.

        :return: dictionary
        """

        # Prepare the dictionary for the parameters which will be used by iminuit

        internal_parameter_dictionary = collections.OrderedDict()

        # NOTE: we use the internal_ versions of value, min_value and max_value because they don't have
        # units, and they are transformed to make the fit easier (for example in log scale)

        # NOTE as well that as in the entire class here, the .parameters dictionary only contains free parameters,
        # as only free parameters are passed to the constructor of the minimizer

        for k, par in self.parameters.items():

            current_name = par.path

            current_value = par._get_internal_value()
            current_delta = par._get_internal_delta()
            current_min = par._get_internal_min_value()
            current_max = par._get_internal_max_value()

            # Now fix sensible values for parameters deltas

            if current_min is None and current_max is None:

                # No boundaries, use 2% of value as initial delta

                if abs(current_delta) < abs(current_value) * 0.02 or not np.isfinite(
                    current_delta
                ):

                    current_delta = abs(current_value) * 0.02

            elif current_min is not None:

                if current_max is not None:

                    # Bounded in both directions. Use 20% of the value

                    current_delta = abs(current_value) * 0.02

                    # Make sure we do not violate the boundaries
                    current_delta = min(
                        current_delta,
                        abs(current_value - current_delta) / 10.0,
                        abs(current_value + current_delta) / 10.0,
                    )

                else:

                    # Bounded only in the negative direction. Make sure we are not at the boundary
                    if np.isclose(
                        current_value, current_min, old_div(
                            abs(current_value), 20)
                    ):

                        log.warning(
                            "The current value of parameter %s is very close to "
                            "its lower bound when starting the fit. Fixing it"
                            % par.name
                        )

                        current_value = current_value + \
                            0.1 * abs(current_value)

                        current_delta = 0.05 * abs(current_value)

                    else:

                        current_delta = min(
                            current_delta, abs(
                                current_value - current_min) / 10.0
                        )

            else:

                if current_max is not None:

                    # Bounded only in the positive direction
                    # Bounded only in the negative direction. Make sure we are not at the boundary
                    if np.isclose(
                        current_value, current_max, old_div(
                            abs(current_value), 20)
                    ):

                        log.warnings(
                            "The current value of parameter %s is very close to "
                            "its upper bound when starting the fit. Fixing it"
                            % par.name
                        )

                        current_value = current_value - \
                            0.04 * abs(current_value)

                        current_delta = 0.02 * abs(current_value)

                    else:

                        current_delta = min(
                            current_delta, abs(
                                current_max - current_value) / 2.0
                        )

            # Sometimes, if the value was 0, the delta could be 0 as well which would crash
            # certain algorithms
            if current_value == 0:

                current_delta = 0.1

            internal_parameter_dictionary[current_name] = (
                current_value,
                current_delta,
                current_min,
                current_max,
            )

        return internal_parameter_dictionary

    @property
    def function(self):

        return self._function

    @property
    def parameters(self):

        return self._external_parameters

    @property
    def Npar(self):

        return self._Npar

    @property
    def verbosity(self):

        return self._verbosity

    def _setup(self, setup_dict):

        raise NotImplementedError("You have to implement this.")

    @property
    def algorithm_name(self):

        return self._algorithm_name

    def minimize(self, compute_covar=True):
        """
        Minimize objective function. This call _minimize, which is implemented by each subclass.

        :param compute_covar:
        :return: best fit values (in external reference) and minimum of the objective function
        """

        # Gather the best fit values from the minimizer and the covariance matrix (if provided)

        try:

            internal_best_fit_values, function_minimum = self._minimize()

        except FitFailed:

            raise

        # Check that all values are finite

        # Check that the best_fit_values are finite
        if not np.all(np.isfinite(internal_best_fit_values)):

            raise FitFailed(
                "_Minimization apparently succeeded, "
                "but best fit values are not all finite: %s"
                % (internal_best_fit_values)
            )

        # Now set the internal values of the parameters to their best fit values and collect the
        # values in external reference
        external_best_fit_values = []

        for i, parameter in enumerate(self.parameters.values()):

            parameter._set_internal_value(internal_best_fit_values[i])

            external_best_fit_values.append(parameter.value)

        # Now compute the covariance matrix, if requested

        if compute_covar:

            covariance = self._compute_covariance_matrix(
                internal_best_fit_values)

        else:

            covariance = None

        # Finally store everything

        self._store_fit_results(internal_best_fit_values,
                                function_minimum, covariance)

        return external_best_fit_values, function_minimum

    def _minimize(self):

        # This should return the list of best fit parameters and the minimum of the function

        raise NotImplemented(
            "This is the method of the base class. Must be implemented by the actual minimizer"
        )

    def set_algorithm(self, algorithm):

        raise NotImplementedError(
            "Must be implemented by the actual minimizer if it provides more than one algorithm"
        )

    def _store_fit_results(
        self, best_fit_values, m_log_like_minimum, covariance_matrix=None
    ):

        self._m_log_like_minimum = m_log_like_minimum

        # Create a pandas DataFrame with the fit results

        values = collections.OrderedDict()
        errors = collections.OrderedDict()

        # to become compatible with python3
        keys_list = list(self.parameters.keys())
        parameters_list = list(self.parameters.values())

        for i in range(self.Npar):

            name = keys_list[i]

            value = best_fit_values[i]

            # Set the parameter to the best fit value (sometimes the optimization happen in a different thread/node,
            # so we need to make sure that the parameter has the best fit value)
            parameters_list[i]._set_internal_value(value)

            if covariance_matrix is not None:

                element = covariance_matrix[i, i]

                if element > 0:

                    error = math.sqrt(covariance_matrix[i, i])

                else:

                    log.warning(
                        "Negative element on diagonal of covariance matrix")

                    error = np.nan

            else:

                error = np.nan

            values[name] = value
            errors[name] = error

        data = collections.OrderedDict()
        data["value"] = pd.Series(values)
        data["error"] = pd.Series(errors)

        self._fit_results = pd.DataFrame(data)
        self._covariance_matrix = covariance_matrix

        # Compute correlation matrix

        self._correlation_matrix = np.zeros_like(self._covariance_matrix)

        if covariance_matrix is not None:

            for i in range(self.Npar):

                variance_i = self._covariance_matrix[i, i]

                for j in range(self.Npar):

                    variance_j = self._covariance_matrix[j, j]

                    if variance_i * variance_j > 0:

                        self._correlation_matrix[i, j] = old_div(
                            self._covariance_matrix[i, j],
                            (math.sqrt(variance_i * variance_j)),
                        )

                    else:

                        # We already issued a warning about this, so let's quietly fail

                        self._correlation_matrix[i, j] = np.nan

    @property
    def fit_results(self):

        return self._fit_results

    @property
    def covariance_matrix(self):

        return self._covariance_matrix

    @property
    def correlation_matrix(self):

        return self._correlation_matrix

    def restore_best_fit(self):
        """
        Reset all the parameters to their best fit value (from the last run fit)

        :return: none
        """

        best_fit_values = self._fit_results["value"].values

        for parameter_name, best_fit_value in zip(
            list(self.parameters.keys()), best_fit_values
        ):

            self.parameters[parameter_name]._set_internal_value(best_fit_value)

        # Regenerate the internal parameter dictionary with the new values
        self._internal_parameters = self._update_internal_parameter_dictionary()

    def _compute_covariance_matrix(self, best_fit_values):
        """
        This function compute the approximate covariance matrix as the inverse of the Hessian matrix,
        which is the matrix of second derivatives of the likelihood function with respect to
        the parameters.

        The sqrt of the diagonal of the result is an accurate estimate of the errors only if the
        log.likelihood is parabolic in the neighborhood of the minimum.

        Derivatives are computed numerically.

        :return: the covariance matrix
        """

        minima = [
            parameter._get_internal_min_value()
            for parameter in list(self.parameters.values())
        ]
        maxima = [
            parameter._get_internal_max_value()
            for parameter in list(self.parameters.values())
        ]

        # Check whether some of the minima or of the maxima are None. If they are, set them
        # to a value 1000 times smaller or larger respectively than the best fit.
        # An error of 3 orders of magnitude is not interesting in general, and this is the only
        # way to be able to compute a derivative numerically

        for i in range(len(minima)):

            if minima[i] is None:

                minima[i] = best_fit_values[i] / 1000.0

            if maxima[i] is None:

                maxima[i] = best_fit_values[i] * 1000.0

        # Transform them in np.array

        minima = np.array(minima)
        maxima = np.array(maxima)

        try:

            hessian_matrix = get_hessian(
                self.function, best_fit_values, minima, maxima)

        except ParameterOnBoundary:

            log.warning(
                "One or more of the parameters are at their boundaries. Cannot compute covariance and"
                " errors")

            n_dim = len(best_fit_values)

            return np.zeros((n_dim, n_dim)) * np.nan

        # Invert it to get the covariance matrix

        try:

            covariance_matrix = np.linalg.inv(hessian_matrix)

        except:

            log.warning(
                "Cannot invert Hessian matrix, looks like the matrix is singular"
            )

            n_dim = len(best_fit_values)

            return np.zeros((n_dim, n_dim)) * np.nan

        # Now check that the covariance matrix is semi-positive definite (it must be unless
        # there have been numerical problems, which can happen when some parameter is unconstrained)

        # The fastest way is to try and compute the Cholesky decomposition, which
        # works only if the matrix is positive definite

        try:

            _ = np.linalg.cholesky(covariance_matrix)

        except:

            log.warning(
                "Covariance matrix is NOT semi-positive definite. Cannot estimate errors. This can "
                "happen for many reasons, the most common being one or more unconstrained parameters"

            )

        return covariance_matrix

    def _get_one_error(self, parameter_name, target_delta_log_like, sign=-1):
        """
        A generic procedure to numerically compute the error for the parameters. You can override this if the
        minimizer provides its own method to compute the error of one parameter. If it provides a method to compute
        all errors are once, override the _get_errors method instead.

        :param parameter_name:
        :param target_delta_log_like:
        :param sign:
        :return:
        """

        # Since the procedure might find a better minimum, we can repeat it
        # up to a maximum of 10 times

        repeats = 0

        while repeats < 10:

            # Let's start optimistic...

            repeat = False

            repeats += 1

            # Restore best fit (which also updates the internal parameter dictionary)

            self.restore_best_fit()

            (
                current_value,
                current_delta,
                current_min,
                current_max,
            ) = self._internal_parameters[parameter_name]

            best_fit_value = current_value

            if sign == -1:

                extreme_allowed = current_min

            else:

                extreme_allowed = current_max

            # If the parameter has no boundary in the direction we are sampling, put a hard limit on
            # 10 times the current value (to avoid looping forever)

            if extreme_allowed is None:

                extreme_allowed = best_fit_value + \
                    sign * 10 * abs(best_fit_value)

            # We need to look for a value for the parameter where the difference between the minimum of the
            # log-likelihood and the likelihood for that value differs by more than target_delta_log_likelihood.
            # This is needed by the root-finding procedure, which needs to know an interval where the biased likelihood
            # function (see below) changes sign

            trials = best_fit_value + sign * np.linspace(0.1, 0.9, 9) * abs(
                best_fit_value
            )

            trials = np.append(trials, extreme_allowed)

            # Make sure we don't go below the allowed minimum or above the allowed maximum

            if sign == -1:

                np.clip(trials, extreme_allowed, np.inf, trials)

            else:

                np.clip(trials, -np.inf, extreme_allowed, trials)

            # There might be more than one value which was below the minimum (or above the maximum), so let's
            # take only unique elements

            trials = np.unique(trials)

            trials.sort()

            if sign == -1:

                trials = trials[::-1]

            # At this point we have a certain number of unique trials which always
            # contain the allowed minimum (or maximum)

            minimum_bound = None
            maximum_bound = None

            # Instance the profile likelihood function
            pl = ProfileLikelihood(self, [parameter_name])

            for i, trial in enumerate(trials):

                this_log_like = pl([trial])

                delta = this_log_like - self._m_log_like_minimum

                if delta < -0.1:

                    log.warning(
                        "Found a better minimum (%.2f) for %s = %s during error "
                        "computation." % (
                            this_log_like, parameter_name, trial)

                    )

                    xs = [x.value for x in list(self.parameters.values())]

                    self._store_fit_results(xs, this_log_like, None)

                    repeat = True

                    break

                if delta > target_delta_log_like:

                    bound1 = trial

                    if i > 0:

                        bound2 = trials[i - 1]

                    else:

                        bound2 = best_fit_value

                    minimum_bound = min(bound1, bound2)
                    maximum_bound = max(bound1, bound2)

                    repeat = False

                    break

            if repeat:

                # We found a better minimum, restart from scratch

                log.warning("Restarting search...")

                continue

            if minimum_bound is None:

                # Cannot find error in this direction (it's probably outside the allowed boundaries)
                log.warning(
                    "Cannot find boundary for parameter %s" % parameter_name

                )

                error = np.nan
                break

            else:

                # Define the "biased likelihood", since brenq only finds zeros of function

                biased_likelihood = (
                    lambda x: pl(x) - self._m_log_like_minimum -
                    target_delta_log_like
                )

                try:

                    precise_bound = scipy.optimize.brentq(
                        biased_likelihood,
                        minimum_bound,
                        maximum_bound,
                        xtol=1e-5,
                        maxiter=1000,
                    )  # type: float
                except:

                    log.warning(
                        "Cannot find boundary for parameter %s" % parameter_name)

                    error = np.nan
                    break

                error = precise_bound - best_fit_value

                break

        return error

    def get_errors(self):
        """
        Compute asymmetric errors using the profile likelihood method (slow, but accurate).

        :return: a dictionary with asymmetric errors for each parameter
        """

        # Restore best fit so error computation starts from there

        self.restore_best_fit()

        # Get errors

        errors_dict = self._get_errors()

        # Transform in external reference if needed

        best_fit_values = self._fit_results["value"]

        for par_name, (negative_error, positive_error) in errors_dict.items():

            parameter = self.parameters[par_name]

            if parameter.has_transformation():

                _, negative_error_external = parameter.internal_to_external_delta(
                    best_fit_values[parameter.path], negative_error
                )

                _, positive_error_external = parameter.internal_to_external_delta(
                    best_fit_values[parameter.path], positive_error
                )

                errors_dict[par_name] = (
                    negative_error_external,
                    positive_error_external,
                )

            else:

                # No need to transform
                pass

        return errors_dict

    def _get_errors(self):
        """
        Override this method if the minimizer provide a function to get all errors at once. If instead it provides
        a method to get one error at the time, override the _get_one_error method

        :return: a ordered dictionary parameter_path -> (negative_error, positive_error)
        """

        # TODO: options for other significance levels

        target_delta_log_like = 0.5

        errors = collections.OrderedDict()

        p = tqdm(total=2 * len(self.parameters), desc="Computing errors")

        for parameter_name in self.parameters:

            negative_error = self._get_one_error(
                parameter_name, target_delta_log_like, -1
            )

            p.update(1)

            positive_error = self._get_one_error(
                parameter_name, target_delta_log_like, +1
            )

            p.update(1)

            errors[parameter_name] = (negative_error, positive_error)

        return errors

    def contours(
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
        **options
    ):
        """
            Generate confidence contours for the given parameters by stepping for the given number of steps between
            the given boundaries. Call it specifying only source_1, param_1, param_1_minimum and param_1_maximum to
            generate the profile of the likelihood for parameter 1. Specify all parameters to obtain instead a 2d
            contour of param_1 vs param_2

            :param param_1: name of the first parameter
            :param param_1_minimum: lower bound for the range for the first parameter
            :param param_1_maximum: upper bound for the range for the first parameter
            :param param_1_n_steps: number of steps for the first parameter
            :param param_2: name of the second parameter
            :param param_2_minimum: lower bound for the range for the second parameter
            :param param_2_maximum: upper bound for the range for the second parameter
            :param param_2_n_steps: number of steps for the second parameter
            :param progress: (True or False) whether to display progress or not
            :param log: by default the steps are taken linearly. With this optional parameter you can provide a tuple of
            booleans which specify whether the steps are to be taken logarithmically. For example,
            'log=(True,False)' specify that the steps for the first parameter are to be taken logarithmically, while they
            are linear for the second parameter. If you are generating the profile for only one parameter, you can specify
             'log=(True,)' or 'log=(False,)' (optional)
            :param: parallel: whether to use or not parallel computation (default:False)
            :return: a : an array corresponding to the steps for the first parameter
                     b : an array corresponding to the steps for the second parameter (or None if stepping only in one
                     direction)
                     contour : a matrix of size param_1_steps x param_2_steps containing the value of the function at the
                     corresponding points in the grid. If param_2_steps is None (only one parameter), then this reduces to
                     an array of size param_1_steps.
            """

        # Figure out if we are making a 1d or a 2d contour

        if param_2 is None:

            n_dimensions = 1
            fixed_parameters = [param_1]

        else:

            n_dimensions = 2
            fixed_parameters = [param_1, param_2]

        # Check the options

        p1log = False
        p2log = False

        if "log" in list(options.keys()):

            assert len(options["log"]) == n_dimensions, (
                "When specifying the 'log' option you have to provide a "
                + "boolean for each dimension you are stepping on."
            )

            p1log = bool(options["log"][0])

            if param_2 is not None:

                p2log = bool(options["log"][1])

        # Generate the steps

        if p1log:

            param_1_steps = np.logspace(
                math.log10(param_1_minimum),
                math.log10(param_1_maximum),
                param_1_n_steps,
            )

        else:

            param_1_steps = np.linspace(
                param_1_minimum, param_1_maximum, param_1_n_steps
            )

        if n_dimensions == 2:

            if p2log:

                param_2_steps = np.logspace(
                    math.log10(param_2_minimum),
                    math.log10(param_2_maximum),
                    param_2_n_steps,
                )

            else:

                param_2_steps = np.linspace(
                    param_2_minimum, param_2_maximum, param_2_n_steps
                )

        else:

            # Only one parameter to step through
            # Put param_2_steps as nan so that the worker can realize that it does not have
            # to step through it

            param_2_steps = np.array([np.nan])

        # Define the worker which will compute the value of the function at a given point in the grid

        # Restore best fit

        if self.fit_results is not None:

            self.restore_best_fit()

        else:

            log.warning(
                "No best fit to restore before contours computation. "
                "Perform the fit before running contours to remove this warnings."
            )

        pr = ProfileLikelihood(self, fixed_parameters)

        if n_dimensions == 1:

            results = pr.step(param_1_steps)

        else:

            results = pr.step(param_1_steps, param_2_steps)

        # Return results

        return (
            param_1_steps,
            param_2_steps,
            np.array(results).reshape(
                (param_1_steps.shape[0], param_2_steps.shape[0])),
        )


class LocalMinimizer(Minimizer):

    pass


class GlobalMinimizer(Minimizer):

    pass


# Check which minimizers are available

try:

    from threeML.minimizer.minuit_minimizer import MinuitMinimizer

except ImportError:

    log.warning("Minuit minimizer not available")

else:

    _minimizers["MINUIT"] = MinuitMinimizer

try:

    from threeML.minimizer.ROOT_minimizer import ROOTMinimizer

except ImportError:

    log.warning("ROOT minimizer not available")

else:

    _minimizers["ROOT"] = ROOTMinimizer

try:

    from threeML.minimizer.multinest_minimizer import MultinestMinimizer

except ImportError:

    log.warning("Multinest minimizer not available")

else:

    _minimizers["MULTINEST"] = MultinestMinimizer

try:

    from threeML.minimizer.pagmo_minimizer import PAGMOMinimizer

except ImportError:

    log.warning("PyGMO is not available")

else:

    _minimizers["PAGMO"] = PAGMOMinimizer

try:

    from threeML.minimizer.scipy_minimizer import ScipyMinimizer

except ImportError:

    log.warning("Scipy minimizer is not available")

else:

    _minimizers["SCIPY"] = ScipyMinimizer

# Check that we have at least one minimizer available

if len(_minimizers) == 0:

    raise SystemError(
        "You do not have any minimizer available! You need to install at least iminuit."
    )



else:
    # Add the GRID minimizer here since it needs at least one other minimizer

    from threeML.minimizer.grid_minimizer import GridMinimizer
    
    _minimizers["GRID"] = GridMinimizer
