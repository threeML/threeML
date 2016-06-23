import collections
import math

import numpy as np
import pandas as pd
import numdifftools as nd
from iminuit import Minuit

from astromodels import SettingOutOfBounds

from threeML.io.progress_bar import ProgressBar
import scipy.optimize

from threeML.exceptions.custom_exceptions import custom_warnings
from threeML.utils.differentiation import get_hessian, ParameterOnBoundary

# Set the warnings to be issued always for this module

custom_warnings.simplefilter("always", RuntimeWarning)


try:

    import ROOT

except ImportError:

    has_ROOT = False

else:

    has_ROOT = True

try:

    import pyOpt

except ImportError:

    has_pyOpt = False

else:

    has_pyOpt = True

# Special constants
FIT_FAILED = 1e12


# Define a bunch of custom exceptions relevant for what is being accomplished here

class CannotComputeCovariance(RuntimeWarning):
    pass


class CannotComputeErrors(RuntimeWarning):
    pass


class MINOSFailed(Exception):
    pass


class ParameterIsNotFree(Exception):
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

    :param minimizer_type: MINUIT, ROOT, or PYOPT
    :param minimizer_algorithm: algorithm (optional, use only for PYOPT)
    :return: the class (i.e., the type) for the requested minimizer
    """

    try:

        return _minimizers[minimizer_type]

    except KeyError:

        raise MinimizerNotAvailable("Minimizer %s is not available on your system" % minimizer_type)


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

            this_index = self._all_parameters.keys().index(parameter_name)

            self._indexes_of_fixed_par[this_index] = True

        self._all_values = np.zeros(len(self._all_parameters))

    def set_fixed_values(self, new_fixed_values):

        # A use [:] so there is an implicit check on the right size of new_fixed_values

        self._fixed_parameters_values[:] = new_fixed_values

    def __call__(self, *trial_values):

        self._all_values[self._indexes_of_fixed_par] = self._fixed_parameters_values
        self._all_values[~self._indexes_of_fixed_par] = trial_values

        return self._function(*self._all_values)


class ProfileLikelihood(object):

    def __init__(self, minimizer_instance, fixed_parameters):

        self._original_minimizer = minimizer_instance

        self._fixed_parameters = fixed_parameters

        assert len(self._fixed_parameters) <= 2, "Can handle only one or two fixed parameters"

        # Get some info from the original minimizer

        self._function = self._original_minimizer.function

        self._all_parameters = self._original_minimizer.parameters

        ftol = self._original_minimizer.ftol

        # Create a copy of the dictionary of parameters

        free_parameters = collections.OrderedDict(self._all_parameters)

        # Remove the fixed ones

        for parameter_name in fixed_parameters:

            free_parameters.pop(parameter_name)

        # Now compute how many free parameters we have

        self._n_free_parameters = len(free_parameters)

        if self._n_free_parameters > 0:

            self._wrapper = FunctionWrapper(self._function,
                                            self._all_parameters,
                                            self._fixed_parameters)

            # Create a copy of the optimizer with the new parameters (i.e., one or two
            # parameters fixed to their current values)

            self._optimizer = type(self._original_minimizer)(self._wrapper, free_parameters, ftol, verbosity=0)

            if self._original_minimizer.algorithm_name is not None:

                self._optimizer.set_algorithm(self._original_minimizer.algorithm_name)

        else:

            # Special case when there are no free parameters after fixing the requested ones
            # There is no profiling necessary here

            self._wrapper = None
            self._optimizer = None

    def step(self, steps1, steps2=None):

        if steps2 is not None:

            assert len(self._fixed_parameters) == 2, "Cannot step in 2d if you fix only one parameter"

            # Find out if the user is giving flipped steps (i.e. param_1 is after param_2 in the
            # parameters dictionary)

            param_1_name = self._fixed_parameters[0]
            param_1_idx = self._all_parameters.keys().index(param_1_name)

            param_2_name = self._fixed_parameters[1]
            param_2_idx = self._all_parameters.keys().index(param_2_name)

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

            assert len(self._fixed_parameters) == 1, "You cannot step in 1d if you fix 2 parameters"

            return self._step1d(steps1)

    def __call__(self, values):

        self._wrapper.set_fixed_values(values)

        _, this_log_like = self._optimizer.minimize(compute_covar=False)

        return this_log_like

    def _step1d(self, steps1):

        log_likes = np.zeros_like(steps1)

        p = ProgressBar(len(steps1))

        for i, step in enumerate(steps1):

            if self._n_free_parameters > 0:

                # Profile out the free parameters

                self._wrapper.set_fixed_values(step)

                _, this_log_like = self._optimizer.minimize(compute_covar=False)

            else:

                # No free parameters, just compute the likelihood

                this_log_like = self._function(step)

            log_likes[i] = this_log_like

            p.increase()

        return log_likes

    def _step2d(self, steps1, steps2):

        log_likes = np.zeros((len(steps1), len(steps2)))

        p = ProgressBar(len(steps1) * len(steps2))

        for i, step1 in enumerate(steps1):

            for j,step2 in enumerate(steps2):

                if self._n_free_parameters > 0:

                    # Profile out the free parameters

                    self._wrapper.set_fixed_values([step1, step2])

                    _, this_log_like = self._optimizer.minimize(compute_covar=False)

                else:

                    # No free parameters, just compute the likelihood

                    this_log_like = self._function(step1, step2)

                log_likes[i,j] = this_log_like

                p.increase()

        return log_likes


class Minimizer(object):

    def __init__(self, function, parameters, ftol=1e-3, verbosity=1):
        """

        :param function: function to be minimized
        :param parameters: ordered dictionary of the FREE parameters in the fit. The order must be the same as
               in the calling sequence of the function to be minimized.
        :param ftol: fractional tolerance to be used in the fit
        :param verbosity: control the verbosity of the output
        :return:
        """

        self.function = function
        self.parameters = parameters
        self.Npar = len(self.parameters.keys())
        self.ftol = ftol
        self.verbosity = verbosity

        self._setup()

        self._fit_results = None
        self._covariance_matrix = None
        self._correlation_matrix = None

        self._algorithm_name = None
        self._m_log_like_minimum = None

    def _setup(self):

        raise NotImplementedError("You have to implement this.")

    @property
    def algorithm_name(self):

        return self._algorithm_name

    def minimize(self):
        raise NotImplemented("This is the method of the base class. Must be implemented by the actual minimizer")

    def set_algorithm(self, algorithm):

        raise NotImplementedError("Must be implemented by the actual minimizer if it provides more than one algorithm")

    def _store_fit_results(self, best_fit_values, m_log_like_minimum, covariance_matrix=None):

        self._m_log_like_minimum = m_log_like_minimum

        # Create a pandas DataFrame with the fit results

        values = collections.OrderedDict()
        errors = collections.OrderedDict()

        for i in range(self.Npar):

            name = self.parameters.keys()[i]

            value = best_fit_values[i]

            if covariance_matrix is not None:

                element = covariance_matrix[i,i]

                if element > 0:

                    error = math.sqrt(covariance_matrix[i, i])

                else:

                    custom_warnings.warn("Negative element on diagonal of covariance matrix", CannotComputeErrors)

                    error = np.nan

            else:

                error = np.nan

            values[name] = value
            errors[name] = error

        data = collections.OrderedDict()
        data['value'] = pd.Series(values)
        data['error'] = pd.Series(errors)

        self._fit_results = pd.DataFrame(data)
        self._covariance_matrix = covariance_matrix

        # Compute correlation matrix

        self._correlation_matrix = np.zeros_like(self._covariance_matrix)

        if covariance_matrix is not None:

            for i in range(self.Npar):

                variance_i = self._covariance_matrix[i,i]

                for j in range(self.Npar):

                    variance_j = self._covariance_matrix[j,j]

                    if variance_i * variance_j > 0:

                        self._correlation_matrix[i,j] = self._covariance_matrix[i,j] / (math.sqrt(variance_i * variance_j))

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

        best_fit_values = self._fit_results['value'].values

        for parameter_name, best_fit_value in zip(self.parameters.keys(), best_fit_values):

            self.parameters[parameter_name].value = best_fit_value

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

        minima = np.array(map(lambda parameter:parameter.min_value, self.parameters.values()))
        maxima = np.array(map(lambda parameter: parameter.max_value, self.parameters.values()))

        try:

            hessian_matrix = get_hessian(self.function, best_fit_values, minima, maxima)

        except ParameterOnBoundary:

            custom_warnings.warn("One or more of the parameters are at their boundaries. Cannot compute covariance and"
                                 " errors", CannotComputeCovariance)

            n_dim = len(best_fit_values)

            return np.zeros((n_dim,n_dim)) * np.nan

        # Invert it to get the covariance matrix

        covariance_matrix = np.linalg.inv(hessian_matrix)

        # Now check that the covariance matrix is semi-positive definite (it must be unless
        # there have been numerical problems, which can happen when some parameter is unconstrained)

        # The fastest way is to try and compute the Cholesky decomposition, which
        # works only if the matrix is positive definite

        try:

            _ = np.linalg.cholesky(covariance_matrix)

        except:

            custom_warnings.warn("Covariance matrix is NOT semi-positive definite. Cannot estimate errors. This can "
                                 "happen for many reasons, the most common being one or more unconstrained parameters",
                                 CannotComputeCovariance)

        return covariance_matrix

    def _get_error(self, parameter_name, target_delta_log_like, sign=-1):

        # Since the procedure might find a better minimum, we can repeat it
        # up to a maximum of 10 times

        repeats = 0

        while repeats < 10:

            # Let's start optimistic...

            repeat = False

            repeats += 1

            self.restore_best_fit()

            best_fit_value = self.parameters[parameter_name].value

            if sign == -1:

                extreme_allowed = self.parameters[parameter_name].min_value

            else:

                extreme_allowed = self.parameters[parameter_name].max_value

            # If the parameter has no boundary in the direction we are sampling, put a hard limit on
            # 10 times the current value (to avoid looping forever)

            if extreme_allowed is None:

                extreme_allowed = best_fit_value + sign * 10 * abs(best_fit_value)

            # We need to look for a value for the parameter where the difference between the minimum of the
            # log-likelihood and the likelihood for that value differs by more than target_delta_log_likelihood.
            # This is needed by the root-finding procedure, which needs to know an interval where the biased likelihood
            # function (see below) changes sign

            trials = best_fit_value + sign * np.linspace(0.1, 0.9, 9) * abs(best_fit_value)

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

                    custom_warnings.warn("Found a better minimum (%.2f) for %s = %s during error "
                                         "computation." % (this_log_like, parameter_name, trial),
                                         BetterMinimumDuringProfiling)

                    xs = map(lambda x:x.value, self.parameters.values())

                    self._store_fit_results(xs, this_log_like, None)

                    repeat = True

                    break

                if delta > target_delta_log_like:

                    bound1 = trial

                    if i > 0:

                        bound2 = trials[i-1]

                    else:

                        bound2 = best_fit_value

                    minimum_bound = min(bound1, bound2)
                    maximum_bound = max(bound1, bound2)

                    repeat = False

                    break

            if repeat:

                # We found a better minimum, restart from scratch

                custom_warnings.warn("Restarting search...", RuntimeWarning)

                continue

            if minimum_bound is None:

                # Cannot find error in this direction (it's probably outside the allowed boundaries)
                custom_warnings.warn("Cannot find lower boundary for parameter %s" % parameter_name, CannotComputeErrors)

                error = np.nan
                break

            else:

                # Define the "biased likelihood", since brenq only finds zeros of function

                biased_likelihood = lambda x: pl(x) - self._m_log_like_minimum - target_delta_log_like

                if sign == -1:

                    precise_bound = scipy.optimize.brentq(biased_likelihood, minimum_bound,
                                                          maximum_bound, xtol=1e-5, maxiter=1000)

                else:

                    precise_bound = scipy.optimize.brentq(biased_likelihood, minimum_bound,
                                                          maximum_bound, xtol=1e-5, maxiter=1000)

                error = precise_bound - best_fit_value

                break

        return error

    def get_errors(self):
        """
        Compute asymmetric errors using the profile likelihood method (slow, but accurate).

        :return: a list with asymmetric errors for each parameter
        """

        # TODO: options for other significance levels

        target_delta_log_like = 0.5

        self.restore_best_fit()

        p = ProgressBar(2 * len(self.parameters))

        errors = collections.OrderedDict()

        for parameter_name in self.parameters:

            negative_error = self._get_error(parameter_name, target_delta_log_like, -1)

            p.increase()

            positive_error = self._get_error(parameter_name, target_delta_log_like, +1)

            p.increase()

            errors[parameter_name] = (negative_error, positive_error)

        return errors

    def contours(self, param_1, param_1_minimum, param_1_maximum, param_1_n_steps,
                         param_2=None, param_2_minimum=None, param_2_maximum=None, param_2_n_steps=None,
                         progress=True, **options):

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

            if 'log' in options.keys():

                assert len(options['log']) == n_dimensions, ("When specifying the 'log' option you have to provide a " +
                                                             "boolean for each dimension you are stepping on.")

                p1log = bool(options['log'][0])

                if param_2 is not None:

                    p2log = bool(options['log'][1])

            # Generate the steps

            if p1log:

                param_1_steps = np.logspace(math.log10(param_1_minimum), math.log10(param_1_maximum),
                                            param_1_n_steps)

            else:

                param_1_steps = np.linspace(param_1_minimum, param_1_maximum,
                                            param_1_n_steps)

            if n_dimensions == 2:

                if p2log:

                    param_2_steps = np.logspace(math.log10(param_2_minimum), math.log10(param_2_maximum),
                                                param_2_n_steps)

                else:

                    param_2_steps = np.linspace(param_2_minimum, param_2_maximum,
                                                param_2_n_steps)

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

                custom_warnings.warn("No best fit to restore before contours computation. "
                                     "Perform the fit before running contours to remove this warnings.")

            pr = ProfileLikelihood(self, fixed_parameters)

            if n_dimensions == 1:

                results = pr.step(param_1_steps)

            else:

                results = pr.step(param_1_steps, param_2_steps)

            # Return results

            return param_1_steps, param_2_steps, np.array(results).reshape((param_1_steps.shape[0],
                                                                            param_2_steps.shape[0]))

    # def print_fit_results(self):
    #     """
    #     Display the results of the last minimization.
    #
    #     :return: (none)
    #     """
    #
    #     data = []
    #
    #     # Also store the maximum length to decide the length for the line
    #
    #     name_length = 0
    #
    #     for parameter_name in self._fit_results.index.values:
    #
    #         value = self._fit_results.at[parameter_name, 'value']
    #
    #         error = self._fit_results.at[parameter_name, 'error']
    #
    #         # Format the value and the error with sensible significant
    #         # numbers
    #         x = uncertainties.ufloat(value, error)
    #
    #         # Add some space around the +/- sign
    #
    #         rep = x.__str__().replace("+/-", " +/- ")
    #
    #         data.append([parameter_name, rep, self.parameters[parameter_name].unit])
    #
    #         if len(parameter_name) > name_length:
    #
    #             name_length = len(parameter_name)
    #
    #     table = Table(rows=data,
    #                   names=["Name", "Best fit value", "Unit"],
    #                   dtype=('S%i' % name_length, str, str))
    #
    #     display(table)
    #
    #     print("\nNOTE: errors on parameters are approximate. Use get_errors().\n")


# This is a function to add a method to a class
# We will need it in the MinuitMinimizer

def add_method(self, method, name=None):
    if name is None:
        name = method.func_name

    setattr(self.__class__, name, method)


class MinuitMinimizer(Minimizer):

    # NOTE: this class is built to be able to work both with iMinuit and with a boost interface to SEAL
    # minuit, i.e., it does not rely on functionality that iMinuit provides which is not of the original
    # minuit. This makes the implementation a little bit more cumbersome, but more adaptable if we want
    # to switch back to the bare bone SEAL minuit

    def __init__(self, function, parameters, ftol=1, verbosity=0):

        super(MinuitMinimizer, self).__init__(function, parameters, ftol, verbosity)

    def _setup(self):

        # Prepare the dictionary for the parameters which will be used by iminuit

        iminuit_init_parameters = {}

        # List of variable names that will be used for iminuit.

        variable_names_for_iminuit = []

        # NOTE: we use the scaled_ versions of value, min_value and max_value because they don't have
        # units, and hence they are much faster to set and retrieve. These are indeed introduced by
        # astromodels to be used for computing-intensive situations like fitting

        for k, par in self.parameters.iteritems():

            current_name = self._parameter_name_to_minuit_name(k)

            variable_names_for_iminuit.append(current_name)

            # Initial value
            iminuit_init_parameters['%s' % current_name] = par.value

            # Initial delta
            iminuit_init_parameters['error_%s' % current_name] = par.delta

            # Limits
            iminuit_init_parameters['limit_%s' % current_name] = (par.min_value, par.max_value)

            # This is useless, since all parameters here are free,
            # but do it anyway for clarity
            iminuit_init_parameters['fix_%s' % current_name] = False

        # This is to tell Minuit that we are dealing with likelihoods,
        # not chi square
        iminuit_init_parameters['errordef'] = 0.5

        iminuit_init_parameters['print_level'] = self.verbosity

        # We need to make a function with the parameters as explicit
        # variables in the calling sequence, so that Minuit will be able
        # to probe the parameter's names
        var_spelled_out = ",".join(variable_names_for_iminuit)

        # A dictionary to keep a way to convert from var. name to
        # variable position in the function calling sequence
        # (will use this in contours)

        self.name_to_position = {k: i for i, k in enumerate(variable_names_for_iminuit)}

        # Write and compile the code for such function

        code = 'def _f(self, %s):\n  return self.function(%s)' % (var_spelled_out, var_spelled_out)
        exec code

        # Add the function just created as a method of the class
        # so it will be able to use the 'self' pointer
        add_method(self, _f, "_f")

        # Finally we can instance the Minuit class
        self.minuit = Minuit(self._f, **iminuit_init_parameters)

        self.minuit.tol = self.ftol  # ftol

        try:

            self.minuit.up = 0.5  # This is a likelihood

        except AttributeError:

            # iMinuit uses errodef, not up

            self.minuit.errordef = 0.5

        self.minuit.strategy = 0  # More accurate

        self._best_fit_parameters = None
        self._function_minimum_value = None

    @staticmethod
    def _parameter_name_to_minuit_name(parameter):
        """
        Translate the name of the parameter to the format accepted by Minuit

        :param parameter: the parameter name, of the form source.component.shape.parname
        :return: a minuit-friendly name for the parameter, such as source_component_shape_parname
        """

        return parameter.replace(".", "_")

    def _migrad_has_converged(self):

        # In the MINUIT manual this is the condition for MIGRAD to have converged
        # 0.002 * tolerance * UPERROR (which is 0.5 for likelihood)

        return self.minuit.edm <= 0.002 * self.minuit.tol * 0.5

    def _run_migrad(self, trials=10):

        # Repeat Migrad up to trials times, until it converges

        minimum = None

        for i in range(trials):

            self.minuit.migrad()

            minimum = self.minuit.fval

            if self._migrad_has_converged():

                # Converged
                break

            else:

                # Try again
                continue

        return minimum

    # Override this because minuit uses different names
    def restore_best_fit(self):
        """
        Set the parameters back to their best fit value

        :return: none
        """

        super(MinuitMinimizer, self).restore_best_fit()

        for k, par in self.parameters.iteritems():

            minuit_name = self._parameter_name_to_minuit_name(k)

            self.minuit.values[minuit_name] = par.value

    def minimize(self, compute_covar=True):
        """
        Minimize the function using MIGRAD

        :param compute_covar: whether to compute the covariance (and error estimates) or not
        :return: best_fit: a dictionary containing the parameters at their best fit values
                 function_minimum : the value for the function at the minimum

                 NOTE: if the minimization fails, the dictionary will be empty and the function_minimum will be set
                 to minimization.FIT_FAILED
        """

        minimum = self._run_migrad(10)

        if not self._migrad_has_converged():

            print("\nMIGRAD did not converge in 10 trials.")

            return collections.OrderedDict(), FIT_FAILED

        else:

            # Get the best fit values

            best_fit_values = []

            for k, par in self.parameters.iteritems():

                minuit_name = self._parameter_name_to_minuit_name(k)

                best_fit_values.append(self.minuit.values[minuit_name])

            # Get covariance matrix

            if compute_covar:

                covariance = self._compute_covariance_matrix(best_fit_values)

            else:

                covariance = None

            # Now store the results

            self._store_fit_results(best_fit_values, minimum, covariance)

            return best_fit_values, minimum

    # Override the default _compute_covariance_matrix
    def _compute_covariance_matrix(self, best_fit_values):

        self.minuit.hesse()

        covariance = np.array(self.minuit.matrix(correlation=False))

        return covariance

    # def print_fit_results(self):
    #     """
    #     Display the results of the last minimization.
    #
    #     :return: (none)
    #     """
    #
    #     # Restore the best fit values, in case something has changed
    #     self._restore_best_fit()
    #
    #     # I do not use the print_param facility in iminuit because
    #     # it does not work well with console output, since it fails
    #     # to autoprobe that it is actually run in a console and uses
    #     # the HTML backend instead
    #
    #     # Create a list of strings to print
    #
    #     data = []
    #
    #     # Also store the maximum length to decide the length for the line
    #
    #     name_length = 0
    #
    #     for k, v in self.parameters.iteritems():
    #
    #         minuit_name = self._parameter_name_to_minuit_name(k)
    #
    #         # Format the value and the error with sensible significant
    #         # numbers
    #         x = uncertainties.ufloat(v.value, self.minuit.errors[minuit_name])
    #
    #         # Add some space around the +/- sign
    #
    #         rep = x.__str__().replace("+/-", " +/- ")
    #
    #         data.append([k, rep, v.unit])
    #
    #         if len(k) > name_length:
    #             name_length = len(k)
    #
    #     table = Table(rows=data,
    #                   names=["Name", "Value", "Unit"],
    #                   dtype=('S%i' % name_length, str, str))
    #
    #     display(table)
    #
    #     print("\nNOTE: errors on parameters are approximate. Use get_errors().\n")

    # Override the default _compute_covariance_matrix

    # def print_correlation_matrix(self):
    #     """
    #     Display the current correlation matrix
    #     :return: (none)
    #     """
    #
    #     # Print a custom covariance matrix because iminuit does
    #     # not guess correctly the frontend when 3ML is used
    #     # from terminal
    #
    #     cov = self.minuit.covariance
    #
    #     if cov is None:
    #         raise CannotComputeCovariance("Cannot compute covariance numerically. This usually means that there are " +
    #                                       " unconstrained parameters. Fix those or reduce their allowed range, or " +
    #                                       "use a simpler model.")
    #
    #     # Get list of parameters
    #
    #     keys = self.parameters.keys()
    #
    #     # Convert them to the format for iminuit
    #
    #     minuit_names = map(lambda k: self._parameter_name_to_minuit_name(k), keys)
    #
    #     # Accumulate rows and compute the maximum length of the names
    #
    #     data = []
    #     length_of_names = 0
    #
    #     for key1, name1 in zip(keys, minuit_names):
    #
    #         if len(name1) > length_of_names:
    #             length_of_names = len(name1)
    #
    #         this_row = []
    #
    #         for key2, name2 in zip(keys, minuit_names):
    #             # Compute correlation between parameter key1 and key2
    #
    #             corr = cov[(name1, name2)] / (math.sqrt(cov[(name1, name1)]) * math.sqrt(cov[(name2, name2)]))
    #
    #             this_row.append(corr)
    #
    #         data.append(this_row)
    #
    #     # Prepare the dtypes for the matrix
    #
    #     dtypes = map(lambda x: float, minuit_names)
    #
    #     # Column names are the parameter names
    #
    #     cols = keys
    #
    #     # Finally generate the matrix with the names
    #
    #     table = NumericMatrix(rows=data,
    #                           names=cols,
    #                           dtype=dtypes)
    #
    #     # Customize the format to avoid too many digits
    #
    #     for col in table.colnames:
    #         table[col].format = '2.2f'
    #
    #     display(table)

    def get_errors(self):
        """
        Compute asymmetric errors using MINOS (slow, but accurate) and print them.

        NOTE: this should be called immediately after the minimize() method

        :return: a dictionary containing the asymmetric errors for each parameter.
        """

        self.restore_best_fit()

        if not self._migrad_has_converged():
            raise CannotComputeErrors("MIGRAD results not valid, cannot compute errors. Did you run the fit first ?")

        try:

            self.minuit.minos()

        except:

            raise

        # except:
        #
        #     raise MINOSFailed("MINOS has failed. This usually means that the fit is very difficult, for example "
        #                       "because of high correlation between parameters. Check the correlation matrix printed"
        #                       "in the fit step, and check contour plots with getContours(). If you are using a "
        #                       "user-defined model, you can also try to "
        #                       "reformulate your model with less correlated parameters.")

        # Make a list for the results

        errors = collections.OrderedDict()

        for k, par in self.parameters.iteritems():

            minuit_name = self._parameter_name_to_minuit_name(k)

            minus_error = self.minuit.merrors[(minuit_name, -1)]
            plus_error = self.minuit.merrors[(minuit_name, 1)]

            errors[k] = ((minus_error,plus_error))

        return errors


# Add Minuit to the available minimizers
_minimizers["MINUIT"] = MinuitMinimizer


if has_pyOpt:

    class PyOptWrapper(object):

        # This is needed by pyopt

        __name__ = "Likelihood"

        def __init__(self, function, dimensions):

            self._function = function
            self._dimensions = dimensions

        def __call__(self, x):

            new_args = map(lambda i: x[i], range(self._dimensions))

            try:

                f = self._function(*new_args)

            except SettingOutOfBounds:

                f = FIT_FAILED

            if f == FIT_FAILED:

                # Likelihood gave nan or other problems, we are likely in a forbidden
                # space

                fail = 1

            else:

                # Likelihood computation successful

                fail = 0

            # The empty list is for the constraints vector. It is empty
            # because this is a unconstrained problem, where unconstrained means
            # there are no additional conditions on top of the boundaries for the
            # parameters (if any)

            return f, [], fail


    def get_pyopt_available_algorithms():
        """
        Returns a dictionary with the name of the optimizers as key and the relative class as value

        :return: dictionary
        """

        optimizers = {}

        for element_name in dir(pyOpt):

            element = eval("pyOpt.%s" % element_name)

            try:

                is_subclass = issubclass(element, pyOpt.Optimizer) and not element == pyOpt.Optimizer

            except TypeError:

                continue

            else:

                if is_subclass:

                    optimizers[element_name] = element

        return optimizers

    _pyopt_algorithms = get_pyopt_available_algorithms()

    # Remove algorithms that do not work in our cases
    # 'ALPSO', 'SLSQP', 'SOLVOPT' "ALGENCAN" NSGA2 ALHSO FILTERSD

    for alg in ['ALPSO', 'SLSQP', 'SOLVOPT', 'ALGENCAN', 'NSGA2', 'ALHSO', 'FILTERSD']:

        if alg in _pyopt_algorithms:

            _pyopt_algorithms.pop(alg)

    class PyOptMinimizer(Minimizer):

        def __init__(self, function, parameters, ftol=1e-1, verbosity=10):

            super(PyOptMinimizer, self).__init__(function, parameters, ftol, verbosity)

        def _setup(self):

            self.functor = PyOptWrapper(self.function, self.Npar)

            self._opt_problem = pyOpt.Optimization('Minimum of -log(likelihood)', self.functor)

            self._opt_problem.addObj('f')

            # Add constraints for parameters

            for i, par in enumerate(self.parameters.values()):

                if par.min_value is not None and par.max_value is not None:

                    self._opt_problem.addVar(par.name, 'c', value=par.value, lower=par.min_value, upper=par.max_value)

                elif par.min_value is not None and par.max_value is None:

                    # Lower limited
                    self._opt_problem.addVar(par.name, 'c', value=par.value, lower=par.min_value, upper=np.inf)

                elif par.min_value is None and par.max_value is not None:

                    # upper limited
                    self._opt_problem.addVar(par.name, 'c', value=par.value, lower=-np.inf, upper=par.max_value)

                else:

                    # No limits
                    self._opt_problem.addVar(par.name, 'c', value=par.value, lower=-np.inf, upper=np.inf)

        def set_algorithm(self, algorithm_name):

            assert algorithm_name in _pyopt_algorithms, "Optimizer %s is not part of pyOpt" % algorithm_name

            # Create an instance of the provided optimizer

            self._algorithm_name = algorithm_name

            self._optimizer_instance = _pyopt_algorithms[algorithm_name]()

        def minimize(self, compute_covar=True):

            # Run optimization

            fstr, xstr, inform = self._optimizer_instance(self._opt_problem)

            # Transform to numpy.array

            best_fit_values = np.array(xstr)

            # Compute errors with the Hessian

            if compute_covar:

                covariance_matrix = self._compute_covariance_matrix(best_fit_values)

            else:

                covariance_matrix = None

            self._store_fit_results(best_fit_values, fstr, covariance_matrix)

            return best_fit_values, fstr


    _minimizers["PYOPT"] = PyOptMinimizer


if has_ROOT:

    class FuncWrapper(ROOT.TPyMultiGenFunction):

        def __init__(self, function, dimensions):

            ROOT.TPyMultiGenFunction.__init__(self, self)
            self.function = function
            self.dimensions = int(dimensions)

        def NDim(self):
            return self.dimensions

        def DoEval(self, args):

            new_args = map(lambda i:args[i],range(self.dimensions))

            return self.function(*new_args)


    class ROOTMinimizer(Minimizer):

        def __init__(self, function, parameters, ftol=1e3, verbosity=1):

            super(ROOTMinimizer, self).__init__(function, parameters, ftol, verbosity)

        def _setup(self):

            # Setup the minimizer algorithm

            self.functor = FuncWrapper(self.function, self.Npar)
            self.minimizer = ROOT.Math.Factory.CreateMinimizer("Minuit", "Minimize")
            self.minimizer.Clear()
            self.minimizer.SetMaxFunctionCalls(1000)
            self.minimizer.SetTolerance(0.1)
            self.minimizer.SetPrintLevel(self.verbosity)
            # self.minimizer.SetStrategy(0)

            self.minimizer.SetFunction(self.functor)

            for i, par in enumerate(self.parameters.values()):

                if par.min_value is not None and par.max_value is not None:

                    self.minimizer.SetLimitedVariable(i, par.name, par.value,
                                                      par.delta, par.min_value,
                                                      par.max_value)

                elif par.min_value is not None and par.max_value is None:

                    # Lower limited
                    self.minimizer.SetLowerLimitedVariable(i, par.name, par.value,
                                                           par.delta, par.min_value)

                elif par.min_value is None and par.max_value is not None:

                    # upper limited
                    self.minimizer.SetUpperLimitedVariable(i, par.name, par.value,
                                                           par.delta, par.max_value)

                else:

                    # No limits
                    self.minimizer.SetVariable(i, par.name, par.value, par.delta)

        def minimize(self, compute_covar=True):

            self.minimizer.SetPrintLevel(int(self.verbosity))

            self.minimizer.Minimize()

            best_fit_values = np.array(map(lambda x: x[0], zip(self.minimizer.X(), range(self.Npar))))

            if compute_covar:

                covariance_matrix = self._compute_covariance_matrix(best_fit_values)

            else:

                covariance_matrix = None

            minimum = self.functor(best_fit_values)

            self._store_fit_results(best_fit_values, minimum, covariance_matrix)

            return best_fit_values, minimum

    _minimizers["ROOT"] = ROOTMinimizer


class ContourWorker(object):

    def __init__(self, function, minuit_values, minuit_args, minuit_param_1, minuit_param_2, name_to_position):

        self._minuit_values = minuit_values

        # Update the values for the parameters with the best fit one

        for key, value in self._minuit_values.iteritems():
            minuit_args[key] = value

        # This is a likelihood
        minuit_args['errordef'] = 0.5

        # Disable printing by iminuit

        minuit_args['print_level'] = 0

        self._minuit_args = minuit_args

        # Store the name of the parameters

        self.minuit_param_1 = minuit_param_1
        self.minuit_param_2 = minuit_param_2

        # Store the function
        self._function = function

        # This is a dictionary which gives the ordinal place for a given parameter.
        # It is used in the corner case where the function has only two parameters,
        # to figure out which is the correct order

        self.name_to_position = name_to_position

    def _create_new_minuit_object(self, args):

        # Now create the new minimizer

        _contour_minuit = Minuit(self._function, **args)

        _contour_minuit.tol = 100

        return _contour_minuit

    def __call__(self, args):

        # Get the values for the parameters
        # If we are stepping in only one direction, value_2 will be nan

        value_1, value_2 = args

        # NOTE: unfortunately iminuit does not allow to change the value of a fixed parameter after
        # the creation of the Minuit class. Hence we need to create a new class each time,
        # which sucks

        # Create a copy of the init args for Minuit

        this_minuit_args = dict(self._minuit_args)

        # Now set the parameters under scrutiny to the current values

        this_minuit_args[self.minuit_param_1] = value_1

        if self.minuit_param_2 is not None:
            this_minuit_args[self.minuit_param_2] = value_2

        # Fix the parameters under scrutiny

        for minuit_name in [self.minuit_param_1, self.minuit_param_2]:

            if minuit_name is None:
                # Only one parameter to analyze

                continue

            if minuit_name not in this_minuit_args.keys():

                raise ParameterIsNotFree("Parameter %s is not a free parameter." % minuit_name)

            else:

                this_minuit_args['fix_%s' % minuit_name] = True

        # Finally create a new minimizer
        this_contour_minuit = self._create_new_minuit_object(this_minuit_args)

        # Handle the corner case where there are no free parameters
        # after fixing the two under scrutiny

        if len(this_contour_minuit.list_of_vary_param()) == 0:

            # All parameters are fixed, just return the likelihood function

            if self.minuit_param_2 is None:

                value = self._function(value_1)

            else:

                # This is needed because the user could specify the
                # variables in a different order than what is specified in the calling sequence
                # of f

                this_variables = [0, 0]
                this_variables[self.name_to_position[self.minuit_param_1]] = value_1
                this_variables[self.name_to_position[self.minuit_param_2]] = value_2

                value = self._function(*this_variables)

            return value

        try:

            this_contour_minuit.migrad()

        # In the following except I cannot catch specific exceptions because I don't exactly know which kind
        # of exception migrad can raise...

        except:

            # In this context this is not such a big deal,
            # because we might be so far from the minimum that
            # the fit cannot converge

            return FIT_FAILED

        return this_contour_minuit.fval
