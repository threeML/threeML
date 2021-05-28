from __future__ import print_function

import collections
from builtins import range

import numpy as np
from iminuit import Minuit

from threeML.io.logging import setup_logger
from threeML.minimizer.minimization import (CannotComputeCovariance,
                                            CannotComputeErrors, FitFailed,
                                            LocalMinimizer)

log = setup_logger(__name__)


class MINOSFailed(Exception):
    pass


# This is a function to add a method to a class
# We will need it in the MinuitMinimizer


def add_method(self, method, name=None):
    if name is None:
        name = method.__name__

    setattr(self.__class__, name, method)


class MinuitMinimizer(LocalMinimizer):

    valid_setup_keys = ("ftol",)

    # @TODO: Is this still relevant?
    # NOTE: this class is built to be able to work both with iMinuit and with a boost interface to SEAL
    # minuit, i.e., it does not rely on functionality that iMinuit provides which is not of the original
    # minuit. This makes the implementation a little bit more cumbersome, but more adaptable if we want
    # to switch back to the bare bone SEAL minuit

    def __init__(self, function, parameters, verbosity=0, setup_dict=None):

        # This will contain the results of the last call to Migrad
        self._last_migrad_results = None

        super(MinuitMinimizer, self).__init__(
            function, parameters, verbosity, setup_dict
        )

    def _setup(self, user_setup_dict):

        # Prepare the dictionaries for the parameters which will be used by iminuit

        iminuit_init_parameters = collections.OrderedDict()

        iminuit_errors = collections.OrderedDict()

        iminuit_limits = collections.OrderedDict()

        iminuit_fixed_parameters = collections.OrderedDict()

        # List of variable names that will be used for iminuit

        variable_names_for_iminuit = []

        # NOTE: we use the internal_ versions of value, min_value and max_value because they don't have
        # units, and they are transformed to make the fit easier (for example in log scale)

        for (
            parameter_path,
            (value, delta, minimum, maximum),
        ) in self._internal_parameters.items():

            current_name = self._parameter_name_to_minuit_name(parameter_path)

            variable_names_for_iminuit.append(current_name)

            # Initial value
            iminuit_init_parameters["%s" % current_name] = value

            # Initial delta
            iminuit_errors["%s" % current_name] = delta

            # Limits
            iminuit_limits["%s" % current_name] = (minimum, maximum)

            # This is useless, since all parameters here are free,
            # but do it anyway for clarity
            iminuit_fixed_parameters["%s" % current_name] = False

        # Tell imnuit what parameter names are
        iminuit_init_parameters["name"] = variable_names_for_iminuit

        # Finally we can instance the Minuit class

        self.minuit = Minuit(self.function, **iminuit_init_parameters)

        for param, value in iminuit_errors.items():
            self.minuit.errors[param] = value

        for param, value in iminuit_limits.items():
            self.minuit.limits[param] = value

        for param, value in iminuit_fixed_parameters.items():
            self.minuit.fixed[param] = value

        # This is to tell Minuit that we are dealing with likelihoods,
        # not chi square

        self.minuit.errordef = Minuit.LIKELIHOOD

        self.minuit.print_level = self.verbosity

        if user_setup_dict is not None:

            if "ftol" in user_setup_dict:

                self.minuit.tol = user_setup_dict["ftol"]

        else:

            # Do nothing and leave the default in iminuit
            pass

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

    # Override this because minuit uses different names
    def restore_best_fit(self):
        """
        Set the parameters back to their best fit value

        :return: none
        """

        # Reset the internal value of all parameters

        super(MinuitMinimizer, self).restore_best_fit()

        # Update also the internal iminuit dictionary

        for k, par in self.parameters.items():

            minuit_name = self._parameter_name_to_minuit_name(k)

            self.minuit.values[minuit_name] = par._get_internal_value()

    def _print_current_status(self):
        """
        To be used to print info before raising an exception
        :return:
        """

        log.error("Last status:")

        for line in str(self._last_migrad_results.fmin).splitlines():
            log.error(line)

        # Print params to get some info about the failure

        for line in str(self.minuit.params).splitlines():
            log.error(line)

        

    def _minimize(self):
        """
        Minimize the function using MIGRAD

        :param compute_covar: whether to compute the covariance (and error estimates) or not
        :return: best_fit: a dictionary containing the parameters at their best fit values
                 function_minimum : the value for the function at the minimum

                 NOTE: if the minimization fails, the dictionary will be empty and the function_minimum will be set
                 to minimization.FIT_FAILED
        """

        # Try a maximum of 10 times and break as soon as the fit is ok

        self.minuit.reset()
        self._last_migrad_results = self.minuit.migrad()

        for i in range(9):

            if self.minuit.valid:

                break

            else:

                # Try again
                self._last_migrad_results = self.minuit.migrad()

        if not self.minuit.valid:

            self._print_current_status()

            raise FitFailed(
                "MIGRAD call failed. This is usually due to unconstrained parameters."
            )

        else:

            # Gather the optimized values for all parameters from the internal
            # iminuit dictionary

            best_fit_values = []

            for k, par in self.parameters.items():

                minuit_name = self._parameter_name_to_minuit_name(k)

                best_fit_values.append(self.minuit.values[minuit_name])

            return best_fit_values, self.minuit.fval

    # Override the default _compute_covariance_matrix
    def _compute_covariance_matrix(self, best_fit_values):

        self.minuit.hesse()

        try:

            covariance = np.array(self.minuit.covariance)

        except RuntimeError:

            # Covariance computation has failed

            # Print current status
            self._print_current_status()

            log.error(
                "HESSE failed. Most probably some of your parameters are unconstrained.")

            raise CannotComputeCovariance(

            )

        return covariance

    def get_errors(self):
        """
        Compute asymmetric errors using MINOS (slow, but accurate) and print them.

        NOTE: this should be called immediately after the minimize() method

        :return: a dictionary containing the asymmetric errors for each parameter.
        """

        self.restore_best_fit()

        if not self.minuit.valid:

            raise CannotComputeErrors(
                "MIGRAD results not valid, cannot compute errors."
            )

        try:

            self.minuit.minos()

        except:

            self._print_current_status()

            raise MINOSFailed(
                "MINOS has failed. This is not necessarily a problem if:\n\n"
                "* There are unconstrained parameters (the error is undefined). This is usually signaled "
                "by an approximated error, printed after the fit, larger than the best fit value\n\n"
                "* The fit is very difficult, because of high correlation between parameters. This is "
                "signaled by values close to 1.0 or -1.0 in the correlation matrix printed after the "
                "fit step.\n\n"
                "In this cases you can check the contour plots with get_contours(). If you are using a "
                "user-defined model, you can also try to reformulate your model with less correlated "
                "parameters."
            )

        # Make a list for the results

        errors = collections.OrderedDict()

        for k, par in self.parameters.items():

            minuit_name = self._parameter_name_to_minuit_name(k)

            minus_error = self.minuit.merrors[minuit_name].lower
            plus_error = self.minuit.merrors[minuit_name].upper

            if par.has_transformation():

                # Need to transform in the external reference

                best_fit_value_internal = self._fit_results.loc[par.path, "value"]

                _, minus_error_external = par.internal_to_external_delta(
                    best_fit_value_internal, minus_error
                )

                _, plus_error_external = par.internal_to_external_delta(
                    best_fit_value_internal, plus_error
                )

            else:

                minus_error_external = minus_error
                plus_error_external = plus_error

            errors[k] = (minus_error_external, plus_error_external)

        return errors
