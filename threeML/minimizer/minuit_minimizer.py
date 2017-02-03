from threeML.minimizer.minimization import Minimizer, FIT_FAILED, CannotComputeErrors

from iminuit import Minuit
import collections
import numpy as np


class MINOSFailed(Exception):
    pass


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

    def __init__(self, function, parameters, ftol=0.1, verbosity=0):

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

        #self.minuit.tol = self.ftol  # ftol

        try:

            self.minuit.up = 0.5  # This is a likelihood

        except AttributeError:

            # iMinuit uses errodef, not up

            self.minuit.errordef = 0.5

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

            raise MINOSFailed("MINOS has failed. This is not necessarily a problem if:\n\n"
                              "* There are unconstrained parameters (the error is undefined). This is usually signaled "
                              "by an approximated error, printed after the fit, larger than the best fit value\n\n"
                              "* The fit is very difficult, because of high correlation between parameters. This is "
                              "signaled by values close to 1.0 or -1.0 in the correlation matrix printed after the "
                              "fit step.\n\n"
                              "In this cases you can check the contour plots with get_contours(). If you are using a "
                              "user-defined model, you can also try to reformulate your model with less correlated "
                              "parameters.")

        # Make a list for the results

        errors = collections.OrderedDict()

        for k, par in self.parameters.iteritems():

            minuit_name = self._parameter_name_to_minuit_name(k)

            minus_error = self.minuit.merrors[(minuit_name, -1)]
            plus_error = self.minuit.merrors[(minuit_name, 1)]

            errors[k] = ((minus_error,plus_error))

        return errors

