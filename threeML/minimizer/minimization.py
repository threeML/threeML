import math
import collections
import re

import numpy
from iminuit import Minuit

from IPython.display import display
import uncertainties

from threeML.io.table import Table, NumericMatrix
from threeML.utils.cartesian import cartesian
from threeML.io.progress_bar import ProgressBar


# Special constants
FIT_FAILED = 1e12


# Define a bunch of custom exceptions relevant for what is being accomplished here

class CannotComputeCovariance(Exception):
    pass


class CannotComputeErrors(Exception):
    pass


class MINOSFailed(Exception):
    pass


class ParameterIsNotFree(Exception):
    pass


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

    def minimize(self):
        raise NotImplemented("This is the method of the base class. Must be implemented by the actual minimizer")


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
    # to switch back to the SEAL minuit

    def __init__(self, function, parameters, ftol=1e3, verbosity=0):

        super(MinuitMinimizer, self).__init__(function, parameters, ftol, verbosity)

        # Prepare the dictionary for the parameters which will be used by iminuit

        iminuit_init_parameters = {}

        # List of variable names that will be used for iminuit. We need to translate the dictionary of dictionaries
        # parameters into a list which will be fed to iminuit to be used as parameters for the fit

        variable_names_for_iminuit = []

        for k, par in parameters.iteritems():
            current_name = "%s_of_%s" % (k[1], k[0])

            variable_names_for_iminuit.append(current_name)

            # Initial value
            iminuit_init_parameters['%s' % current_name] = par.value

            # Initial delta
            iminuit_init_parameters['error_%s' % current_name] = par.delta

            # Limits
            iminuit_init_parameters['limit_%s' % current_name] = (par.minValue, par.maxValue)

            # This is useless, since all parameters here are free,
            # but do it anyway for clarity
            iminuit_init_parameters['fix_%s' % current_name] = False

        # This is to tell Minuit that we are dealing with likelihoods,
        # not chi square
        iminuit_init_parameters['errordef'] = 0.5

        iminuit_init_parameters['print_level'] = verbosity

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

        self.minuit.tol = ftol  # ftol

        try:

            self.minuit.up = 0.5  # This is a likelihood

        except AttributeError:

            # iMinuit uses errodef, not up

            self.minuit.errordef = 0.5

        self.minuit.strategy = 1  # More accurate

    def _migrad_has_converged(self):

        # In the MINUIT manual this is the condition for MIGRAD to have converged
        # 0.002 * tolerance * UPERROR (which is 0.5 for likelihood)

        return self.minuit.edm <= 0.002 * self.minuit.tol * 0.5

    def _run_migrad(self, trials=10):

        # Repeat Migrad up to trials times, until it converges

        for i in range(trials):

            self.minuit.migrad()

            if self._migrad_has_converged():

                # Converged
                break

            else:

                # Try again
                continue

    def minimize(self):
        """
        Minimize the function using MIGRAD

        :return: best_fit: a dictionary containing the parameters at their best fit values
                 function_minimum : the value for the function at the minimum

                 NOTE: if the minimization fails, the dictionary will be empty and the function_minimum will be set
                 to minimization.FIT_FAILED
        """

        self._run_migrad(10)

        if not self._migrad_has_converged():

            print("\nMIGRAD did not converge in 10 trials.")

            return collections.OrderedDict(), FIT_FAILED

        else:

            # Make a ordered dict for the results

            best_fit_parameters = collections.OrderedDict()

            for k, par in self.parameters.iteritems():
                current_name = "%s_of_%s" % (k[1], k[0])

                best_fit_parameters[current_name] = self.minuit.values[current_name]

            # NOTE: hesse must be called AFTER having stored the parameters because it
            # will change the value of the parameters

            self.minuit.hesse()

            # Restore parameters to their best fit

            for k, par in self.parameters.iteritems():
                current_name = "%s_of_%s" % (k[1], k[0])

                par.setValue(best_fit_parameters[current_name])

            value_at_the_minimum = self.minuit.fval

            return best_fit_parameters, value_at_the_minimum

    def print_fit_results(self):
        """
        Display the results of the last minimization.

        :return: (none)
        """

        # I do not use the print_param facility in iminuit because
        # it does not work well with console output, since it fails
        # to autoprobe that it is actually run in a console and uses
        # the HTML backend instead

        # Create a list of strings to print

        data = []

        # Also store the maximum length to decide the length for the line

        name_length = 0

        for k, v in self.parameters.iteritems():

            current_name = "%s_of_%s" % (k[1], k[0])

            # Format the value and the error with sensible significant
            # numbers
            x = uncertainties.ufloat(v.value, self.minuit.errors[current_name])

            # Add some space around the +/- sign

            rep = x.__str__().replace("+/-", " +/- ")

            data.append([current_name, rep, v.unit])

            if len(v.name) > name_length:
                name_length = len(current_name)

        table = Table(rows=data,
                      names=["Name", "Value", "Unit"],
                      dtype=('S%i' % name_length, str, 'S15'))

        display(table)

        print("\nNOTE: errors on parameters are approximate. Use get_errors().\n")

    def print_correlation_matrix(self):
        """
        Display the current correlation matrix
        :return: (none)
        """

        # Print a custom covariance matrix because iminuit does
        # not guess correctly the frontend when 3ML is used
        # from terminal

        cov = self.minuit.covariance

        if cov is None:
            raise CannotComputeCovariance("Cannot compute covariance numerically. This usually means that there are " +
                                          " unconstrained parameters. Fix those or reduce their allowed range, or " +
                                          "use a simpler model.")

        # Get list of parameters

        keys = self.parameters.keys()

        # Convert them to the format for iminuit

        parameter_names = map(lambda k: "%s_of_%s" % (k[1], k[0]), keys)

        # Accumulate rows and compute the maximum length of the names

        data = []
        length_of_names = 0

        for key1, name1 in zip(keys, parameter_names):

            if len(name1) > length_of_names:
                length_of_names = len(name1)

            this_row = []

            for key2, name2 in zip(keys, parameter_names):
                # Compute correlation between parameter key1 and key2

                corr = cov[(name1, name2)] / (math.sqrt(cov[(name1, name1)]) * math.sqrt(cov[(name2, name2)]))

                this_row.append(corr)

            data.append(this_row)

        # Prepare the dtypes for the matrix

        dtypes = map(lambda x: float, parameter_names)

        # Column names are the parameter names

        cols = parameter_names

        # Finally generate the matrix with the names

        table = NumericMatrix(rows=data,
                              names=cols,
                              dtype=dtypes)

        # Customize the format to avoid too many digits

        for col in table.colnames:
            table[col].format = '2.2f'

        display(table)

    def get_errors(self):
        """
        Compute asymmetric errors using MINOS (slow, but accurate) and print them.

        :return: a dictionary containing the asymmetric errors for each parameter.
        """

        # Run again the fit because the user might have changed the parameter
        # configuration

        self._run_migrad()

        # Now set aside the current values for the parameters,
        # because minos will change them
        # Make a ordered dict for the results

        best_fit_parameters = collections.OrderedDict()

        for k, par in self.parameters.iteritems():
            current_name = "%s_of_%s" % (k[1], k[0])

            best_fit_parameters[current_name] = self.minuit.values[current_name]

        if not self._migrad_has_converged():
            raise CannotComputeErrors("MIGRAD results not valid, cannot compute errors. Did you run the fit first ?")

        try:

            self.minuit.minos()

        except:

            raise MINOSFailed("MINOS has failed. This usually means that the fit is very difficult, for example "
                              "because of high correlation between parameters. Check the correlation matrix printed"
                              "in the fit step, and check contour plots with getContours(). If you are using a "
                              "user-defined model, you can also try to "
                              "reformulate your model with less correlated parameters.")

        # Make a ordered dict for the results

        errors = collections.OrderedDict()

        for k, par in self.parameters.iteritems():
            current_name = "%s_of_%s" % (k[1], k[0])

            errors[current_name] = (self.minuit.merrors[(current_name, -1)], self.minuit.merrors[(current_name, 1)])

            # Set the parameter back to the best fit value
            par.setValue(best_fit_parameters[current_name])

        # Print a table with the errors

        data = []
        name_length = 0

        for k, v in self.parameters.iteritems():

            current_name = "%s_of_%s" % (k[1], k[0])

            # Format the value and the error with sensible significant
            # numbers

            # Process the negative error

            x = uncertainties.ufloat(v.value, abs(errors[current_name][0]))

            # Split the uncertainty in number, negative error, and exponent (if any)

            num, uncm, exponent = re.match('\(?(\-?[0-9]+\.?[0-9]+) ([0-9]+\.[0-9]+)\)?(e[\+|\-][0-9]+)?',
                                           x.__str__().replace("+/-", " ")).groups()

            # Process the positive error

            x = uncertainties.ufloat(v.value, abs(errors[current_name][1]))

            # Split the uncertainty in number, positive error, and exponent (if any)

            _, uncp, _ = re.match('\(?(\-?[0-9]+\.?[0-9]+) ([0-9]+\.[0-9]+)\)?(e[\+|\-][0-9]+)?',
                                  x.__str__().replace("+/-", " ")).groups()

            if exponent is None:

                # Number without exponent

                pretty_string = "%s -%s +%s" % (num, uncm, uncp)

            else:

                # Number with exponent

                pretty_string = "(%s -%s +%s)%s" % (num, uncm, uncp, exponent)

            data.append([current_name, pretty_string, v.unit])

            if len(v.name) > name_length:
                name_length = len(current_name)

        # Create and display the table

        table = Table(rows=data,
                      names=["Name", "Value", "Unit"],
                      dtype=('S%i' % name_length, str, 'S15'))

        display(table)

        return errors

    def contours(self, source_1, param_1, param_1_minimum, param_1_maximum, param_1_n_steps,
                 source_2=None, param_2=None, param_2_minimum=None, param_2_maximum=None, param_2_n_steps=None,
                 progress=True, **options):
        """
        Generate confidence contours for the given parameters by stepping for the given number of steps between
        the given boundaries. Call it specifying only source_1, param_1, param_1_minimum and param_1_maximum to
        generate the profile of the likelihood for parameter 1. Specify all parameters to obtain instead a 2d
        contour of param_1 vs param_2

        :param source_1: name of the source for the first parameter
        :param param_1: name of the first parameter
        :param param_1_minimum: lower bound for the range for the first parameter
        :param param_1_maximum: upper bound for the range for the first parameter
        :param param_1_n_steps: number of steps for the first parameter
        :param source_2: name of the source for the second parameter
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
        :return: a : an array corresponding to the steps for the first parameter
                 b : an array corresponding to the steps for the second parameter (or None if stepping only in one
                 direction)
                 contour : a matrix of size param_1_steps x param_2_steps containing the value of the function at the
                 corresponding points in the grid. If param_2_steps is None (only one parameter), then this reduces to
                 an array of size param_1_steps.
        """

        # Figure out if we are making a 1d or a 2d contour

        if source_2 is None:

            n_dimensions = 1

        else:

            n_dimensions = 2

        # Check the options

        p1log = False
        p2log = False

        if 'log' in options.keys():

            assert len(options['log']) == n_dimensions, ("When specifying the 'log' option you have to provide a " +
                                                         "boolean for each dimension you are stepping on.")

            p1log = bool(options['log'][0])

            if param_2 is not None:
                p2log = bool(options['log'][1])

        # First create another minimizer to avoid messing up with the existing one

        # Duplicate the options used for the original minimizer

        new_args = dict(self.minuit.fitarg)

        # Update the values for the parameters with the best fit one

        for key, value in self.minuit.values.iteritems():
            new_args[key] = value

        # Fix the parameters under scrutiny

        for s, p in zip([source_1, source_2], [param_1, param_2]):

            if s is None:
                # Only one parameter to analyze

                continue

            key = "%s_of_%s" % (p, s)

            if key not in new_args.keys():

                raise ParameterIsNotFree("Parameter %s is not a free parameter for source %s." % (p, s))

            else:

                new_args['fix_%s' % key] = True

        # This is a likelihood
        new_args['errordef'] = 0.5

        # Disable printing by iminuit

        new_args['print_level'] = 0

        # Now create the new minimizer

        self._contour_minuit = Minuit(self._f, **new_args)

        # Generate the steps

        if p1log:

            param_1_steps = numpy.logspace(math.log10(param_1_minimum), math.log10(param_1_maximum),
                                           param_1_n_steps)

        else:

            param_1_steps = numpy.linspace(param_1_minimum, param_1_maximum,
                                           param_1_n_steps)

        if n_dimensions == 2:

            if p2log:

                param_2_steps = numpy.logspace(math.log10(param_2_minimum), math.log10(param_2_maximum),
                                               param_2_n_steps)

            else:

                param_2_steps = numpy.linspace(param_2_minimum, param_2_maximum,
                                               param_2_n_steps)

        else:

            # Only one parameter to step through
            # Put param_2_steps as nan so that the worker can realize that it does not have
            # to step through it

            param_2_steps = numpy.array([numpy.nan])

        # Generate the grid

        grid = cartesian([param_1_steps, param_2_steps])

        # Define the worker which will compute the value of the function at a given point in the grid

        def contour_worker(args):

            # Get the point coordinates (aa,bb)
            # If we are stepping in only one direction, bb will be nan

            aa, bb = args

            # Get name of the parameter

            name1 = "%s_of_%s" % (param_1, source_1)

            # Will change this if needed

            name2 = None

            # First of all restore the best fit values
            # for k,v in values.iteritems():

            #    self.minuit.values[ k ] = v

            # Now set the parameters under scrutiny to the current values

            # Since iminuit does not allow to fix parameters after initialization,
            # I am forced to create a new minimizer (which sucks)

            newargs = dict(self.minuit.fitarg)

            newargs['fix_%s' % name1] = True
            newargs['%s' % name1] = aa

            if numpy.isfinite(bb):

                # Stepping in two directions

                name2 = "%s_of_%s" % (param_2, source_2)

                newargs['fix_%s' % name2] = True
                newargs['%s' % name2] = bb

            else:

                # We are stepping through one param only.
                # Do nothing

                pass

            newargs['errordef'] = 0.5
            newargs['print_level'] = 0

            m = Minuit(self._f, **newargs)

            # High tolerance for speed
            m.tol = 100

            # mpl.warning("Running migrad")

            # Handle the corner case where there are no free parameters
            # after fixing the two under scrutiny

            if len(m.list_of_vary_param()) == 0:

                # All parameters are fixed, just return the likelihood function

                if name2 is None:

                    val = self._f(aa)

                else:

                    # This is needed because the user could specify the
                    # variables in a different order than what is specified in the calling sequence
                    # of f

                    myvars = [0, 0]
                    myvars[self.name_to_position[name1]] = aa
                    myvars[self.name_to_position[name2]] = bb

                    val = self._f(*myvars)

                return val

            try:

                m.migrad()

            # In the following except I cannot catch specific exceptions because I don't exactly know which kind
            # of exception migrad can raise...

            except:

                # In this context this is not such a big deal,
                # because we might be so far from the minimum that
                # the fit cannot converge

                return FIT_FAILED

            return m.fval

        # We are finally ready to do the computation

        if progress:

            # Computation with progress bar

            progress_bar = ProgressBar(grid.shape[0])

            # Define a wrapper which will increase the progress before as well as run the actual computation

            def wrap(args):

                results = contour_worker(args)

                progress_bar.increase()

                return results

            # Do the computation

            r = map(wrap, grid)

        else:

            # Computation without the progress bar

            r = map(contour_worker, grid)

        # Return results

        return param_1_steps, param_2_steps, numpy.array(r).reshape((param_1_steps.shape[0], param_2_steps.shape[0]))
