from builtins import zip
from builtins import range
import ROOT
import numpy as np
import ctypes

from threeML.minimizer.minimization import (
    LocalMinimizer,
    FitFailed,
    CannotComputeCovariance,
)
from threeML.io.dict_with_pretty_print import DictWithPrettyPrint

# These are the status returned by Minuit
#     status = 1    : Covariance was made pos defined
#     status = 2    : Hesse is invalid
#     status = 3    : Edm is above max
#     status = 4    : Reached call limit
#     status = 5    : Any other failure
_status_translation = {
    1: "Covariance was made pos. defined",
    2: "Hesse is invalid",
    3: "Edm is above maximum",
    4: "Reached call limit",
    5: "Unknown failure",
}

# Status for HESSE
# status += 100*hesseStatus where hesse status is:
# status = 1 : hesse failed
# status = 2 : matrix inversion failed
# status = 3 : matrix is not pos defined
_hesse_status_translation = {
    100: "HESSE failed",
    200: "Covariance matrix inversion failed",
    300: "Covariance matrix is not positive defined",
}

#root_class = None
#try:
#    root_class = ROOT.TPyMultiGenFunction
#except AttributeError:
#    root_class = ROOT.Math.IMultiGenFunction

class FuncWrapper(ROOT.Math.IMultiGenFunction):
    
    def setup(self, function, dimensions):
        self.function = function
        self.dimensions = int(dimensions)

    def NDim(self):
        return self.dimensions

    def DoEval(self, args):
        new_args = [args[i] for i in range(self.dimensions)]
        return self.function(*new_args)
    
    def Clone(self):
        f = FuncWrapper()
        f.setup(f.function, f.dimensions)
        ROOT.SetOwnership(f, False)
        return f


class ROOTMinimizer(LocalMinimizer):

    valid_setup_keys = ("ftol", "max_function_calls", "strategy")

    def __init__(self, function, parameters, verbosity=0, setup_dict=None):

        super(ROOTMinimizer, self).__init__(function, parameters, verbosity, setup_dict)

    def _setup(self, user_setup_dict):

        # Defaults

        setup_dict = {"ftol": 1.0, "max_function_calls": 100000, "strategy": 1}

        # Update defaults if needed
        if user_setup_dict is not None:

            for key in user_setup_dict:

                setup_dict[key] = user_setup_dict[key]

        # Setup the minimizer algorithm

        self.functor = FuncWrapper()
        self.functor.setup(self.function, self.Npar)
        self.minimizer = ROOT.Minuit2.Minuit2Minimizer("Minimize")
        self.minimizer.Clear()
        self.minimizer.SetMaxFunctionCalls(setup_dict["max_function_calls"])
        self.minimizer.SetPrintLevel(self.verbosity)
        self.minimizer.SetErrorDef(0.5)
        self.minimizer.SetStrategy(setup_dict["strategy"])
        self.minimizer.SetTolerance(setup_dict["ftol"])

        self.minimizer.SetFunction(self.functor)
        self.minimizer.SetPrintLevel(int(self.verbosity))

        # Set up the parameters in internal reference

        for i, (par_name, (cur_value, cur_delta, cur_min, cur_max)) in enumerate(
            self._internal_parameters.items()
        ):

            if cur_min is not None and cur_max is not None:

                # Variable with lower and upper limit

                self.minimizer.SetLimitedVariable(
                    i, par_name, cur_value, cur_delta, cur_min, cur_max
                )

            elif cur_min is not None and cur_max is None:

                # Lower limited
                self.minimizer.SetLowerLimitedVariable(
                    i, par_name, cur_value, cur_delta, cur_min
                )

            elif cur_min is None and cur_max is not None:

                # upper limited
                self.minimizer.SetUpperLimitedVariable(
                    i, par_name, cur_value, cur_delta, cur_max
                )

            else:

                # No limits
                self.minimizer.SetVariable(i, par_name, cur_value, cur_delta)

    def _minimize(self, compute_covar=True):

        # Minimize with MIGRAD

        success = self.minimizer.Minimize()

        if not success:

            # Get status
            status = self.minimizer.Status()

            if status in _status_translation:

                msg = "MIGRAD did not converge. Reason: %s (status: %i)" % (
                    _status_translation[status],
                    status,
                )

            else:

                msg = (
                    "MIGRAD failed with status %i "
                    "(see https://root.cern.ch/root/html/ROOT__Minuit2__Minuit2Minimizer.html)"
                    % status
                )

            raise FitFailed(msg)

        # Gather results

        minimum = self.minimizer.MinValue()

        best_fit_values = np.array(
            [x[0] for x in zip(self.minimizer.X(), list(range(self.Npar)))]
        )

        return best_fit_values, minimum

    def _compute_covariance_matrix(self, best_fit_values):

        # Gather the current status so we can offset it later
        status_before_hesse = self.minimizer.Status()

        # Use Hesse to compute the covariance matrix accurately

        self.minimizer.Hesse()

        # Gather the current status and remove the offset so that we get the HESSE status
        status_after_hesse = self.minimizer.Status() - status_before_hesse

        if status_after_hesse > 0:

            failure_reason = _hesse_status_translation[status_after_hesse]

            raise CannotComputeCovariance(
                "HESSE failed. Reason: %s (status: %i)"
                % (failure_reason, status_after_hesse)
            )

        # Gather the covariance matrix and return it

        covariance_matrix = np.zeros((self.Npar, self.Npar))

        for i in range(self.Npar):

            for j in range(self.Npar):

                covariance_matrix[i, j] = self.minimizer.CovMatrix(i, j)

        return covariance_matrix

    def _get_errors(self):

        # Re-implement this in order to use MINOS

        errors = DictWithPrettyPrint()

        for i, par_name in enumerate(self.parameters):

            err_low = ctypes.c_double(0)
            err_up = ctypes.c_double(0)

            self.minimizer.GetMinosError(i, err_low, err_up)

            errors[par_name] = (err_low.value, err_up.value)

        return errors

        # GetMinosError(unsigned
        # int
        # i, double & errLow, double & errUp, int = 0)
