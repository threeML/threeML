import ROOT
import numpy as np

from threeML.minimizer.minimization import Minimizer, FIT_FAILED


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

    def __init__(self, function, parameters, ftol=1e3, verbosity=0):

        super(ROOTMinimizer, self).__init__(function, parameters, ftol, verbosity)

    def _setup(self):

        # Setup the minimizer algorithm

        self.functor = FuncWrapper(self.function, self.Npar)
        self.minimizer = ROOT.Math.Factory.CreateMinimizer("Minuit2", "Minimize")
        self.minimizer.Clear()
        self.minimizer.SetMaxFunctionCalls(1000)
        self.minimizer.SetTolerance(0.1)
        self.minimizer.SetPrintLevel(self.verbosity)
        # self.minimizer.SetStrategy(0)

        self.minimizer.SetFunction(self.functor)

        for i, par in enumerate(self.parameters.values()):

            if par.min_value is not None and par.max_value is not None:

                self.minimizer.SetLimitedVariable(i, par.path, par.value,
                                                  par.delta, par.min_value,
                                                  par.max_value)

            elif par.min_value is not None and par.max_value is None:

                # Lower limited
                self.minimizer.SetLowerLimitedVariable(i, par.path, par.value,
                                                       par.delta, par.min_value)

            elif par.min_value is None and par.max_value is not None:

                # upper limited
                self.minimizer.SetUpperLimitedVariable(i, par.path, par.value,
                                                       par.delta, par.max_value)

            else:

                # No limits
                self.minimizer.SetVariable(i, par.path, par.value, par.delta)

    def minimize(self, compute_covar=True):

        self.minimizer.SetPrintLevel(int(self.verbosity))

        self.minimizer.Minimize()

        best_fit_values = np.array(map(lambda x: x[0], zip(self.minimizer.X(), range(self.Npar))))

        if compute_covar:

            self.minimizer.Hesse()

            # The ROOT Minimizer instance already got the covariance matrix,
            # we just need to copy it

            covariance_matrix = np.zeros((self.Npar, self.Npar))

            for i in range(self.Npar):

                for j in range(self.Npar):

                    covariance_matrix[i,j] = self.minimizer.CovMatrix(i,j)

        else:

            covariance_matrix = None

        minimum = self.minimizer.MinValue()

        self._store_fit_results(best_fit_values, minimum, covariance_matrix)

        return best_fit_values, minimum

