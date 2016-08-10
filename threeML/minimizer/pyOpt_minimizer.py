from threeML.minimizer.minimization import Minimizer, FIT_FAILED
from astromodels import SettingOutOfBounds

import pyOpt
import numpy as np


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



