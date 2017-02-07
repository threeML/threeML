import PyGMO
from PyGMO.problem import base
import numpy as np
import dill
import sys

from threeML.minimizer.minimization import Minimizer


class WrapperUnpickler(object):

    def __call__(self, dill_package, dim):

        function = dill.loads(dill_package)

        return PAGMOWrapper(function, dim=dim)


class PAGMOWrapper(base):

    def __init__(self, function=None, parameters=None, dim=1):

        super(PAGMOWrapper, self).__init__(dim)

        self.__dim = dim

        self._objective_function = function

        if parameters is not None:

            minima = []
            maxima = []

            for param in parameters.values():

                min_val, max_val = param.bounds

                if min_val is None or max_val is None:

                    raise RuntimeError("In order to use the PAGMO minimizer, you have to provide a minimum and a "
                                       "maximum for all parameters in the model.")

                minima.append(min_val)
                maxima.append(max_val)

            self.set_bounds(minima, maxima)

            self._minima = minima
            self._maxima = maxima

    # Reimplement the virtual method that defines the objective function.
    def _objfun_impl(self, x):

        val = self._objective_function(*x)

        # Note that we return a tuple with one element only. In PyGMO the objective functions
        # return tuples so that multi-objective optimization is also possible.
        return (val,)

    # Finally we also reimplement a virtual method that adds some output to the __repr__ method
    def human_readable_extra(self):
        return "\n\t Problem dimension: " + str(self.__dim)

    def __reduce__(self):

        dill_package = dill.dumps(self._objective_function)

        state = {'minima': self._minima, 'maxima': self._maxima}

        return WrapperUnpickler(), (dill_package,self.__dim), state

    def __setstate__(self, state):

        self._minima = state['minima']
        self._maxima = state['maxima']

        self.set_bounds(self._minima, self._maxima)


class PAGMOMinimizer(Minimizer):

    def __init__(self, function, parameters, ftol=1e-3, verbosity=10):

        super(PAGMOMinimizer, self).__init__(function, parameters, ftol, verbosity)

        self._max_evolutions = 10000
        self._evolution_step = 20

    def _setup(self):

        pass

    def set_algorithm(self, algorithm_instance):

        assert isinstance(algorithm_instance, PyGMO.algorithm._algorithm._base), "The algorithm must be an " \
                                                                                 "instance of a PyGMO algorithm"

        # Create an instance of the provided optimizer

        self._algorithm_name = algorithm_instance.__class__.__name__

        # By default the population number is 5 times the number of parameters

        self._algorithm = algorithm_instance

    def minimize(self, compute_covar=True):

        try:

            best_fit_values, final_value = evolve(self.function,
                                                  self.parameters,
                                                  self._algorithm,
                                                  evolution_step=self._evolution_step,
                                                  max_evolutions=self._max_evolutions,
                                                  ftol=self.ftol)

        except:

            raise

            exc_type, message, _ = sys.exc_info()

            raise RuntimeError("Could not evolve the population. Exc. type: %s, message: %s" % (exc_type, message))

        # Compute errors with the Hessian

        if compute_covar:

            covariance_matrix = self._compute_covariance_matrix(best_fit_values)

        else:

            covariance_matrix = None

        self._store_fit_results(best_fit_values, final_value, covariance_matrix)

        return best_fit_values, final_value

# This cannot be part of a class, unfortunately, because of how PyGMO serialize objects

def evolve(function, parameters, algorithm, evolution_step=20, max_evolutions=1000, ftol=1e-3):

    Npar = len(parameters)

    functor = PAGMOWrapper(function=function, parameters=parameters, dim=Npar)

    _island = PyGMO.island(algorithm, functor, Npar * 5)

    # Get initial value
    initial_value = _island.population.champion.f[0]

    # Keep evolving the population until the final value does not change by more than ftol

    previous_iter_value = initial_value

    final_value = 0

    for i in xrange(max_evolutions):

        # Evolve the population a certain number of times

        _island.evolve(evolution_step)

        # Check if we have improved, if not break out of the loop

        current_iter_value = _island.population.champion.f[0]

        if abs(previous_iter_value - current_iter_value) < ftol:

            # Converged
            break

        else:

            # Continue evolving
            previous_iter_value = current_iter_value

    _island.join()
    xOpt = _island.population.champion.x
    fOpt = _island.population.champion.f[0]

    # Transform to numpy.array

    best_fit_values = np.array(xOpt)

    return best_fit_values, fOpt