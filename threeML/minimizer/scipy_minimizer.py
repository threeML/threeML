from builtins import zip

import numpy as np
import numba as nb

import scipy.optimize

from threeML.io.logging import setup_logger
from threeML.minimizer.minimization import FitFailed, LocalMinimizer
from threeML.utils.differentiation import get_jacobian

log = setup_logger(__name__)


_SUPPORTED_ALGORITHMS = ["L-BFGS-B", "TNC", "SLSQP"]


class ScipyMinimizer(LocalMinimizer):

    valid_setup_keys = ("tol", "algorithm")

    def __init__(self, function, parameters, verbosity=10, setup_dict=None):

        super(ScipyMinimizer, self).__init__(
            function, parameters, verbosity, setup_dict
        )

    def _setup(self, user_setup_dict):

        if user_setup_dict is None:

            default_setup = {"algorithm": "L-BFGS-B", "tol": 0.0001}

            
            
            self._setup_dict = default_setup

        else:

            if "algorithm" in user_setup_dict:

                
                if not user_setup_dict["algorithm"] in _SUPPORTED_ALGORITHMS:

                    log.error("Supported algorithms are %s" % (",".join(_SUPPORTED_ALGORITHMS)))

                    raise AssertionError()

                log.info(f"scipy minimizer algorithm set to:{user_setup_dict['algorithm']}")
                
            # We can assume that the setup has been already checked against the setup_keys
            for key in user_setup_dict:

                
                self._setup_dict[key] = user_setup_dict[key]

    

    def set_algorithm(self, algorithm: str) -> None:
        """
        set the algorithm for the scipy minimizer.
        Valid entries are "L-BFGS-B", "TNC", "SLSQP"
        
        :param algorithm: 
        :type algorithm: str
        :returns: 

        """
        if algorithm not in _SUPPORTED_ALGORITHMS:

            log.error("Supported algorithms are %s" % (",".join(_SUPPORTED_ALGORITHMS)))

            raise AssertionError()
        

        self._setup_dict["algorithm"] = algorithm

        
        
    # This cannot be part of a class, unfortunately, because of how PyGMO serialize objects
    
    @staticmethod
    def _check_bounds(x, minima, maxima):

#        return _check_bounds(x, minima, maxima)

        for val, min_val, max_val in zip(x, minima, maxima):

            if min_val is not None:

                if val < min_val:

                    return False

            if max_val is not None:

                if val > max_val:

                    return False
            

        return True

    def _minimize(self):

        # Build initial point
        x0 = []
        bounds = []
        minima = []
        maxima = []

        for i, (par_name, (cur_value, cur_delta, cur_min, cur_max)) in enumerate(
            self._internal_parameters.items()
        ):

            x0.append(cur_value)

            # scipy's algorithms will always try to evaluate the function exactly at the boundaries, which will
            # fail because the Jacobian is not defined there... let's fix this by using a slightly larger or smaller
            # minimum and maximum within the scipy algorithm than the real boundaries (saved in minima and maxima)

            minima.append(cur_min)
            maxima.append(cur_max)

            if cur_min is not None:

                cur_min = cur_min + 0.00005 * abs(cur_min)

            if cur_max is not None:

                cur_max = cur_max - 0.00005 * abs(cur_max)

            bounds.append((cur_min, cur_max))

        def wrapper(x):

            if not self._check_bounds(x, minima, maxima):

                return np.inf

            return self.function(*x)

        def wrapper_2(*x):

            return wrapper(x)

        def jacobian(x):

            if not self._check_bounds(x, minima, maxima):

                return np.inf

            jacv = get_jacobian(wrapper_2, x, minima, maxima)

            return jacv

        res = scipy.optimize.minimize(
            wrapper,
            np.array(x0),
            method=self._setup_dict["algorithm"],
            bounds=bounds,
            jac=jacobian,
            tol=self._setup_dict["tol"],
        )

        # Make sure the optimization worked

        if not res.success:

            raise FitFailed(
                "Could not converge. Message from solver: %s (status: %i)"
                % (res.message, res.status)
            )

        # Transform the result to numpy.array

        best_fit_values = np.array(res.x)

        return best_fit_values, float(res.fun)

@nb.njit(fastmath=True)
def _check_bounds(x, minima, maxima):

    for val, min_val, max_val in zip(x, minima, maxima):

        if min_val is not None:

            if val < min_val:

                return False

        if max_val is not None:

            if val > max_val:

                return False


    return True

