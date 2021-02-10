from typing import Any, Dict, List, Optional

import numpy as np
from pyswarms.base import SwarmOptimizer
from pyswarms.single.global_best import GlobalBestPSO
from pyswarms.single.local_best import LocalBestPSO

from threeML.config import threeML_config
from threeML.io.logging import setup_logger
from threeML.minimizer.minimization import GlobalMinimizer, LocalMinimizer

log = setup_logger(__name__)


class PySwarmsWrapper(object):

    def __init__(self, function, parameters, dim):
        """
        A wrapper to prepare out likelihood profiles for
        pyswarms
        """
        self._dim = dim
        self._objective_function = function

        minima = []
        maxima = []

        for param, (cur_value, cur_delta, cur_min, cur_max) in parameters.items():

            if cur_min is None or cur_max is None:

                log.error(f"{param}"
                          "In order to use the PySwarms minimizer, you have to provide a minimum and a "
                          "maximum for all parameters in the model.")

                raise RuntimeError(
                )

            minima.append(cur_min)
            maxima.append(cur_max)

        self._bounds = (np.array(minima), np.array(maxima))

        self._parameters = parameters

    def __call__(self, x):

        # break up for number of particles
        # and this optimizes so we need the minus

        # the objective function here is minus log like
        # so we need to return - 2 * log like

        val =  2 * np.array([self._objective_function(*y) for y in x])

#        print(x)
        
        return val

    @property
    def bounds(self):
        return self._bounds

    @property
    def dim(self):
        return self._dim


class PySwarmsMinimizer(GlobalMinimizer):

    valid_setup_keys = ("algorithm", "options", "n_particles",
                        "n_iterations", "kwargs", "second_minimization")

    def __init__(self, function, parameters, verbosity=1, setup_dict=None):

        super(PySwarmsMinimizer, self).__init__(
            function, parameters, verbosity, setup_dict
        )

    def _setup(self, user_setup_dict):

        default_setup = {

            "algorithm": GlobalBestPSO,
            "options": {'c1': 0.5, 'c2': 0.3, 'w': 0.9},
            "n_particles": 10,
            "n_iterations": 1000,
            "kwargs": {}
        }

        if user_setup_dict is None:

            user_setup_dict = default_setup

        for k, v in user_setup_dict.items():

            default_setup[k] = v

        log.debug(default_setup)

        if not issubclass(default_setup["algorithm"], SwarmOptimizer):

            log.error("You must set a PySwarms algorithm")
            raise RuntimeError()

        # for k, v in default_setup.items():

        #     self._setup_dict[k] = v

        self._setup_dict = default_setup

    def _minimize(self):

        n_par = len(self._internal_parameters)

        wrapper = PySwarmsWrapper(
            function=self.function, parameters=self._internal_parameters, dim=n_par)

        init_pos = []

        for k, (val, _, _, _) in self._internal_parameters.items():

            init_pos.append(val)

        optimizer: SwarmOptimizer = self._setup_dict["algorithm"](
            n_particles=self._setup_dict["n_particles"],
            dimensions=n_par,
            options=self._setup_dict["options"],
            bounds=wrapper.bounds,
            #init_pos=np.array(init_pos)
        )

        cost, best_fit_values = optimizer.optimize(
            wrapper, iters=self._setup_dict["n_iterations"])

        return best_fit_values, -0.5 * cost
