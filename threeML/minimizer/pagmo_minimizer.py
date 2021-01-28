from __future__ import print_function
from builtins import range
from builtins import object
import numpy as np
import os
from threeML.utils.progress_bar import tqdm, trange

from threeML.minimizer.minimization import GlobalMinimizer
from threeML.parallel.parallel_client import is_parallel_computation_active

import pygmo as pg


class PAGMOWrapper(object):
    def __init__(self, function, parameters, dim):

        self._dim_ = dim

        self._objective_function = function

        minima = []
        maxima = []

        for param, (cur_value, cur_delta, cur_min, cur_max) in parameters.items():

            if cur_min is None or cur_max is None:

                raise RuntimeError(
                    "In order to use the PAGMO minimizer, you have to provide a minimum and a "
                    "maximum for all parameters in the model."
                )

            minima.append(cur_min)
            maxima.append(cur_max)

        self._minima = minima
        self._maxima = maxima
        self._parameters = parameters

    def fitness(self, x):

        val = self._objective_function(*x)

        # Note that we return a tuple with one element only. In PyGMO the objective functions
        # return tuples so that multi-objective optimization is also possible.
        return (val,)

    def get_bounds(self):

        return (self._minima, self._maxima)

    def get_name(self):

        return "JointLikelihood"


class PAGMOMinimizer(GlobalMinimizer):

    valid_setup_keys = (
        "islands",
        "population_size",
        "evolution_cycles",
        "second_minimization",
        "algorithm",
    )

    def __init__(self, function, parameters, verbosity=10, setup_dict=None):

        super(PAGMOMinimizer, self).__init__(
            function, parameters, verbosity, setup_dict
        )

    def _setup(self, user_setup_dict):

        if user_setup_dict is None:

            default_setup = {
                "islands": 8,
                "population_size": self._Npar * 20,
                "evolution_cycles": 1,
            }

            self._setup_dict = default_setup

        else:

            assert "algorithm" in user_setup_dict, (
                "You have to provide a pygmo.algorithm instance using "
                "the algorithm keyword"
            )

            algorithm_instance = user_setup_dict["algorithm"]

            assert isinstance(
                algorithm_instance, pg.algorithm
            ), "The algorithm must be an instance of a PyGMO algorithm"

            # We can assume that the setup has been already checked against the setup_keys
            for key in user_setup_dict:

                self._setup_dict[key] = user_setup_dict[key]

    # This cannot be part of a class, unfortunately, because of how PyGMO serialize objects

    def _minimize(self):

        # Gather the setup
        islands = self._setup_dict["islands"]
        pop_size = self._setup_dict["population_size"]
        evolution_cycles = self._setup_dict["evolution_cycles"]

        # Print some info
        print("\nPAGMO setup:")
        print("------------")
        print("- Number of islands:            %i" % islands)
        print("- Population size per island:   %i" % pop_size)
        print("- Evolutions cycles per island: %i\n" % evolution_cycles)

        Npar = len(self._internal_parameters)

        if is_parallel_computation_active():

            wrapper = PAGMOWrapper(
                function=self.function, parameters=self._internal_parameters, dim=Npar
            )

            # use the archipelago, which uses the ipyparallel computation

            archi = pg.archipelago(
                udi=pg.ipyparallel_island(),
                n=islands,
                algo=self._setup_dict["algorithm"],
                prob=wrapper,
                pop_size=pop_size,
            )
            archi.wait()

            # Display some info
            print("\nSetup before parallel execution:")
            print("--------------------------------\n")
            print(archi)

            # Evolve populations on islands
            print("Evolving... (progress not available for parallel execution)")

            # For some weird reason, ipyparallel looks for _winreg on Linux (where does
            # not exist, being a Windows module). Let's mock it with an empty module'
            mocked = False
            if os.path.exists("_winreg.py") is False:

                with open("_winreg.py", "w+") as f:

                    f.write("pass")

                mocked = True

            archi.evolve()

            # Wait for completion (evolve() is async)

            archi.wait_check()

            # Now remove _winreg.py if needed
            if mocked:

                os.remove("_winreg.py")

            # Find best and worst islands

            fOpts = np.array([x[0] for x in archi.get_champions_f()])
            xOpts = archi.get_champions_x()

        else:

            # do not use ipyparallel. Evolve populations on islands serially

            wrapper = PAGMOWrapper(
                function=self.function, parameters=self._internal_parameters, dim=Npar
            )

            xOpts = []
            fOpts = np.zeros(islands)

            for island_id in trange(islands, desc="pygmo minimization"):

                pop = pg.population(prob=wrapper, size=pop_size)

                for i in range(evolution_cycles):

                    pop = self._setup_dict["algorithm"].evolve(pop)

                # Gather results

                xOpts.append(pop.champion_x)
                fOpts[island_id] = pop.champion_f[0]



        # Find best and worst islands

        min_idx = fOpts.argmin()
        max_idx = fOpts.argmax()

        fOpt = fOpts[min_idx]
        fWorse = fOpts[max_idx]
        xOpt = np.array(xOpts)[min_idx]

        # Some information
        print("\nSummary of evolution:")
        print("---------------------")
        print("Best population has minimum %.3f" % (fOpt))
        print("Worst population has minimum %.3f" % (fWorse))
        print("")

        # Transform to numpy.array

        best_fit_values = np.array(xOpt)

        return best_fit_values, fOpt
