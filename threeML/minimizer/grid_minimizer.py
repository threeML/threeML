import collections
import numpy as np
import itertools

from threeML.minimizer.minimization import Minimizer, get_minimizer
from astromodels import Parameter


class AllFitFailed(RuntimeError):
    pass


class GridMinimizer(Minimizer):

    def __init__(self, function, parameters, ftol=1e3, verbosity=1):

        self._miminizer = None
        self._grid = collections.OrderedDict()

        # Keep a copy of the original values for the parameters

        self._original_values = collections.OrderedDict()

        for par_name, par in parameters.items():

            self._original_values[par_name] = par.value

        super(GridMinimizer, self).__init__(function, parameters, ftol, verbosity)

    def _setup(self):

        # Nothing special to do

        pass

    def set_algorithm(self, algorithm):

        pass

    def set_minimizer(self, minimizer_type):
        """
        Sets the minimizer to use for each point in the grid

        :param minimizer_type: one of the accepted minimizers
        :return: None
        """

        self._minimizer = get_minimizer(minimizer_type)(self.function, self.parameters, self.ftol)

    def add_parameter_to_grid(self, parameter, grid):
        """
        Add a parameter to the grid

        :param parameter: an instance of a parameter
        :param grid: a list (or a numpy.array) with the values the parameter is supposed to take during the grid search
        :return: None
        """

        assert isinstance(parameter, Parameter)

        assert parameter in self.parameters.values(), "Parameter %s is not part of the current model" % parameter.name

        grid = np.array(grid)

        assert grid.ndim == 1, "The grid for parameter %s must be 1-dimensional" % parameter.name

        # Check that the grid is legal
        assert grid.max() <= parameter.max_value, "The maximum value in the grid (%s) is above the maximum legal value " \
                                                  "(%s) for parameter %s" %(grid.max(), parameter.max_value,
                                                                            parameter.name)

        assert grid.min() >= parameter.min_value, "The minimum value in the grid (%s) is above the minimum legal " \
                                                  "value (%s) for parameter %s" % (grid.min(), parameter.min_value,
                                                                                   parameter.name)

        self._grid[parameter.path] = grid

    def minimize(self, compute_covar=True):

        assert len(self._grid) > 0, "You need to set up a grid using add_parameter_to_grid"

        assert self._minimizer is not None, "You need to chose a minimizer using the set_minimizer method"

        # For each point in the grid, perform a fit

        parameters = self._grid.keys()

        overall_minimum = 1e20
        best_fit_values = None
        covariance_matrix = None

        for values_tuple in itertools.product(*self._grid.values()):

            # Reset everything to the original values, so that the fit will always start
            # from there, instead that from the values obtained in the last iterations, which
            # might have gone completely awry

            for par_name, par_value in self._original_values.items():

                self.parameters[par_name].value = par_value

            # Now set the parameters in the grid to their starting values

            for i, this_value in enumerate(values_tuple):

                self.parameters[parameters[i]].value = this_value

            # Perform fit

            try:

                this_best_fit_values, this_minimum = self._minimizer.minimize()

            except:

                continue

            # If this minimum is the overall minimum, save the result

            if this_minimum < overall_minimum:

                overall_minimum = this_minimum
                best_fit_values = this_best_fit_values
                covariance_matrix = self._minimizer.covariance_matrix

        if best_fit_values is None:

            raise AllFitFailed("All fit starting from values in the grid have failed!")

        self._store_fit_results(best_fit_values, overall_minimum, covariance_matrix)

        return best_fit_values, overall_minimum