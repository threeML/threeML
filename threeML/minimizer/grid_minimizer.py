import collections
import itertools
from builtins import str

import numpy as np
from astromodels import Parameter

from threeML.config.config import threeML_config
from threeML.io.logging import setup_logger
from threeML.minimizer.minimization import GlobalMinimizer
from threeML.utils.progress_bar import tqdm

log = setup_logger(__name__)


class AllFitFailed(RuntimeError):
    pass


class GridMinimizer(GlobalMinimizer):

    valid_setup_keys = ("grid", "second_minimization", "callbacks")

    def __init__(self, function, parameters, verbosity=1):

        self._grid = collections.OrderedDict()

        # Keep a copy of the original values for the parameters

        self._original_values = collections.OrderedDict()

        for par_name, par in parameters.items():

            self._original_values[par_name] = par.value

        super(GridMinimizer, self).__init__(function, parameters, verbosity)

        # This list will contain callbacks, if any
        self._callbacks = []

    def _setup(self, user_setup_dict):

        if user_setup_dict is None:

            return

        # This minimizer MUST be set up with a grid, so we enforce that user_setup_dict is not None
        assert (
            user_setup_dict is not None
        ), "You have to setup a grid for this minimizer"

        assert "grid" in user_setup_dict, "You have to setup a grid for this minimizer"

        assert (
            "second_minimization" in user_setup_dict
        ), "You have to set up a second minimizer"

        # Setup grid

        for parameter, grid in user_setup_dict["grid"].items():

            log.debug(f"added {parameter} to the grid")

            self.add_parameter_to_grid(parameter, grid)

        # Setup inner minimization
        self._2nd_minimization = user_setup_dict["second_minimization"]

        # If there are callbacks, set them up
        if "callbacks" in user_setup_dict:

            for callback in user_setup_dict["callbacks"]:

                self.add_callback(callback)

    def add_callback(self, function):
        """
        This adds a callback function which is called after each point in the grid has been used.

        :param function: a function receiving in input a tuple containing the point in the grid and the minimum of the
        function reached starting from that point. The function should return nothing
        :return: none
        """

        self._callbacks.append(function)

    def remove_callbacks(self):
        """
        Remove all callbacks added with add_callback

        :return: none
        """
        self._callbacks = []

    def add_parameter_to_grid(self, parameter, grid):
        """
        Add a parameter to the grid

        :param parameter: an instance of a parameter or a parameter path
        :param grid: a list (or a numpy.array) with the values the parameter is supposed to take during the grid search
        :return: None
        """

        if isinstance(parameter, Parameter):

            assert parameter in list(self.parameters.values()), (
                "Parameter %s is not part of the " "current model" % parameter.name
            )

        else:

            # Assume parameter is a path
            parameter_path = str(parameter)

            # Make a list of paths for the parameters
            v = list(self.parameters.values())
            parameters_paths = [x.path for x in v]

            try:

                idx = parameters_paths.index(parameter_path)

            except ValueError:

                log.error("Could not find parameter %s in current model" %
                          parameter_path)

                raise ValueError()

            parameter = v[idx]

        grid = np.array(grid)

        assert grid.ndim == 1, (
            "The grid for parameter %s must be 1-dimensional" % parameter.name
        )

        # Check that the grid is legal
        if parameter.max_value is not None:

            assert grid.max() < parameter.max_value, (
                "The maximum value in the grid (%s) is above the maximum "
                "legal value (%s) for parameter %s"
                % (grid.max(), parameter.max_value, parameter.name)
            )

        if parameter.min_value is not None:

            assert grid.min() > parameter.min_value, (
                "The minimum value in the grid (%s) is above the minimum legal "
                "value (%s) for parameter %s"
                % (grid.min(), parameter.min_value, parameter.name)
            )

            log.debug(f"grid successfully added: {grid}")

        self._grid[parameter.path] = grid

    def _minimize(self):

        assert (
            len(self._grid) > 0
        ), "You need to set up a grid using add_parameter_to_grid"

        if self._2nd_minimization is None:

            raise RuntimeError(
                "You did not setup this global minimizer (GRID). You need to use the .setup() method"
            )

        # For each point in the grid, perform a fit

        parameters = list(self._grid.keys())

        overall_minimum = 1e20
        internal_best_fit_values = None

        n_iterations = np.prod([x.shape for x in list(self._grid.values())])

        if threeML_config["interface"]["show_progress_bars"]:
            p = tqdm(total=n_iterations, desc="Grid Minimization")

        for values_tuple in itertools.product(*list(self._grid.values())):

            # Reset everything to the original values, so that the fit will always start
            # from there, instead that from the values obtained in the last iterations, which
            # might have gone completely awry

            for par_name, par_value in self._original_values.items():

                self.parameters[par_name].value = par_value

            # Now set the parameters in the grid to their starting values

            for i, this_value in enumerate(values_tuple):

                self.parameters[parameters[i]].value = this_value

            # Get a new instance of the minimizer. We need to do this instead of reusing an existing instance
            # because some minimizers (like iminuit) keep internal track of their status, so that reusing
            # a minimizer will create correlation between the different points
            # NOTE: this line necessarily needs to be after the values of the parameters has been set to the
            # point, because the init method of the minimizer instance will use those values to set the starting
            # point for the fit

            _minimizer = self._2nd_minimization.get_instance(
                self.function, self.parameters, verbosity=0
            )

            # Perform fit

            try:

                # We call _minimize() and not minimize() so that the best fit values are
                # in the internal system.

                this_best_fit_values_internal, this_minimum = _minimizer._minimize()

            except:

                # A failure is not a problem here, only if all of the fit fail then we have a problem
                # but this case is handled later

                continue

            # If this minimum is the overall minimum, save the result

            if this_minimum < overall_minimum:

                overall_minimum = this_minimum
                internal_best_fit_values = this_best_fit_values_internal

            # Use callbacks (if any)
            for callback in self._callbacks:

                callback(values_tuple, this_minimum)

            if threeML_config["interface"]["show_progress_bars"]:
                p.update(1)

        if internal_best_fit_values is None:

            raise AllFitFailed(
                "All fit starting from values in the grid have failed!")

        return internal_best_fit_values, overall_minimum
