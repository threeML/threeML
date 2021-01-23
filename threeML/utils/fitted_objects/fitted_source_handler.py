from builtins import map, object, zip

__author__ = "grburgess"

import functools
import itertools

import numpy as np
from astromodels import use_astromodels_memoization

from threeML.utils.progress_bar import tqdm


class GenericFittedSourceHandler(object):
    def __init__(
        self,
        analysis_result,
        new_function,
        parameter_names,
        parameters,
        confidence_level,
        equal_tailed,
        *independent_variable_range
    ):
        """
        A generic 3ML fitted source  post-processor. This should be sub-classed in general

        :param analysis_result: a 3ML analysis result
        :param new_function: the function to use the fitted values to compute new values
        :param parameter_names: a list of parameter names
        :param parameters: astromodels parameter dictionary
        :param confidence_level: the confidence level to compute error
        :param independent_variable_range: the range(s) of independent values to compute the new function over
        """

        # bind the class properties

        self._analysis_results = analysis_result
        self._analysis = analysis_result
        self._independent_variable_range = independent_variable_range
        self._cl = confidence_level
        self._equal_tailed = equal_tailed
        self._function = new_function
        self._parameter_names = parameter_names
        self._parameters = parameters

        # if only 1-D then we must place into its own tuple to
        # keep from confusing itertools

        if len(self._independent_variable_range) == 1:
            self._independent_variable_range = (
                self._independent_variable_range[0],)

        # figure out the output shape of the best fit and errors

        self._out_shape = tuple(map(len, self._independent_variable_range))

        # construct the propagated function

        self._build_propagated_function()

        # fold the function through its independent values
        self._evaluate()

    def __add__(self, other):
        """
        The basics of adding are handled in the VariatesContainer
        :param other: another fitted source handler
        :return: a VariatesContainer with the summed values
        """

        # assure that the shapes will be the same
        assert (
            other._out_shape == self._out_shape
        ), "cannot sum together arrays with different shapes!"

        # this will get the value container for the other values

        return self.values + other.values

    def __radd__(self, other):

        if other == 0:

            return self

        else:

            return self.values + other.values

    def _transform(self, value):
        """
        dummy transform to be overridden in a subclass
        :param value:
        :return: transformed value
        """

        return value

    def update_tag(self, tag, value):

        pass

    def _build_propagated_function(self):
        """
        builds a propagated function using RandomVariates propagation

        :return:
        """

        arguments = {}

        # because we might be using composite functions,
        # we have to keep track of parameter names in a non-elegant way
        for par, name in zip(list(self._parameters.values()), self._parameter_names):

            if par.free:

                this_variate = self._analysis_results.get_variates(par.path)

                # Do not use more than 1000 values (would make computation too slow for nothing)

                if len(this_variate) > 1000:
                    this_variate = np.random.choice(this_variate, size=1000)

                arguments[name] = this_variate

            else:

                # use the fixed value rather than a variate

                arguments[name] = par.value

        # create the propagtor

        self._propagated_function = self._analysis_results.propagate(
            self._function, **arguments
        )

    def _evaluate(self):
        """

        calculate the best or mean fit of the new function or
        quantity

        :return:
        """
        # if there are independent variables
        if self._independent_variable_range:

            variates = []

            # scroll through the independent variables
            n_iterations = np.product(self._out_shape)

            with use_astromodels_memoization(False):

                for variables in tqdm(
                    itertools.product(*self._independent_variable_range),
                    desc="Propagating errors",
                ):
                    variates.append(self._propagated_function(*variables))
        # otherwise just evaluate
        else:

            variates = self._propagated_function()

        # create a variates container

        self._propagated_variates = VariatesContainer(
            variates, self._out_shape, self._cl, self._transform, self._equal_tailed
        )

    @property
    def values(self):
        """

        :return: The VariatesContainer
        """

        return self._propagated_variates

    @property
    def samples(self):
        """

        :return: the raw samples of the variates
        """

        return self._propagated_variates.samples

    @property
    def median(self):
        """

        :return: the median of the variates
        """

        return self._propagated_variates.median

    @property
    def average(self):
        """

        :return: the average of the variates
        """

        return self._propagated_variates.average

    @property
    def upper_error(self):
        """

        :return: the upper error of the variates
        """

        return self._propagated_variates.upper_error

    @property
    def lower_error(self):
        """

        :return: the lower error of the variates
        """

        return self._propagated_variates.lower_error


def transform(method):
    """
    A wrapper to call the _transform method for outputs of Variates container class
    :param method:
    :return:
    """

    @functools.wraps(method)
    def wrapped(instance, *args, **kwargs):
        return instance._transform(method(instance, *args, **kwargs))

    return wrapped


class VariatesContainer(object):
    def __init__(self, values, out_shape, cl, transform, equal_tailed=True):
        """
        A container to store an *List* of RandomVariates and transform their outputs
        to the appropriate shape. This cannot be done with normal numpy array operations
        because an array of RandomVariates becomes a normal ndarray. Therefore, we calculate
        the averages, errors, etc, and transform those.

        Additionally, any unit association must be done post calculation as well because the
        numpy array constructor sees a unit array as a regular array and again loses the RandomVariates
        properties. Therefore, the transform method is used which applies a function to the output properties,
        e.g., a unit association and or conversion.



        :param values: a flat List of RandomVariates
        :param out_shape: the array shape for the output variables
        :param cl: the confidence level to calculate error intervals on
        :param transform: a method to transform the outputs
        :param equal_tailed: whether to use equal-tailed error intervals or not
        """

        self._values = values  # type: list

        self._out_shape = out_shape  # type: tuple

        self._cl = cl  # type: float

        self._equal_tailed = equal_tailed  # type: bool

        self._transform = transform  # type: callable

        # calculate mean and median and transform them into the provided
        # output shape

        self._average = np.array([val.average for val in self._values])

        self._average = self._average.reshape(self._out_shape)

        self._median = np.array([val.median for val in self._values])

        self._median = self._median.reshape(self._out_shape)

        # construct the error intervals

        upper_error = []
        lower_error = []

        # if equal tailed errors requested
        if equal_tailed:

            for val in self._values:

                error = val.equal_tail_interval(self._cl)
                upper_error.append(error[1])
                lower_error.append(error[0])

        else:

            # else use the hdp

            for val in self._values:

                error = val.highest_posterior_density_interval(self._cl)
                upper_error.append(error[1])
                lower_error.append(error[0])

        # reshape the errors into the output shape

        self._upper_error = np.array(upper_error).reshape(self._out_shape)
        self._lower_error = np.array(lower_error).reshape(self._out_shape)

        samples = []

        for val in self._values:

            samples.append(val.samples)

        n_samples = len(samples[0])

        samples_shape = list(self._out_shape) + [n_samples]

        self._samples_shape = tuple(samples_shape)

        self._samples = np.array(samples).reshape(samples_shape)

    @property
    def values(self):
        """
        :return: the list of of RandomVariates
        """

        return self._values

    @property
    @transform
    def samples(self):
        """

        :return: the transformed raw samples
        """
        return self._samples

    @property
    @transform
    def average(self):
        """

        :return: the transformed average
        """

        return self._average

    @property
    @transform
    def median(self):
        """

        :return: the transformed median
        """

        return self._median

    @property
    @transform
    def upper_error(self):
        """

        :return: the transformed upper error
        """

        return self._upper_error

    @property
    @transform
    def lower_error(self):
        """

        :return: the transformed lower error
        """

        return self._lower_error

    def __add__(self, other):
        """


        :param other:
        :return:
        """

        assert (
            other._out_shape == self._out_shape
        ), "cannot sum together arrays with different shapes!"

        # this will get the value container for the other values

        other_values = other.values

        summed_values = [v + vo for v, vo in zip(self._values, other_values)]

        return VariatesContainer(
            summed_values,
            self._out_shape,
            self._cl,
            self._transform,
            self._equal_tailed,
        )

    def __radd__(self, other):

        if other == 0:

            return self

        else:

            other_values = other.values

            summed_values = [v + vo for v,
                             vo in zip(self._values, other_values)]

            return VariatesContainer(
                summed_values,
                self._out_shape,
                self._cl,
                self._transform,
                self._equal_tailed,
            )
