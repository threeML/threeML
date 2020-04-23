__author__ = "grburgess <J. Michael Burgess>"

from astromodels.functions.functions import DiracDelta, StepFunctionUpper
import numpy as np


def step_generator(intervals, parameter):
    """

    Generates sum of step or dirac delta functions for the given intervals
    and parameter. This can be used to link time-independent parameters
    of a model to time.

    If the intervals provided are 1-D, i.e, they are the means of time bins or
    the TOA of photons, then a sum of dirac deltas is returned with their centers
    at the times provided

    If the intervals are 2-D (start, stop), sum of step functions is created with
    the bounds at the start and stop times of the interval.

    The parameter is used to set the bounds and initial value, min, max of the
    non-zero points of the functions

    :param intervals: an array of the 1- or 2-D intervals to be used
    :param parameter: astromodels parameter
    """

    intervals = np.atleast_2d(intervals)

    # need to make sure the shape is right
    # assert self._intervals.shape

    # Check if the interval is 2D or 1D
    if intervals.shape[0] > 1 and intervals.shape[1] == 2:

        n_intervals = intervals.shape[0]

        is_2d = True

    elif intervals.shape[0] == 1:

        n_intervals = intervals.shape[1]
        intervals = intervals[0]

        is_2d = False

    else:

        raise RuntimeError("These intervals are not yet supported")

    # Copy the parameter values
    parameter_min = parameter.min_value
    parameter_max = parameter.max_value
    initial_value = parameter.value

    if is_2d:

        # For 2D intervals, we grab a step function

        func = StepFunctionUpper()

        # Sum up the functions

        for i in range(n_intervals - 1):

            func += StepFunctionUpper()

        # Go through and iterate over intervals to set the parameter values

        for i, interval in enumerate(intervals):

            i = i + 1

            func.free_parameters["value_%d" % i].value = initial_value
            func.free_parameters["value_%d" % i].min_value = parameter_min
            func.free_parameters["value_%d" % i].max_value = parameter_max

            func.parameters["upper_bound_%d" % i].value = interval[1]

            func.parameters["lower_bound_%d" % i].value = interval[0]

    else:

        # For 1-D intervals, just create a sum of delta functions

        func = DiracDelta()

        for i in range(n_intervals - 1):

            func += DiracDelta()

        # Set up the values

        for i, interval in enumerate(intervals):

            i = i + 1

            func.free_parameters["value_%d" % i].value = initial_value
            func.free_parameters["value_%d" % i].min_value = parameter_min
            func.free_parameters["value_%d" % i].max_value = parameter_max

            func.parameters["zero_point_%d" % i].value = interval

    return func
