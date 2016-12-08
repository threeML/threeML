__author__ = "djJfunk <J. Michael Burgess>"

from astromodels.functions.functions import DiracDelta, StepFunctionUpper
import numpy as np


def step_generator(intervals, parameter):
    """




    :param intervals: the 1- or 2-D intervals to be used
    :param parameter:
    """

    intervals = np.atleast_2d(intervals)

    # need to make sure the shape is right
    # assert self._intervals.shape


    if intervals.shape[0] > 1 and intervals.shape[1] == 2:

        n_intervals = intervals.shape[0]

        is_2d = True

    elif intervals.shape[0] == 1:

        n_intervals = intervals.shape[1]
        intervals = intervals[0]

        is_2d = False

    else:

        raise RuntimeError("These intervals are not yet supported")

    parameter_min = parameter.min_value
    parameter_max = parameter.max_value
    initial_value = parameter.value

    if is_2d:

        func = StepFunctionUpper()

        for i in range(n_intervals - 1):

            func += StepFunctionUpper()

        for i, interval in enumerate(intervals):

            func.free_parameters['value_%d' % i].value = initial_value
            func.free_parameters['value_%d' % i].min_value = parameter_min
            func.free_parameters['value_%d' % i].max_value = parameter_max

            func.parameters['upper_bound_%d' % i] = interval[1]

            func.parameters['lower_bound_%d' % i] = interval[0]


    else:

        func = DiracDelta()

        for i in range(self._n_intervals - 1):

            func += DiracDelta()

        for i, interval in enumerate(intervals):

            func.free_parameters['value_%d' % i].value = initial_value
            func.free_parameters['value_%d' % i].min_value = parameter_min
            func.free_parameters['value_%d' % i].max_value = parameter_max

            func.parameters['upper_bound_%d' % i] = interval[1]

            func.parameters['lower_bound_%d' % i] = interval[0]

    return func

    def get_interval_function(self):

    def get_fixed_point_funtion(self):
