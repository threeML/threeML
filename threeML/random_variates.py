import numpy as np

from threeML.io.uncertainty_formatter import uncertainty_formatter


class RandomVariates(np.ndarray):
    """
    A subclass of np.array which is meant to contain samples for one parameter. This class contains methods to easily
    compute properties for the parameter (errors and so on)
    """

    def __new__(cls, input_array, value=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type

        obj = np.asarray(input_array).view(cls)

        # Add the value
        obj._orig_value = value

        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):

        # see InfoArray.__array_finalize__ for comments
        if obj is None:
            return

        # Add the value
        self._orig_value = getattr(obj, "_orig_value", None)

    def __array_wrap__(self, out_arr, context=None):

        # This gets called at the end of any operation, where out_arr is the result of the operation
        # We need to update _orig_value so that the final results will have it

        out_arr._orig_value = out_arr.median

        # then just call the parent
        return super(RandomVariates, self).__array_wrap__(out_arr, context)

    # def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):

    #     # TODO: must make this return single numbers is needed

    #     args = []
    #     in_no = []
    #     for i, input_ in enumerate(inputs):
    #         if isinstance(input_, RandomVariates):
    #             in_no.append(i)
    #             args.append(input_.view(np.ndarray))
    #         else:
    #             args.append(input_)

    #     outputs = kwargs.pop('out', None)
    #     out_no = []

    #     if outputs:
    #         out_args = []
    #         for j, output in enumerate(outputs):
    #             if isinstance(output, RandomVariates):
    #                 out_no.append(j)
    #                 out_args.append(output.view(np.ndarray))
    #             else:
    #                 out_args.append(output)
    #         kwargs['out'] = tuple(out_args)
    #     else:
    #         outputs = (None,) * ufunc.nout

    #     results = super(RandomVariates, self).__array_ufunc__(ufunc, method,
    #                                              *args, **kwargs)
    #     if results is NotImplemented:
    #         return NotImplemented

    #     if method == 'at':
    #         return

    #     if ufunc.nout == 1:
    #         results = (results,)

    #     results = tuple((np.asarray(result).view(RandomVariates)
    #                      if output is None else output)
    #                     for result, output in zip(results, outputs))

    #     return results[0] if len(results) == 1 else results

    @property
    def median(self):
        """Returns median value"""

        # the np.asarray casting avoids the calls to __new__ and __array_finalize_ of this class

        return float(np.median(np.asarray(self)))

    # @property
    # def mean(self):
    #     """Returns average value"""

    #     return float(np.asarray(self).mean())

    @property
    def std(self):
        """Returns sample std value"""

        return float(np.asarray(self).std())

    @property
    def var(self):
        """Returns sample variance value"""

        return float(np.asarray(self).var())

    @property
    def average(self):
        """Returns average value"""

        return float(np.asarray(self).mean())

    @property
    def value(self):

        return float(self._orig_value)

    @property
    def samples(self):

        return np.asarray(self)

    def highest_posterior_density_interval(self, cl=0.68):
        """
        Returns the Highest Posterior Density interval (HPD) for the parameter, for the given credibility level.

        NOTE: the returned interval is the HPD only if the posterior is not multimodal. If it is multimodal, you should
        probably report the full posterior, not only an interval.

        :param cl: credibility level (0 < cl < 1)
        :return: (low_bound, hi_bound)
        """

        assert 0 < cl < 1, "The credibility level should be 0 < cl < 1"

        # NOTE: we cannot sort the array, because we would destroy the covariance with other physical quantities,
        # so we get a copy instead. This copy will live only for the duration of this method (but of course will be
        # collected only whenevery the garbage collector decides to).

        ordered = np.sort(np.array(self))

        n = ordered.size

        # This is the probability that the interval should span
        interval_integral = cl

        # If all values have the same probability, then the hpd is degenerate, but its length is from 0 to
        # the value corresponding to the (interval_integral * n)-th sample.
        # This is the index of the rightermost element which can be part of the interval

        index_of_rightmost_possibility = int(np.floor(interval_integral * n))

        # Compute the index of the last element that is eligible to be the left bound of the interval

        index_of_leftmost_possibility = n - index_of_rightmost_possibility

        # Now compute the width of all intervals that might be the one we are looking for

        interval_width = (
            ordered[index_of_rightmost_possibility:]
            - ordered[:index_of_leftmost_possibility]
        )

        # This might happen if there are too few values
        if len(interval_width) == 0:
            raise RuntimeError("Too few elements for interval calculation")

        # Find the index of the shortest interval

        idx_of_minimum = np.argmin(interval_width)

        # Find the extremes of the shortest interval

        hpd_left_bound = ordered[idx_of_minimum]
        hpd_right_bound = ordered[idx_of_minimum + index_of_rightmost_possibility]

        return hpd_left_bound, hpd_right_bound

    def equal_tail_interval(self, cl=0.68):
        """
        Returns the equal tail interval, i.e., an interval centered on the median of the distribution with
        the same probability on the right and on the left of the mean.

        If the distribution of the parameter is Gaussian and cl=0.68, this is equivalent to the 1 sigma confidence
        interval.

        :param cl: confidence level (0 < cl < 1)
        :return: (low_bound, hi_bound)
        """

        assert 0 < cl < 1, "Confidence level must be 0 < cl < 1"

        half_cl = cl / 2.0 * 100.0

        low_bound, hi_bound = np.percentile(
            np.asarray(self), [50.0 - half_cl, 50.0 + half_cl]
        )

        return float(low_bound), float(hi_bound)

    # np.ndarray already has a mean() and a std() methods

    def __repr__(self):

        # Get representation for the HPD

        min_bound, max_bound = self.highest_posterior_density_interval(0.68)

        hpd_string = uncertainty_formatter(self.median, min_bound, max_bound)

        # Get representation for the equal-tail interval

        min_bound, max_bound = self.equal_tail_interval(0.68)

        eqt_string = uncertainty_formatter(self.median, min_bound, max_bound)

        # Put them together

        representation = "equal-tail: %s, hpd: %s" % (eqt_string, hpd_string)

        return representation

    def __str__(self):

        return self.__repr__()
