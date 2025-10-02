import copy

import matplotlib.pyplot as plt
import numpy as np

from threeML.io.plotting.step_plot import step_plot
from threeML.utils.interval import Interval, IntervalSet
from threeML.utils.statistics.stats_tools import sqrt_sum_of_squares, sqrt_sum_of_squares_pairwise


class Histogram(IntervalSet):
    INTERVAL_TYPE = Interval

    def __init__(
        self,
        list_of_intervals,
        contents=None,
        stat_errors=None,
        sys_errors=None,
        is_poisson=False,
    ):
        if contents is None:
            self._contents = np.zeros(len(list_of_intervals))

        else:
            assert len(list_of_intervals) == len(
                contents
            ), "contents and intervals are not the same dimension "

            self._contents = np.array(contents)

        if stat_errors is not None:
            assert len(stat_errors) == len(
                contents
            ), "contents and stat errors are not the same dimension "

            assert is_poisson is False, "cannot have errors and is_poisson True"

            self._stat_errors = np.array(stat_errors)

        else:
            self._stat_errors = None

        if sys_errors is not None:
            assert len(sys_errors) == len(
                contents
            ), "contents and errors are not the same dimension "

            self._sys_errors = np.array(sys_errors)

        else:
            self._sys_errors = None

        self._is_poisson = is_poisson

        super(Histogram, self).__init__(list_of_intervals)

        # make some assertions so that we make sure the histogram makes sense

        assert self.is_contiguous(), "Histograms must have contiguous bins"

        assert self.is_sorted, "Histogram bins must be ordered"

    def bin_entries(self, entires):
        """Add the entries into the proper bin.

        :param entires: list of events
        :return:
        """

        which_bins = np.digitize(entires, self.edges) - 1

        for bin in which_bins:
            try:
                self._contents[bin] += 1

            except IndexError:
                # ignore if we are outside the bins
                pass

    def __add__(self, other):
        assert self == other, "The bins are not equal"

        new_contents = self.contents + other.contents

        if self._is_poisson:
            assert (
                other.is_poisson
            ), "Trying to add a Poisson and non-poisson histogram together"

            new_stat_errors = None

        else:
            assert (
                not other.is_poisson
            ), "Trying to add a Poisson and non-poisson histogram together"

<<<<<<< HEAD
            if self._stat_errors is not None:

=======
            if self._errors is not None:
>>>>>>> dev
                assert (
                    other._stat_errors is not None
                ), "This histogram has errors, but the other does not"

                new_stat_errors = np.array(
                    [
                        sqrt_sum_of_squares([e1, e2])
                        for e1, e2 in zip(self._stat_errors, other._stat_errors)
                    ]
                )

            else:
<<<<<<< HEAD

                new_stat_errors = None
        
        if self._sys_errors is not None and other._sys_errors is not None:

=======
                new_errors = None

        if self._sys_errors is not None and other.sys_errors is not None:
>>>>>>> dev
            new_sys_errors = np.array(
                [
                    sqrt_sum_of_squares([e1, e2])
                    for e1, e2 in zip(self._sys_errors * self._contents, other.sys_errors * other._contents)
                ]
            ) / new_contents

        elif self._sys_errors is not None:
<<<<<<< HEAD

            new_sys_errors = self._sys_errors * self._contents / new_contents

        elif other.sys_errors is not None:

            new_sys_errors = other.sys_errors * other._contents / new_contents
=======
            new_sys_errors = self._sys_errors

        elif other.sys_errors is not None:
            new_sys_errors = other.sys_errors
>>>>>>> dev

        else:
            new_sys_errors = None

        # because Hist gets inherited very deeply, when we add we will not know exactly
        # what all the additional class members will be, so we will make a copy of the
        # class This is not ideal and there is probably a better way to do this
        # TODO: better new hist constructor

        new_hist = copy.deepcopy(self)

        new_hist._contents = new_contents
        new_hist._stat_errors = new_stat_errors
        new_hist._sys_errors = new_sys_errors

        return new_hist

    def has_systematic_errors(self):
        """
        check if the histogram has systematic errors
        :return:
        """

        return self._sys_errors is not None
    
    def has_stat_errors(self):
        """
        check if the histogram has statistical errors
        :return:
        """

        return self._stat_errors is not None or self._is_poisson

    @property
    def errors(self):
<<<<<<< HEAD
        """ If the histogram has systematic errors, return the sum in 
        quadrature of the statistical and systematic error, otherwise return 
        the statistical error.
        """

        if self._sys_errors is None:
            
            return self._stat_errors
        
        return sqrt_sum_of_squares_pairwise(self._stat_errors, self._sys_errors * self._contents)

    @property
    def total_error(self):

        return sqrt_sum_of_squares(self.errors)
=======
        return self._errors

    @property
    def total_error(self):
        return sqrt_sum_of_squares(self._errors)
>>>>>>> dev

    @property
    def total_stat_error(self):

        return sqrt_sum_of_squares(self._stat_errors)

    @property
    def stat_errors(self):

        return self._stat_errors
    
    @property
    def sys_errors(self):
        return self._sys_errors

    @property
    def contents(self):
        return self._contents

    @property
    def total(self):
        return sum(self._contents)

    @property
    def is_poisson(self):
        return self._is_poisson

    @classmethod
    def from_numpy_histogram(
        cls, hist, errors=None, sys_errors=None, is_poisson=False, **kwargs
    ):
        """
        create a Histogram from a numpy histogram.
        Example:

            r = np.random.randn(1000)
            np_hist = np.histogram(r)
            hist = Histogram.from_numpy_histogram(np_hist)


        :param hist: a np.histogram tuple
        :param errors: list of errors for each bin in the numpy histogram
        :param sys_errors: list of systematic (relative) errors for each bin in the numpy histogram
        :param is_poisson: if the data is Poisson distributed or not
        :param kwargs: any kwargs to pass along
        :return: a Histogram object
        """

        # extract the contents and the bin edges

        contents = hist[0]  # type: np.ndarray
        edges = hist[1]  # type: np.ndarray

        # convert the edges to an interval set

        bounds = IntervalSet.from_list_of_edges(edges)  # type: IntervalSet

        # return a histogram

        return cls(
            list_of_intervals=bounds,
            contents=contents,
            errors=errors,
            sys_errors=sys_errors,
            is_poisson=is_poisson,
            **kwargs,
        )

    @classmethod
    def from_entries(cls, list_of_intervals, entries):
        """Create a histogram from a list of intervals and entries to bin.

        :param list_of_intervals:
        :param entries:
        :return:
        """

        new_hist = cls(list_of_intervals=list_of_intervals, is_poisson=True)

        new_hist.bin_entries(entires=entries)

        return new_hist

    def display(self, fill=False, fill_min=0.0, x_label="x", y_label="y", **kwargs):
        fig, ax = plt.subplots()

        step_plot(
            xbins=self.bin_stack,
            y=self._contents,
            ax=ax,
            fill=fill,
            fill_min=fill_min,
            **kwargs,
        )

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        return fig
