import numba as nb
import numpy as np

from threeML.config.config import threeML_config
from threeML.exceptions.custom_exceptions import custom_warnings
from threeML.io.logging import setup_logger
from threeML.utils.bayesian_blocks import (bayesian_blocks,
                                           bayesian_blocks_not_unique)
from threeML.utils.numba_utils import VectorFloat64, VectorInt64
from threeML.utils.progress_bar import tqdm
from threeML.utils.statistics.stats_tools import Significance
from threeML.utils.time_interval import TimeIntervalSet

log = setup_logger(__name__)


class NotEnoughData(RuntimeError):
    pass


class Rebinner(object):
    """
    A class to rebin vectors keeping a minimum value per bin. It supports array with a mask, so that elements excluded
    through the mask will not be considered for the rebinning

    """

    def __init__(self, vector_to_rebin_on, min_value_per_bin, mask=None):

        # Basic check that it is possible to do what we have been requested to do

        total = np.sum(vector_to_rebin_on)

        if total < min_value_per_bin:

            log.error("Vector total is %s, cannot rebin at %s per bin"
                      % (total, min_value_per_bin))

            raise NotEnoughData()

        # Check if we have a mask, if not prepare a empty one
        if mask is not None:

            mask = np.array(mask, bool)

            assert mask.shape[0] == len(vector_to_rebin_on), (
                "The provided mask must have the same number of "
                "elements as the vector to rebin on"
            )

        else:

            mask = np.ones_like(vector_to_rebin_on, dtype=bool)

        self._mask = mask

        # Rebin taking the mask into account

        self._starts = []
        self._stops = []
        self._grouping = np.zeros_like(vector_to_rebin_on)

        n = 0
        bin_open = False

        n_grouped_bins = 0

        for index, b in enumerate(vector_to_rebin_on):

            if not mask[index]:

                # This element is excluded by the mask

                if not bin_open:

                    # Do nothing
                    continue

                else:

                    # We need to close the bin here
                    self._stops.append(index)
                    n = 0
                    bin_open = False

                    # If we have grouped more than one bin

                    if n_grouped_bins > 1:

                        # group all these bins
                        self._grouping[index - n_grouped_bins + 1: index] = -1
                        self._grouping[index] = 1

                    # reset the number of bins in this group

                    n_grouped_bins = 0

            else:

                # This element is included by the mask

                if not bin_open:
                    # Open a new bin
                    bin_open = True

                    self._starts.append(index)
                    n = 0

                # Add the current value to the open bin

                n += b

                n_grouped_bins += 1

                # If we are beyond the requested value, close the bin

                if n >= min_value_per_bin:
                    self._stops.append(index + 1)

                    n = 0

                    bin_open = False

                    # If we have grouped more than one bin

                    if n_grouped_bins > 1:

                        # group all these bins
                        self._grouping[index - n_grouped_bins + 1: index] = -1
                        self._grouping[index] = 1

                    # reset the number of bins in this group

                    n_grouped_bins = 0

        # At the end of the loop, see if we left a bin open, if we did, close it

        if bin_open:
            self._stops.append(len(vector_to_rebin_on))

        assert len(self._starts) == len(self._stops), (
            "This is a bug: the starts and stops of the bins are not in " "equal number"
        )

        self._min_value_per_bin = min_value_per_bin

        self._n_bins = len(self._starts)
        self._starts = np.array(self._starts)
        self._stops = np.array(self._stops)

        log.debug(
            f"Vector was rebinned from {len(vector_to_rebin_on)} to {self._n_bins}")

    @property
    def n_bins(self):
        """
        Returns the number of bins defined.

        :return:
        """

        return self._n_bins

    @property
    def grouping(self):

        return self._grouping

    def rebin(self, *vectors):

        rebinned_vectors = []

        for vector in vectors:

            assert len(vector) == len(self._mask), (
                "The vector to rebin must have the same number of elements of the"
                "original (not-rebinned) vector"
            )

            # Transform in array because we need to use the mask

            if vector.dtype == np.int64:

                rebinned_vectors.append(_rebin_vector_int(
                    vector, self._starts, self._stops, self._mask, self._n_bins))

            else:

                rebinned_vectors.append(_rebin_vector_float(
                    vector, self._starts, self._stops, self._mask, self._n_bins))

        return rebinned_vectors

    def rebin_errors(self, *vectors):
        """
        Rebin errors by summing the squares

        Args:
            *vectors:

        Returns:
            array of rebinned errors

        """

        rebinned_vectors = []

        for vector in vectors:  # type: np.ndarray[np.ndarray]

            assert len(vector) == len(self._mask), (
                "The vector to rebin must have the same number of elements of the"
                "original (not-rebinned) vector"
            )

            rebinned_vector = []

            for low_bound, hi_bound in zip(self._starts, self._stops):

                rebinned_vector.append(
                    np.sqrt(np.sum(vector[low_bound:hi_bound] ** 2)))

            rebinned_vectors.append(np.array(rebinned_vector))

        return rebinned_vectors

    def get_new_start_and_stop(self, old_start, old_stop):

        assert len(old_start) == len(self._mask) and len(
            old_stop) == len(self._mask)

        new_start = np.zeros(len(self._starts))
        new_stop = np.zeros(len(self._starts))

        for i, (low_bound, hi_bound) in enumerate(zip(self._starts, self._stops)):
            new_start[i] = old_start[low_bound]
            new_stop[i] = old_stop[hi_bound - 1]

        return new_start, new_stop


class TemporalBinner(TimeIntervalSet):
    """
    An extension of the TimeInterval set that includes binning capabilities

    """

    @classmethod
    def bin_by_significance(
        cls,
        arrival_times,
        background_getter,
        background_error_getter=None,
        sigma_level=10,
        min_counts=1,
        tstart=None,
        tstop=None,
    ):
        """

        Bin the data to a given significance level for a given background method and sigma
        method. If a background error function is given then it is assumed that the error distribution
        is gaussian. Otherwise, the error distribution is assumed to be Poisson.

        :param background_getter: function of a start and stop time that returns background counts
        :param background_error_getter: function of a start and stop time that returns background count errors
        :param sigma_level: the sigma level of the intervals
        :param min_counts: the minimum counts per bin

        :return:
        """

        if tstart is None:

            tstart = arrival_times.min()

        else:

            tstart = float(tstart)

        if tstop is None:

            tstop = arrival_times.max()

        else:

            tstop = float(tstop)

        starts = []

        stops = []

        # Switching to a fast search
        # Idea inspired by Damien Begue

        # these factors change the time steps
        # in the fast search. should experiment
        if sigma_level > 25:

            increase_factor = 0.5
            decrease_factor = 0.5

        else:

            increase_factor = 0.25
            decrease_factor = 0.25

        current_start = arrival_times[0]

        # first we need to see if the interval provided has enough counts

        _, counts = TemporalBinner._select_events(
            arrival_times, current_start, arrival_times[-1]
        )

        # if it does not, the flag for the big loop never gets set
        end_all_search = not TemporalBinner._check_exceeds_sigma_interval(
            current_start,
            arrival_times[-1],
            counts,
            sigma_level,
            background_getter,
            background_error_getter,
        )

        # We will start the search at the mid point of the whole interval

        mid_point = 0.5 * (arrival_times[-1] + current_start)

        current_stop = mid_point

        # initialize the fast search flag

        end_fast_search = False

        # resolve once for functions used in the loop
        searchsorted = np.searchsorted

        # this is the main loop
        # as long as we have not reached the end of the interval
        # the loop will run

        if threeML_config["interface"]["show_progress_bars"]:
            pbar = tqdm(
                total=arrival_times.shape[0], desc="Binning by significance")

        while not end_all_search:

            # start of the fast search
            # we reset the flag for the interval
            # having been decreased in the last pass
            decreased_interval = False

            while not end_fast_search:

                # we calculate the sigma of the current region
                _, counts = TemporalBinner._select_events(
                    arrival_times, current_start, current_stop
                )

                sigma_exceeded = TemporalBinner._check_exceeds_sigma_interval(
                    current_start,
                    current_stop,
                    counts,
                    sigma_level,
                    background_getter,
                    background_error_getter,
                )

                time_step = abs(current_stop - current_start)

                # if we do not exceed the sigma
                # we need to increase the time interval
                if not sigma_exceeded:

                    # however, if in the last pass we had to decrease
                    # the interval, it means we have found where we
                    # we need to start the slow search
                    if decreased_interval:

                        # mark where we are in the list
                        start_idx = searchsorted(arrival_times, current_stop)

                        # end the fast search
                        end_fast_search = True

                    # otherwise we increase the interval
                    else:

                        # unless, we would increase it too far
                        if (
                            current_stop + time_step * increase_factor
                        ) >= arrival_times[-1]:

                            # mark where we are in the interval
                            start_idx = searchsorted(
                                arrival_times, current_stop)

                            # then we also want to go ahead and get out of the fast search
                            end_fast_search = True

                        else:

                            # increase the interval
                            current_stop += time_step * increase_factor

                # if we did exceede the sigma level we will need to step
                # back in time to find where it was NOT exceeded
                else:

                    # decrease the interval
                    current_stop -= time_step * decrease_factor

                    # inform the loop that we have been back stepping
                    decreased_interval = True

            # Now we are ready for the slow forward search
            # where we count up all the photons

            # we have already counted up the photons to this point
            total_counts = counts

            # start searching from where the fast search ended
            if threeML_config["interface"]["show_progress_bars"]:
                pbar.update(counts)

            for time in arrival_times[start_idx:]:

                total_counts += 1
                if threeML_config["interface"]["show_progress_bars"]:
                    pbar.update(1)
                if total_counts < min_counts:

                    continue

                else:

                    # first use the background function to know the number of background counts
                    bkg = background_getter(current_start, time)

                    sig = Significance(total_counts, bkg)

                    if background_error_getter is not None:

                        bkg_error = background_error_getter(
                            current_start, time)

                        sigma = sig.li_and_ma_equivalent_for_gaussian_background(
                            bkg_error
                        )[0]

                    else:

                        sigma = sig.li_and_ma()[0]

                        # now test if we have enough sigma

                    if sigma >= sigma_level:

                        # if we succeeded we want to mark the time bins
                        stops.append(time)

                        starts.append(current_start)

                        # set up the next fast search
                        # by looking past this interval
                        current_start = time

                        current_stop = 0.5 * (arrival_times[-1] + time)

                        end_fast_search = False

                        # get out of the for loop
                        break

            # if we never exceeded the sigma level by the
            # end of the search, we never will
            if end_fast_search:

                # so lets kill the main search
                end_all_search = True

        if not starts:

            log.error(
                "The requested sigma level could not be achieved in the interval. Try decreasing it."
            )

        else:

            return cls.from_starts_and_stops(starts, stops)

    @classmethod
    def bin_by_constant(cls, arrival_times, dt):
        """
        Create bins with a constant dt

        :param dt: temporal spacing of the bins
        :return: None
        """

        tmp = np.arange(arrival_times[0], arrival_times[-1], dt)
        starts = tmp
        stops = tmp + dt

        return cls.from_starts_and_stops(starts, stops)

    @classmethod
    def bin_by_bayesian_blocks(cls, arrival_times, p0, bkg_integral_distribution=None):
        """Divide a series of events characterized by their arrival time in blocks
        of perceptibly constant count rate. If the background integral distribution
        is given, divide the series in blocks where the difference with respect to
        the background is perceptibly constant.

        :param arrival_times: An iterable (list, numpy.array...) containing the arrival
                         time of the events.
                         NOTE: the input array MUST be time-ordered, and without
                         duplicated entries. To ensure this, you may execute the
                         following code:
                         tt_array = numpy.asarray(self._arrival_times)
                         tt_array = numpy.unique(tt_array)
                         tt_array.sort()
                         before running the algorithm.
        :param p0: The probability of finding a variations (i.e., creating a new
                      block) when there is none. In other words, the probability of
                      a Type I error, i.e., rejecting the null-hypothesis when is
                      true. All found variations will have a post-trial significance
                      larger than p0.
        :param bkg_integral_distribution : the integral distribution for the
                      background counts. It must be a function of the form f(x),
                      which must return the integral number of counts expected from
                      the background component between time 0 and x.

        """

        try:

            final_edges = bayesian_blocks(
                arrival_times,
                arrival_times[0],
                arrival_times[-1],
                p0,
                bkg_integral_distribution,
            )

        except Exception as e:

            if "duplicate" in str(e):

                log.warning(
                    "There were possible duplicate time tags in the data. We will try to run a different algorithm"
                )

                final_edges = bayesian_blocks_not_unique(
                    arrival_times, arrival_times[0], arrival_times[-1], p0
                )
            else:

                print(e)

                raise RuntimeError()

        starts = np.asarray(final_edges)[:-1]
        stops = np.asarray(final_edges)[1:]

        return cls.from_starts_and_stops(starts, stops)

    @classmethod
    def bin_by_custom(cls, starts, stops):
        """
        Simplicity function to make custom bins. This form keeps introduction of
        custom bins uniform for other binning methods

        :param start: start times of the bins
        :param stop:  stop times of the bins
        :return:
        """

        return cls.from_starts_and_stops(starts, stops)

    @staticmethod
    def _check_exceeds_sigma_interval(
        start,
        stop,
        counts,
        sigma_level,
        background_getter,
        background_error_getter=None,
    ):
        """

        see if an interval exceeds a given sigma level


        :param start:
        :param stop:
        :param counts:
        :param sigma_level:
        :param background_getter:
        :param background_error_getter:
        :return:
        """

        bkg = background_getter(start, stop)

        sig = Significance(counts, bkg)

        if background_error_getter is not None:

            bkg_error = background_error_getter(start, stop)

            sigma = sig.li_and_ma_equivalent_for_gaussian_background(bkg_error)[
                0]

        else:

            sigma = sig.li_and_ma()[0]

        # now test if we have enough sigma

        if sigma >= sigma_level:

            return True

        else:

            return False

    @staticmethod
    def _select_events(arrival_times, start, stop):
        """
        get the events and total counts over an interval

        :param start:
        :param stop:
        :param events:
        :return:
        """

        lt_idx = start <= arrival_times
        gt_idx = arrival_times <= stop

        idx = np.logical_and(lt_idx, gt_idx)

        return idx, arrival_times[idx].shape[0]


#####
@nb.njit(fastmath=True)
def _rebin_vector_float(vector, start, stop, mask, N):
    """
    faster rebinner using numba
    """
    rebinned_vector = VectorFloat64(0)

    for n in range(N):

        rebinned_vector.append(np.sum(vector[start[n]:stop[n]]))

    arr = rebinned_vector.arr

    test = np.abs(
        (np.sum(arr) + 1e-100)
        / (np.sum(vector[mask]) + 1e-100)
        - 1
    )

    assert (

        test < 1e-4
    )

    return arr


@nb.njit(fastmath=True)
def _rebin_vector_int(vector, start, stop, mask, N):
    """
    faster rebinner using numba
    """
    rebinned_vector = VectorInt64(0)

    for n in range(N):

        rebinned_vector.append(np.sum(vector[start[n]:stop[n]]))

    arr = rebinned_vector.arr

    test = np.abs(
        (np.sum(arr) + 1e-100)
        / (np.sum(vector[mask]) + 1e-100)
        - 1
    )

    assert (

        test < 1e-4
    )

    return arr
