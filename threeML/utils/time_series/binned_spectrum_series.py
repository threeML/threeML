import numpy as np

from threeML.utils.time_series.time_series import TimeSeries
from threeML.plugins.spectrum.binned_spectrum_set import BinnedSpectrumSet
from threeML.utils.time_interval import TimeIntervalSet
from threeML.exceptions.custom_exceptions import custom_warnings
from threeML.io.progress_bar import progress_bar
from threeML.config.config import threeML_config
from threeML.utils.time_series.polynomial import polyfit, Polynomial


class BinnedSpectrumSeries(TimeSeries):

    def __init__(self,binned_spectrum_set,first_channel=1, rsp_file=None, ra=None, dec=None,
                 mission=None, instrument=None, verbose=True):
        # type: (BinnedSpectrumSet, int, str, float, float, str, str, bool) -> None
        """

        :param binned_spectrum_set:
        :param first_channel:
        :param rsp_file:
        :param ra:
        :param dec:
        :param mission:
        :param instrument:
        :param verbose:
        """


        # pass up to TimeSeries

        super(BinnedSpectrumSeries, self).__init__(binned_spectrum_set.time_intervals.absolute_start,
                                                   binned_spectrum_set.time_intervals.absolute_stop,
                                                   binned_spectrum_set.n_channels,
                                                   binned_spectrum_set.qaulity_per_bin[0],
                                                   first_channel,
                                                   rsp_file,
                                                   ra,
                                                   dec,
                                                   mission,
                                                   instrument,
                                                   verbose)


        self._binned_spectrum_set = binned_spectrum_set

    @property
    def bins(self):


        return self._binned_spectrum_set.time_intervals




    def counts_over_interval(self, start, stop):
        """
        return the number of counts in the selected interval
        :param start: start of interval
        :param stop:  stop of interval
        :return:
        """

        # this will be a boolean list and the sum will be the
        # number of events

        bins = self._select_bins(start, stop)

        total_counts = 0

        for idx in bins:

            total_counts += self._binned_spectrum_set[idx].counts.sum()


        return total_counts

    def _select_bins(self, start, stop):
        """
        return an index of the selected bins
        :param start: start time
        :param stop: stop time
        :return: int indexes
        """

        start_idx  = self._binned_spectrum_set.time_to_index(start)
        stop_idx = self._binned_spectrum_set.time_to_index(stop)

        return np.arange(start_idx,stop_idx + 1)

    def _adjust_to_true_intervals(self, time_intervals):
        """

        adjusts time selections to those of the Binned spectrum set


        :param time_intervals: a time interval set
        :return: an adjusted time interval set
        """

        true_starts = self._binned_spectrum_set.time_intervals.start_times
        true_stops = self._binned_spectrum_set.time_intervals.stop_times

        new_starts = []
        new_stops = []

        for interval in time_intervals:
            idx = np.searchsorted(true_starts, interval.start_time)

            new_start = true_starts[idx]

            idx = np.searchsorted(true_stops, interval.stop_time)

            new_stop = true_stops[idx]

            new_starts.append(new_start)

            new_stops.append(new_stop)

        # alright, now we can make appropriate time intervals

        return TimeIntervalSet.from_starts_and_stops(new_starts, new_stops)


    def _fit_polynomials(self, *fit_intervals):
        """
        fits a polynomial to all channels over the input time intervals

        :param fit_intervals: str input intervals
        :return:
        """



        self._poly_fit_exists = True

        self._fit_method_info['bin type'] = 'Binned'
        self._fit_method_info['fit method'] = threeML_config['event list']['binned fit method']

        tmp_poly_intervals = TimeIntervalSet.from_strings(*fit_intervals)

        poly_intervals = self._adjust_to_true_intervals(tmp_poly_intervals)

        selected_counts = []
        selected_exposure = []
        selected_midpoints = []

        for selection in poly_intervals:
            # get the mask of these events
            mask = self._binned_spectrum_set.time_intervals.containing_interval(selection.start_time,
                                                            selection.stop_time,
                                                            as_mask=True)

            selected_counts.append(self._binned_spectrum_set.counts_per_bin[mask])
            selected_exposure.append(self._binned_spectrum_set.exposure_per_bin[mask])
            selected_midpoints.append(self._time_intervals.half_times[mask])

        selected_counts = np.array(selected_counts)

        optimal_polynomial_grade = self._fit_global_and_determine_optimum_grade(selected_counts.sum(axis=1),
                                                                          selected_midpoints,
                                                                          selected_exposure)
        # if self._verbose:
        #     print("Auto-determined polynomial order: %d" % optimal_polynomial_grade)
        #     print('\n')

        n_channels = selected_counts.shape[1]

        polynomials = []

        with progress_bar(n_channels, title="Fitting background") as p:
            for counts in selected_counts.T:
                polynomial, _ = polyfit(counts,
                                        selected_midpoints,
                                        optimal_polynomial_grade,
                                        selected_exposure)

                polynomials.append(polynomial)
                p.increase()

        self._polynomials = polynomials


class BinnedSpectrumSeriesWithDeadTime(BinnedSpectrumSeries):

    def __init__(self, binned_spectrum_set, dead_time, first_channel=1, rsp_file=None, ra=None, dec=None,
                 mission=None, instrument=None, verbose=True):
        # type: (BinnedSpectrumSet, np.ndarray, int, str, float, float, str, str, bool) -> None
        """

        :param binned_spectrum_set:
        :param deadtime:
        :param first_channel:
        :param rsp_file:
        :param ra:
        :param dec:
        :param mission:
        :param instrument:
        :param verbose:
        """


        assert len(dead_time) == len(binned_spectrum_set), 'deadtime per bin is not the same length as all the bins'

        super(BinnedSpectrumSeriesWithDeadTime, self).__init__(binned_spectrum_set,
                                                               first_channel,
                                                               rsp_file,
                                                               ra,
                                                               dec,
                                                               mission,
                                                               instrument,
                                                               verbose)


        self._dead_time = np.array(dead_time)

    def set_active_time_intervals(self, *args):
        """
        Set the time interval(s) to be used during the analysis.

        Specified as 'tmin-tmax'. Intervals are in seconds. Example:

        set_active_time_intervals("0.0-10.0")

        which will set the energy range 0-10. seconds.
        """

        self._time_selection_exists = True



        time_intervals = TimeIntervalSet.from_strings(*args)

        time_intervals.merge_intersecting_intervals(in_place=True)


        # lets adjust the time intervals to the actual ones since they are prebinned


        time_intervals = self._adjust_to_true_intervals(time_intervals)

        self._counts = np.zeros(self._n_channels)


        # now we need to sum up the counts, figure out the interval indices

        interval_indices = []

        for interval in time_intervals:
            self._counts += self.counts_over_interval(interval.start,interval.stop)

            interval_indices.extend(self._select_bins(interval.start,interval.stop))


        self._time_intervals = time_intervals


        tmp_counts = []
        tmp_err = []  # Temporary list to hold the err counts per chan

        if self._poly_fit_exists:

            if not self._poly_fit_exists:
                raise RuntimeError('A polynomial fit to the channels does not exist!')

            for chan in range(self._n_channels):

                total_counts = 0
                counts_err = 0

                for tmin, tmax in zip(self._time_intervals.start_times, self._time_intervals.stop_times):
                    # Now integrate the appropriate background polynomial
                    total_counts += self._polynomials[chan].integral(tmin, tmax)
                    counts_err += (self._polynomials[chan].integral_error(tmin, tmax)) ** 2

                tmp_counts.append(total_counts)

                tmp_err.append(np.sqrt(counts_err))

            self._poly_counts = np.array(tmp_counts)

            self._poly_count_err = np.array(tmp_err)

        # Dead time correction

        exposure = 0.
        for interval in self._time_intervals:
            exposure += interval.duration

        if self._dead_time is not None:

            total_dead_time = self._dead_time[interval_indices].sum()
        else:

            total_dead_time = 0.

        self._exposure = exposure - total_dead_time

        self._active_dead_time = total_dead_time

    def exposure_over_interval(self, start, stop):
        """
        calculate the exposure over the given interval

        :param start: start time
        :param stop:  stop time
        :return:
        """


        mask = self._select_bins(start, stop)


        # adjust to the actaul start and stop



        if self._dead_time is not None:

            interval_deadtime = (self._dead_time[mask]).sum()

        else:

            interval_deadtime = 0

        return ( - start) - interval_deadtime