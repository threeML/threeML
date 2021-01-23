from __future__ import division, print_function

from builtins import range, zip

import numpy as np
from past.utils import old_div

from threeML.config.config import threeML_config
from threeML.io.logging import setup_logger, silence_console_log
from threeML.io.plotting.light_curve_plots import binned_light_curve_plot
from threeML.parallel.parallel_client import ParallelClient
from threeML.utils.progress_bar import tqdm
from threeML.utils.spectrum.binned_spectrum_set import BinnedSpectrumSet
from threeML.utils.time_interval import TimeIntervalSet
from threeML.utils.time_series.polynomial import polyfit
from threeML.utils.time_series.time_series import TimeSeries

log = setup_logger(__name__)


class BinnedSpectrumSeries(TimeSeries):
    def __init__(
        self,
        binned_spectrum_set,
        first_channel=1,
        ra=None,
        dec=None,
        mission=None,
        instrument=None,
        verbose=True,
    ):
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

        super(BinnedSpectrumSeries, self).__init__(
            binned_spectrum_set.time_intervals.absolute_start,
            binned_spectrum_set.time_intervals.absolute_stop,
            binned_spectrum_set.n_channels,
            binned_spectrum_set.quality_per_bin[0],
            first_channel,
            ra,
            dec,
            mission,
            instrument,
            verbose,
            binned_spectrum_set._binned_spectrum_list[0].edges,
        )

        self._binned_spectrum_set = binned_spectrum_set

    @property
    def bins(self):
        """
        the time bins of the spectrum set
        :return: TimeIntervalSet
        """

        return self._binned_spectrum_set.time_intervals

    @property
    def binned_spectrum_set(self):
        """
        returns the spectrum set
        :return: binned_spectrum_set
        """

        return self._binned_spectrum_set

    def view_lightcurve(self, start=-10, stop=20.0, dt=1.0, use_binner=False):
        # type: (float, float, float, bool) -> None
        """
        :param start:
        :param stop:
        :param dt:
        :param use_binner:

        """

        # git a set of bins containing the intervals

        bins = self._binned_spectrum_set.time_intervals.containing_interval(
            start, stop
        )  # type: TimeIntervalSet

        cnts = []
        width = []

        for bin in bins:

            cnts.append(self.counts_over_interval(
                bin.start_time, bin.stop_time))
            width.append(bin.duration)

        # now we want to get the estimated background from the polynomial fit

        if self.poly_fit_exists:

            bkg = []
            for j, tb in enumerate(bins):
                tmpbkg = 0.0
                for poly in self.polynomials:
                    tmpbkg += poly.integral(tb.start_time, tb.stop_time)

                bkg.append(old_div(tmpbkg, width[j]))

        else:

            bkg = None

        # pass all this to the light curve plotter

        if self.time_intervals is not None:

            selection = self.time_intervals.bin_stack

        else:

            selection = None

        if self.poly_intervals is not None:

            bkg_selection = self.poly_intervals.bin_stack

        else:

            bkg_selection = None

        # plot the light curve

        fig = binned_light_curve_plot(
            time_bins=bins.bin_stack,
            cnts=np.array(cnts),
            width=np.array(width),
            bkg=bkg,
            selection=selection,
            bkg_selections=bkg_selection,
        )

        return fig

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

        for idx in np.where(bins)[0]:

            # sum over channels because we just want the total counts

            total_counts += self._binned_spectrum_set[idx].counts.sum()

        return total_counts

    def count_per_channel_over_interval(self, start, stop):
        """
        return the number of counts in the selected interval
        :param start: start of interval
        :param stop:  stop of interval
        :return:
        """

        # this will be a boolean list and the sum will be the
        # number of events

        bins = self._select_bins(start, stop)

        total_counts = np.zeros(self._n_channels)

        for idx in np.where(bins)[0]:

            # don't sum over channels because we want the spectrum
            total_counts += self._binned_spectrum_set[idx].counts

        return total_counts

    def _select_bins(self, start, stop):
        """
        return an index of the selected bins
        :param start: start time
        :param stop: stop time
        :return: int indices
        """

        return self._binned_spectrum_set.time_intervals.containing_interval(
            start, stop, as_mask=True
        )

    def _adjust_to_true_intervals(self, time_intervals):
        """

        adjusts time selections to those of the Binned spectrum set


        :param time_intervals: a time interval set
        :return: an adjusted time interval set
        """

        # get all the starts and stops from these time intervals

        true_starts = np.array(
            self._binned_spectrum_set.time_intervals.start_times)
        true_stops = np.array(
            self._binned_spectrum_set.time_intervals.stop_times)

        new_starts = []
        new_stops = []

        # now go thru all the intervals
        for interval in time_intervals:

            # find where the suggest intervals hits the true interval

            # searchsorted is fast, but is not returing what we want
            # we want the actaul values of the bins closest to the input

            # idx = np.searchsorted(true_starts, interval.start_time,side)

            idx = (np.abs(true_starts - interval.start_time)).argmin()

            new_start = true_starts[idx]

            # idx = np.searchsorted(true_stops, interval.stop_time)

            idx = (np.abs(true_stops - interval.stop_time)).argmin()

            new_stop = true_stops[idx]

            new_starts.append(new_start)

            new_stops.append(new_stop)

        # alright, now we can make appropriate time intervals

        return TimeIntervalSet.from_starts_and_stops(new_starts, new_stops)

    def _fit_polynomials(self, bayes=False):
        """
        fits a polynomial to all channels over the input time intervals

        :param fit_intervals: str input intervals
        :return:
        """

        # mark that we have fit a poly now

        self._poly_fit_exists = True

        # we need to adjust the selection to the true intervals of the time-binned spectra

        tmp_poly_intervals = self._poly_intervals
        poly_intervals = self._adjust_to_true_intervals(tmp_poly_intervals)
        self._poly_intervals = poly_intervals

        # now lets get all the counts, exposure and midpoints for the
        # selection

        selected_counts = []
        selected_exposure = []
        selected_midpoints = []

        for selection in poly_intervals:

            # get the mask of these bins

            mask = self._select_bins(selection.start_time, selection.stop_time)

            # the counts will be (time, channel) here,
            # so the mask is selecting time.
            # a sum along axis=0 is a sum in time, while axis=1 is a sum in energy

            selected_counts.extend(
                self._binned_spectrum_set.counts_per_bin[mask])

            selected_exposure.extend(
                self._binned_spectrum_set.exposure_per_bin[mask])
            selected_midpoints.extend(
                self._binned_spectrum_set.time_intervals.mid_points[mask]
            )

        selected_counts = np.array(selected_counts)
        selected_midpoints = np.array(selected_midpoints)
        selected_exposure = np.array(selected_exposure)

        # Now we will find the the best poly order unless the use specified one
        # The total cnts (over channels) is binned

        if self._user_poly_order == -1:
            with silence_console_log():

                self._optimal_polynomial_grade = (
                    self._fit_global_and_determine_optimum_grade(
                        selected_counts.sum(axis=1),
                        selected_midpoints,
                        selected_exposure,
                        bayes=bayes,
                    )
                )

            log.info(
                "Auto-determined polynomial order: %d"
                % self._optimal_polynomial_grade
            )

        else:

            self._optimal_polynomial_grade = self._user_poly_order

        if threeML_config["parallel"]["use-parallel"]:

            def worker(counts):

                polynomial, _ = polyfit(
                    selected_midpoints,
                    counts,
                    self._optimal_polynomial_grade,
                    selected_exposure,
                    bayes=bayes,
                )

                return polynomial

            client = ParallelClient()

            with silence_console_log():

                polynomials = client.execute_with_progress_bar(
                    worker, selected_counts.T, name=f"Fitting {self._instrument} background")

        else:

            polynomials = []

            # now fit the light curve of each channel
            # and save the estimated polynomial

            with silence_console_log():

                for counts in tqdm(
                    selected_counts.T, desc=f"Fitting {self._instrument} background"
                ):

                    polynomial, _ = polyfit(
                        selected_midpoints,
                        counts,
                        self._optimal_polynomial_grade,
                        selected_exposure,
                        bayes=bayes,
                    )

                    polynomials.append(polynomial)

        self._polynomials = polynomials

    def set_active_time_intervals(self, *args):
        """
        Set the time interval(s) to be used during the analysis.
        Specified as 'tmin-tmax'. Intervals are in seconds. Example:

        set_active_time_intervals("0.0-10.0")

        which will set the time range 0-10. seconds.
        """

        # mark that we now have a time selection

        self._time_selection_exists = True

        # lets build a time interval set from the selections
        # and then merge intersecting intervals

        time_intervals = TimeIntervalSet.from_strings(*args)
        time_intervals.merge_intersecting_intervals(in_place=True)

        # lets adjust the time intervals to the actual ones since they are prebinned

        time_intervals = self._adjust_to_true_intervals(time_intervals)

        # start out with no time bins selection
        all_idx = np.zeros(
            len(self._binned_spectrum_set.time_intervals), dtype=bool)

        # now we need to sum up the counts and total time

        total_time = 0

        for interval in time_intervals:

            # the select bins method is called.
            # since we are sure that the interval bounds
            # are aligned with the true ones, we do not care if
            # it is inner or outer

            all_idx = np.logical_or(
                all_idx, self._select_bins(
                    interval.start_time, interval.stop_time)
            )

            total_time += interval.duration

        # sum along the time axis
        self._counts = self._binned_spectrum_set.counts_per_bin[all_idx].sum(
            axis=0)

        # the selected time intervals

        self._time_intervals = time_intervals

        tmp_counts = []
        tmp_err = []  # Temporary list to hold the err counts per chan

        if self._poly_fit_exists:

            if not self._poly_fit_exists:
                raise RuntimeError(
                    "A polynomial fit to the channels does not exist!")

            for chan in range(self._n_channels):

                total_counts = 0
                counts_err = 0

                for tmin, tmax in zip(
                    self._time_intervals.start_times, self._time_intervals.stop_times
                ):
                    # Now integrate the appropriate background polynomial
                    total_counts += self._polynomials[chan].integral(
                        tmin, tmax)
                    counts_err += (
                        self._polynomials[chan].integral_error(tmin, tmax)
                    ) ** 2

                tmp_counts.append(total_counts)

                tmp_err.append(np.sqrt(counts_err))

            self._poly_counts = np.array(tmp_counts)

            self._poly_count_err = np.array(tmp_err)

        self._exposure = self._binned_spectrum_set.exposure_per_bin[all_idx].sum(
        )

        self._active_dead_time = total_time - self._exposure

    def exposure_over_interval(self, start, stop):
        """
        calculate the exposure over the given interval

        :param start: start time
        :param stop:  stop time
        :return:
        """

        mask = self._select_bins(start, stop)

        return self._binned_spectrum_set.exposure_per_bin[mask].sum()
