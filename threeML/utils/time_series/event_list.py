from __future__ import division, print_function

from builtins import range, zip

from past.utils import old_div

__author__ = "grburgess"

import collections
import copy
import os

import numpy as np
import pandas as pd
from pandas import HDFStore
from threeML.utils.progress_bar import tqdm, trange

from threeML.config.config import threeML_config
from threeML.exceptions.custom_exceptions import custom_warnings
from threeML.io.file_utils import sanitize_filename
from threeML.io.logging import setup_logger, silence_console_log
from threeML.io.plotting.light_curve_plots import binned_light_curve_plot
from threeML.io.rich_display import display
from threeML.parallel.parallel_client import ParallelClient
from threeML.utils.binner import TemporalBinner
from threeML.utils.time_interval import TimeIntervalSet
from threeML.utils.time_series.polynomial import polyfit, unbinned_polyfit
from threeML.utils.time_series.time_series import TimeSeries

log = setup_logger(__name__)


class ReducingNumberOfThreads(Warning):
    pass


class ReducingNumberOfSteps(Warning):
    pass


class OverLappingIntervals(RuntimeError):
    pass


# find out how many splits we need to make
def ceildiv(a, b):
    return -(-a // b)


class EventList(TimeSeries):
    def __init__(
        self,
        arrival_times,
        measurement,
        n_channels,
        start_time=None,
        stop_time=None,
        native_quality=None,
        first_channel=0,
        ra=None,
        dec=None,
        mission=None,
        instrument=None,
        verbose=True,
        edges=None,
    ):
        """
        The EventList is a container for event data that is tagged in time and in PHA/energy. It handles event selection,
        temporal polynomial fitting, temporal binning, and exposure calculations (in subclasses). Once events are selected
        and/or polynomials are fit, the selections can be extracted via a PHAContainer which is can be read by an OGIPLike
        instance and translated into a PHA instance.


        :param  n_channels: Number of detector channels
        :param  start_time: start time of the event list
        :param  stop_time: stop time of the event list
        :param  first_channel: where detchans begin indexing
        :param  rsp_file: the response file corresponding to these events
        :param  arrival_times: list of event arrival times
        :param  measurement: list of event energies or pha channels
        :param native_quality: native pha quality flags
        :param edges: The histogram boundaries if not specified by a response
        :param mission:
        :param instrument:
        :param verbose:
        :param  ra:
        :param  dec:
        """

        # pass up to TimeSeries

        super(EventList, self).__init__(
            start_time,
            stop_time,
            n_channels,
            native_quality,
            first_channel,
            ra,
            dec,
            mission,
            instrument,
            verbose,
            edges,
        )

        self._arrival_times = np.asarray(arrival_times)
        self._measurement = np.asarray(measurement)

        self._temporal_binner = None

        assert (
            self._arrival_times.shape[0] == self._measurement.shape[0]
        ), "Arrival time (%d) and energies (%d) have different shapes" % (
            self._arrival_times.shape[0],
            self._measurement.shape[0],
        )

    @property
    def n_events(self):

        return self._arrival_times.shape[0]

    @property
    def arrival_times(self):

        return self._arrival_times

    @property
    def measurement(self):
        return self._measurement

    @property
    def bins(self):

        if self._temporal_binner is not None:

            return self._temporal_binner
        else:

            raise RuntimeError("This EventList has no binning specified")

    def bin_by_significance(self, start, stop, sigma, mask=None, min_counts=1):
        """

        Interface to the temporal binner's significance binning model

         :param start: start of the interval to bin on
         :param stop:  stop of the interval ot bin on
         :param sigma: sigma-level of the bins
         :param mask: (bool) use the energy mask to decide on ,significance
         :param min_counts:  minimum number of counts per bin
         :return:
        """

        if mask is not None:

            # create phas to check
            phas = np.arange(self._first_channel, self._n_channels)[mask]

            this_mask = np.zeros_like(self._arrival_times, dtype=np.bool)

            for channel in phas:
                this_mask = np.logical_or(
                    this_mask, self._measurement == channel)

            events = self._arrival_times[this_mask]

        else:

            events = copy.copy(self._arrival_times)

        events = events[np.logical_and(events <= stop, events >= start)]

        def tmp_bkg_getter(a, b): return self.get_total_poly_count(a, b, mask)
        def tmp_err_getter(a, b): return self.get_total_poly_error(a, b, mask)

        # self._temporal_binner.bin_by_significance(tmp_bkg_getter,
        #                                           background_error_getter=tmp_err_getter,
        #                                           sigma_level=sigma,
        #                                           min_counts=min_counts)

        self._temporal_binner = TemporalBinner.bin_by_significance(
            events,
            tmp_bkg_getter,
            background_error_getter=tmp_err_getter,
            sigma_level=sigma,
            min_counts=min_counts,
        )

    def bin_by_constant(self, start, stop, dt=1):
        """
        Interface to the temporal binner's constant binning mode

        :param start: start time of the bins
        :param stop: stop time of the bins
        :param dt: temporal spacing of the bins
        :return:
        """

        events = self._arrival_times[
            np.logical_and(self._arrival_times >= start,
                           self._arrival_times <= stop)
        ]

        self._temporal_binner = TemporalBinner.bin_by_constant(events, dt)

    def bin_by_custom(self, start, stop):
        """
        Interface to temporal binner's custom bin mode


        :param start: start times of the bins
        :param stop:  stop times of the bins
        :return:
        """

        self._temporal_binner = TemporalBinner.bin_by_custom(start, stop)
        # self._temporal_binner.bin_by_custom(start, stop)

    def bin_by_bayesian_blocks(self, start, stop, p0, use_background=False):

        events = self._arrival_times[
            np.logical_and(self._arrival_times >= start,
                           self._arrival_times <= stop)
        ]

        # self._temporal_binner = TemporalBinner(events)

        if use_background:

            def integral_background(
                t): return self.get_total_poly_count(start, t)

            self._temporal_binner = TemporalBinner.bin_by_bayesian_blocks(
                events, p0, bkg_integral_distribution=integral_background
            )

        else:

            self._temporal_binner = TemporalBinner.bin_by_bayesian_blocks(
                events, p0)

    def view_lightcurve(self, start=-10, stop=20.0, dt=1.0, use_binner=False):
        # type: (float, float, float, bool) -> None
        """
        :param start:
        :param stop:
        :param dt:
        :param use_binner:

        """

        if use_binner:

            # we will use the binner object to bin the
            # light curve and ignore the normal linear binning

            bins = self.bins.time_edges

            # perhaps we want to look a little before or after the binner
            if start < bins[0]:
                pre_bins = np.arange(start, bins[0], dt).tolist()[:-1]

                pre_bins.extend(bins)

                bins = pre_bins

            if stop > bins[-1]:
                post_bins = np.arange(bins[-1], stop, dt)

                bins.extend(post_bins[1:])

        else:

            # otherwise, just use regular linear binning

            bins = np.arange(start, stop + dt, dt)

        cnts, bins = np.histogram(self.arrival_times, bins=bins)
        time_bins = np.array([[bins[i], bins[i + 1]]
                              for i in range(len(bins) - 1)])

        # width = np.diff(bins)
        width = []

        # now we want to get the estimated background from the polynomial fit

        if self.poly_fit_exists:

            # we will store the bkg rate for each time bin

            bkg = []

            for j, tb in enumerate(time_bins):

                # zero out the bkg
                tmpbkg = 0.0

                # we will use the exposure for the width

                this_width = self.exposure_over_interval(tb[0], tb[1])

                # sum up the counts over this interval

                for poly in self.polynomials:

                    tmpbkg += poly.integral(tb[0], tb[1])

                # capture the exposure

                width.append(this_width)

                # capture the bkg *rate*

                bkg.append(old_div(tmpbkg, this_width))

        else:

            bkg = None

            for j, tb in enumerate(time_bins):

                this_width = self.exposure_over_interval(tb[0], tb[1])

                width.append(this_width)

        width = np.array(width)

        # pass all this to the light curve plotter

        if self.time_intervals is not None:

            selection = self.time_intervals.bin_stack

        else:

            selection = None

        if self.poly_intervals is not None:

            bkg_selection = self.poly_intervals.bin_stack

        else:

            bkg_selection = None

        return binned_light_curve_plot(
            time_bins=time_bins,
            cnts=cnts,
            width=width,
            bkg=bkg,
            selection=selection,
            bkg_selections=bkg_selection,
        )

    def counts_over_interval(self, start, stop):
        """
        return the number of counts in the selected interval
        :param start: start of interval
        :param stop:  stop of interval
        :return:
        """

        # this will be a boolean list and the sum will be the
        # number of events

        return self._select_events(start, stop).sum()

    def count_per_channel_over_interval(self, start, stop):

        channels = list(
            range(self._first_channel, self._n_channels + self._first_channel)
        )

        counts_per_channel = np.zeros(len(channels))

        selection = self._select_events(start, stop)

        for i, channel in enumerate(channels):
            channel_mask = self._measurement[selection] == channel

            counts_per_channel[i] += channel_mask.sum()

        return counts_per_channel

    def _select_events(self, start, stop):
        """
        return an index of the selected events
        :param start: start time
        :param stop: stop time
        :return:
        """

        return np.logical_and(start <= self._arrival_times, self._arrival_times <= stop)

    def _fit_polynomials(self, bayes=False):
        """

        Binned fit to each channel. Sets the polynomial array that will be used to compute
        counts over an interval



        :return:
        """

        self._poly_fit_exists = True

        # Select all the events that are in the background regions
        # and make a mask

        all_bkg_masks = []

        for selection in self._poly_intervals:
            all_bkg_masks.append(
                np.logical_and(
                    self._arrival_times >= selection.start_time,
                    self._arrival_times <= selection.stop_time,
                )
            )
        poly_mask = all_bkg_masks[0]

        # If there are multiple masks:
        if len(all_bkg_masks) > 1:
            for mask in all_bkg_masks[1:]:
                poly_mask = np.logical_or(poly_mask, mask)

        # Select the all the events in the poly selections
        # We only need to do this once

        total_poly_events = self._arrival_times[poly_mask]

        # For the channel energies we will need to down select again.
        # We can go ahead and do this to avoid repeated computations

        total_poly_energies = self._measurement[poly_mask]

        # This calculation removes the unselected portion of the light curve
        # so that we are not fitting zero counts. It will be used in the channel calculations
        # as well

        bin_width = 1.0  # seconds
        these_bins = np.arange(self._start_time, self._stop_time, bin_width)

        cnts, bins = np.histogram(total_poly_events, bins=these_bins)

        # Find the mean time of the bins and calculate the exposure in each bin
        mean_time = []
        exposure_per_bin = []
        for i in range(len(bins) - 1):
            m = np.mean((bins[i], bins[i + 1]))
            mean_time.append(m)

            exposure_per_bin.append(
                self.exposure_over_interval(bins[i], bins[i + 1]))

        mean_time = np.array(mean_time)

        exposure_per_bin = np.array(exposure_per_bin)

        # Remove bins with zero counts
        all_non_zero_mask = []

        for selection in self._poly_intervals:
            all_non_zero_mask.append(
                np.logical_and(
                    mean_time >= selection.start_time, mean_time <= selection.stop_time
                )
            )

        non_zero_mask = all_non_zero_mask[0]
        if len(all_non_zero_mask) > 1:
            for mask in all_non_zero_mask[1:]:
                non_zero_mask = np.logical_or(mask, non_zero_mask)

        # Now we will find the the best poly order unless the use specified one
        # The total cnts (over channels) is binned to .1 sec intervals

        if self._user_poly_order == -1:

            

            self._optimal_polynomial_grade = (
                self._fit_global_and_determine_optimum_grade(
                    cnts[non_zero_mask],
                    mean_time[non_zero_mask],
                    exposure_per_bin[non_zero_mask],
                    bayes=bayes
                )
            )

            log.info(
                "Auto-determined polynomial order: %d" % self._optimal_polynomial_grade
            )

        else:

            self._optimal_polynomial_grade = self._user_poly_order

        channels = list(
            range(self._first_channel, self._n_channels + self._first_channel)
        )

        if threeML_config["parallel"]["use-parallel"]:

            def worker(channel):

                channel_mask = total_poly_energies == channel

                # Mask background events and current channel
                # poly_chan_mask = np.logical_and(poly_mask, channel_mask)
                # Select the masked events

                current_events = total_poly_events[channel_mask]

                cnts, bins = np.histogram(current_events, bins=these_bins)

                polynomial, _ = polyfit(
                    mean_time[non_zero_mask],
                    cnts[non_zero_mask],
                    self._optimal_polynomial_grade,
                    exposure_per_bin[non_zero_mask],
                    bayes=bayes
                )

                return polynomial

            client = ParallelClient()



            polynomials = client.execute_with_progress_bar(
                    worker, channels, name=f"Fitting {self._instrument} background")

        else:

            polynomials = []

            

            for channel in tqdm(channels, desc=f"Fitting {self._instrument} background"):

                channel_mask = total_poly_energies == channel

                # Mask background events and current channel
                # poly_chan_mask = np.logical_and(poly_mask, channel_mask)
                # Select the masked events

                current_events = total_poly_events[channel_mask]

                # now bin the selected channel counts

                cnts, bins = np.histogram(current_events, bins=these_bins)

                # Put data to fit in an x vector and y vector

                polynomial, _ = polyfit(
                    mean_time[non_zero_mask],
                    cnts[non_zero_mask],
                    self._optimal_polynomial_grade,
                    exposure_per_bin[non_zero_mask],
                    bayes=bayes
                )

                polynomials.append(polynomial)

        # We are now ready to return the polynomials

        self._polynomials = polynomials

    def _unbinned_fit_polynomials(self, bayes=False):

        self._poly_fit_exists = True

        # Select all the events that are in the background regions
        # and make a mask

        all_bkg_masks = []

        total_duration = 0.0

        poly_exposure = 0

        for selection in self._poly_intervals:
            total_duration += selection.duration

            poly_exposure += self.exposure_over_interval(
                selection.start_time, selection.stop_time
            )

            all_bkg_masks.append(
                np.logical_and(
                    self._arrival_times >= selection.start_time,
                    self._arrival_times <= selection.stop_time,
                )
            )
        poly_mask = all_bkg_masks[0]

        # If there are multiple masks:
        if len(all_bkg_masks) > 1:
            for mask in all_bkg_masks[1:]:
                poly_mask = np.logical_or(poly_mask, mask)

        # Select the all the events in the poly selections
        # We only need to do this once

        total_poly_events = self._arrival_times[poly_mask]

        # For the channel energies we will need to down select again.
        # We can go ahead and do this to avoid repeated computations

        total_poly_energies = self._measurement[poly_mask]

        # Now we will find the the best poly order unless the use specified one
        # The total cnts (over channels) is binned to .1 sec intervals

        if self._user_poly_order == -1:


            self._optimal_polynomial_grade = (
                self._unbinned_fit_global_and_determine_optimum_grade(
                    total_poly_events, poly_exposure, bayes=bayes
                )
            )

            log.info(
                "Auto-determined polynomial order: %d" % self._optimal_polynomial_grade
            )

        else:

            self._optimal_polynomial_grade = self._user_poly_order

        channels = list(
            range(self._first_channel, self._n_channels + self._first_channel)
        )

        # Check whether we are parallelizing or not

        t_start = self._poly_intervals.start_times
        t_stop = self._poly_intervals.stop_times

        if threeML_config["parallel"]["use-parallel"]:

            def worker(channel):
                channel_mask = total_poly_energies == channel

                # Mask background events and current channel
                # poly_chan_mask = np.logical_and(poly_mask, channel_mask)
                # Select the masked events

                current_events = total_poly_events[channel_mask]

                polynomial, _ = unbinned_polyfit(
                    current_events,
                    self._optimal_polynomial_grade,
                    t_start,
                    t_stop,
                    poly_exposure,
                    bayes=bayes
                )

                return polynomial

            client = ParallelClient()


            polynomials = client.execute_with_progress_bar(
                    worker, channels, name=f"Fitting {self._instrument} background")

        else:

            polynomials = []

            

            for channel in tqdm(channels, desc=f"Fitting {self._instrument} background"):
                channel_mask = total_poly_energies == channel

                # Mask background events and current channel
                # poly_chan_mask = np.logical_and(poly_mask, channel_mask)
                # Select the masked events

                current_events = total_poly_events[channel_mask]

                polynomial, _ = unbinned_polyfit(
                    current_events,
                    self._optimal_polynomial_grade,
                    t_start,
                    t_stop,
                    poly_exposure,
                    bayes=bayes
                )

                polynomials.append(polynomial)

        # We are now ready to return the polynomials

        self._polynomials = polynomials


class EventListWithDeadTime(EventList):
    def __init__(
        self,
        arrival_times,
        measurement,
        n_channels,
        start_time=None,
        stop_time=None,
        dead_time=None,
        first_channel=0,
        quality=None,
        ra=None,
        dec=None,
        mission=None,
        instrument=None,
        verbose=True,
        edges=None,
    ):
        """
        An EventList where the exposure is calculated via and array of dead times per event. Summing these dead times over an
        interval => live time = interval - dead time



        :param  n_channels: Number of detector channels
        :param  start_time: start time of the event list
        :param  stop_time: stop time of the event list
        :param  dead_time: an array of deadtime per event
        :param  first_channel: where detchans begin indexing
        :param  quality: native pha quality flags
        :param  rsp_file: the response file corresponding to these events
        :param  arrival_times: list of event arrival times
        :param  measurement: list of event energies or pha channels
        :param edges: The histogram boundaries if not specified by a response
        :param  mission: mission name
        :param  instrument: instrument name
        :param  verbose: verbose level
        :param  ra:
        :param  dec:
        """

        super(EventListWithDeadTime, self).__init__(
            arrival_times,
            measurement,
            n_channels,
            start_time,
            stop_time,
            quality,
            first_channel,
            ra,
            dec,
            mission,
            instrument,
            verbose,
            edges,
        )

        if dead_time is not None:

            self._dead_time = np.asarray(dead_time)

            assert (
                self._arrival_times.shape[0] == self._dead_time.shape[0]
            ), "Arrival time (%d) and Dead Time (%d) have different shapes" % (
                self._arrival_times.shape[0],
                self._dead_time.shape[0],
            )

        else:

            self._dead_time = None

    def exposure_over_interval(self, start, stop):
        """
        calculate the exposure over the given interval

        :param start: start time
        :param stop:  stop time
        :return:
        """

        mask = self._select_events(start, stop)

        if self._dead_time is not None:

            interval_deadtime = (self._dead_time[mask]).sum()

        else:

            interval_deadtime = 0

        return (stop - start) - interval_deadtime

    def set_active_time_intervals(self, *args):
        """Set the time interval(s) to be used during the analysis.

        Specified as 'tmin-tmax'. Intervals are in seconds. Example:

        set_active_time_intervals("0.0-10.0")

        which will set the energy range 0-10. seconds.
        """

        self._time_selection_exists = True

        interval_masks = []

        time_intervals = TimeIntervalSet.from_strings(*args)

        time_intervals.merge_intersecting_intervals(in_place=True)

        for interval in time_intervals:
            tmin = interval.start_time
            tmax = interval.stop_time

            mask = self._select_events(tmin, tmax)

            interval_masks.append(mask)

        self._time_intervals = time_intervals

        time_mask = interval_masks[0]
        if len(interval_masks) > 1:
            for mask in interval_masks[1:]:
                time_mask = np.logical_or(time_mask, mask)

        tmp_counts = []  # Temporary list to hold the total counts per chan

        for chan in range(self._first_channel, self._n_channels + self._first_channel):

            channel_mask = self._measurement == chan
            counts_mask = np.logical_and(channel_mask, time_mask)
            total_counts = len(self._arrival_times[counts_mask])

            tmp_counts.append(total_counts)

        self._counts = np.array(tmp_counts)

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

        # Dead time correction

        exposure = 0.0
        for interval in self._time_intervals:
            exposure += interval.duration

        if self._dead_time is not None:

            total_dead_time = self._dead_time[time_mask].sum()
        else:

            total_dead_time = 0.0

        self._exposure = exposure - total_dead_time

        self._active_dead_time = total_dead_time


class EventListWithDeadTimeFraction(EventList):
    def __init__(
        self,
        arrival_times,
        measurement,
        n_channels,
        start_time=None,
        stop_time=None,
        dead_time_fraction=None,
        first_channel=0,
        quality=None,
        ra=None,
        dec=None,
        mission=None,
        instrument=None,
        verbose=True,
        edges=None,
    ):
        """
        An EventList where the exposure is calculated via and array dead time fractions per event .
        Summing these dead times over an
        interval => live time = interval - dead time



        :param  n_channels: Number of detector channels
        :param  start_time: start time of the event list
        :param  stop_time: stop time of the event list
        :param  dead_time: an array of deadtime fraction
        :param  first_channel: where detchans begin indexing
        :param  quality: native pha quality flags
        :param  rsp_file: the response file corresponding to these events
        :param  arrival_times: list of event arrival times
        :param  measurement: list of event energies or pha channels
        :param edges: The histogram boundaries if not specified by a response
        :param  mission: mission name
        :param  instrument: instrument name
        :param  verbose: verbose level
        :param  ra:
        :param  dec:
        """

        super(EventListWithDeadTimeFraction, self).__init__(
            arrival_times,
            measurement,
            n_channels,
            start_time,
            stop_time,
            quality,
            first_channel,
            ra,
            dec,
            mission,
            instrument,
            verbose,
            edges,
        )

        if dead_time_fraction is not None:

            self._dead_time_fraction = np.asarray(dead_time_fraction)

            assert (
                self._arrival_times.shape[0] == self._dead_time_fraction.shape[0]
            ), "Arrival time (%d) and Dead Time (%d) have different shapes" % (
                self._arrival_times.shape[0],
                self._dead_time_fraction.shape[0],
            )

        else:

            self._dead_time_fraction = None

    def exposure_over_interval(self, start, stop):
        """
        calculate the exposure over the given interval

        :param start: start time
        :param stop:  stop time
        :return:
        """

        mask = self._select_events(start, stop)

        interval = stop - start

        if self._dead_time_fraction is not None:

            interval_deadtime = (
                self._dead_time_fraction[mask]).mean() * interval

        else:

            interval_deadtime = 0

        return interval - interval_deadtime

    def set_active_time_intervals(self, *args):
        """Set the time interval(s) to be used during the analysis.

        Specified as 'tmin-tmax'. Intervals are in seconds. Example:

        set_active_time_intervals("0.0-10.0")

        which will set the energy range 0-10. seconds.
        """

        self._time_selection_exists = True

        interval_masks = []

        time_intervals = TimeIntervalSet.from_strings(*args)

        time_intervals.merge_intersecting_intervals(in_place=True)

        for interval in time_intervals:
            tmin = interval.start_time
            tmax = interval.stop_time

            mask = self._select_events(tmin, tmax)

            interval_masks.append(mask)

        self._time_intervals = time_intervals

        time_mask = interval_masks[0]
        if len(interval_masks) > 1:
            for mask in interval_masks[1:]:
                time_mask = np.logical_or(time_mask, mask)

        tmp_counts = []  # Temporary list to hold the total counts per chan

        for chan in range(self._first_channel, self._n_channels + self._first_channel):
            channel_mask = self._measurement == chan
            counts_mask = np.logical_and(channel_mask, time_mask)
            total_counts = len(self._arrival_times[counts_mask])

            tmp_counts.append(total_counts)

        self._counts = np.array(tmp_counts)

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

        # Dead time correction

        exposure = 0.0
        total_dead_time = 0.0
        for interval, imask in zip(self._time_intervals, interval_masks):
            exposure += interval.duration
            if self._dead_time_fraction is not None:
                total_dead_time += (
                    interval.duration * self._dead_time_fraction[imask].mean()
                )

        self._exposure = exposure - total_dead_time

        self._active_dead_time = total_dead_time


class EventListWithLiveTime(EventList):
    def __init__(
        self,
        arrival_times,
        measurement,
        n_channels,
        live_time,
        live_time_starts,
        live_time_stops,
        start_time=None,
        stop_time=None,
        quality=None,
        first_channel=0,
        rsp_file=None,
        ra=None,
        dec=None,
        mission=None,
        instrument=None,
        verbose=True,
        edges=None,
    ):
        """
        An EventList where the exposure is calculated via and array of livetimes per interval.



        :param  arrival_times: list of event arrival times
        :param  measurement: list of event energies or pha channels
        :param live_time: array of livetime fractions
        :param live_time_starts: start of livetime fraction bins
        :param live_time_stops:  stop of livetime fraction bins
        :param mission: mission name
        :param instrument: instrument name
        :param  n_channels: Number of detector channels
        :param  start_time: start time of the event list
        :param  stop_time: stop time of the event list
        :param quality: native pha quality flags
        :param  first_channel: where detchans begin indexing
        :param edges: The histogram boundaries if not specified by a response
        :param  rsp_file: the response file corresponding to these events
        :param verbose:
        :param  ra:
        :param  dec:
        """

        assert len(live_time) == len(
            live_time_starts
        ), "Live time fraction (%d) and live time start (%d) have different shapes" % (
            len(live_time),
            len(live_time_starts),
        )

        assert len(live_time) == len(
            live_time_stops
        ), "Live time fraction (%d) and live time stop (%d) have different shapes" % (
            len(live_time),
            len(live_time_stops),
        )

        super(EventListWithLiveTime, self).__init__(
            arrival_times,
            measurement,
            n_channels,
            start_time,
            stop_time,
            quality,
            first_channel,
            ra,
            dec,
            mission,
            instrument,
            verbose,
        )

        self._live_time = np.asarray(live_time)
        self._live_time_starts = np.asarray(live_time_starts)
        self._live_time_stops = np.asarray(live_time_stops)

    def exposure_over_interval(self, start, stop):
        """

        :param start: start time of interval
        :param stop: stop time of interval
        :return: exposure
        """

        # First see if the interval is completely contained inside a
        # livetime interval. In this case we only compute that.

        # Note that this is explicitly on the open boundary of the
        # intervals because the closed boundary is covered in the
        # next step

        inside_idx = np.logical_and(
            self._live_time_starts < start, stop < self._live_time_stops
        )

        # see if it contains elements

        if self._live_time[inside_idx].size > 0:

            # we want to take a fraction of the live time covered

            dt = self._live_time_stops[inside_idx] - \
                self._live_time_starts[inside_idx]

            fraction = old_div((stop - start), dt)

            total_livetime = self._live_time[inside_idx] * fraction

        else:

            # First we get the live time of bins that are fully contained in the given interval
            # We now go for the closed interval because it is possible to have overlap with other intervals
            # when a closed interval exists... but not when there is only an open interval

            full_inclusion_idx = np.logical_and(
                start <= self._live_time_starts, stop >= self._live_time_stops
            )

            full_inclusion_livetime = self._live_time[full_inclusion_idx].sum()

            # Now we get the fractional parts on the left and right

            # Get the fractional part of the left bin

            left_remainder_idx = np.logical_and(
                start <= self._live_time_stops, self._live_time_starts <= start
            )

            dt = (
                self._live_time_stops[left_remainder_idx]
                - self._live_time_starts[left_remainder_idx]
            )

            # we want the distance to the stop of this bin

            distance_from_next_bin = self._live_time_stops[left_remainder_idx] - start

            fraction = old_div(distance_from_next_bin, dt)

            left_fractional_livetime = self._live_time[left_remainder_idx] * fraction

            # Get the fractional part of the right bin

            right_remainder_idx = np.logical_and(
                self._live_time_starts <= stop, stop <= self._live_time_stops
            )

            dt = (
                self._live_time_stops[right_remainder_idx]
                - self._live_time_starts[right_remainder_idx]
            )

            # we want the distance from the last full bin

            distance_from_next_bin = stop - \
                self._live_time_starts[right_remainder_idx]

            fraction = old_div(distance_from_next_bin, dt)

            right_fractional_livetime = self._live_time[right_remainder_idx] * fraction

            # sum up all the live time

            total_livetime = (
                full_inclusion_livetime
                + left_fractional_livetime
                + right_fractional_livetime
            )

        # the sum at the end converts all the arrays to floats

        return total_livetime.sum()

    def set_active_time_intervals(self, *args):
        """Set the time interval(s) to be used during the analysis.

        Specified as 'tmin-tmax'. Intervals are in seconds. Example:

        set_active_time_intervals("0.0-10.0")

        which will set the energy range 0-10. seconds.
        """

        self._time_selection_exists = True

        interval_masks = []

        time_intervals = TimeIntervalSet.from_strings(*args)

        time_intervals.merge_intersecting_intervals(in_place=True)

        for interval in time_intervals:
            tmin = interval.start_time
            tmax = interval.stop_time
            mask = self._select_events(tmin, tmax)

            interval_masks.append(mask)

        self._time_intervals = time_intervals

        time_mask = interval_masks[0]
        if len(interval_masks) > 1:
            for mask in interval_masks[1:]:
                time_mask = np.logical_or(time_mask, mask)

        tmp_counts = []  # Temporary list to hold the total counts per chan

        for chan in range(self._first_channel, self._n_channels + self._first_channel):
            channel_mask = self._measurement == chan
            counts_mask = np.logical_and(channel_mask, time_mask)
            total_counts = len(self._arrival_times[counts_mask])

            tmp_counts.append(total_counts)

        self._counts = np.array(tmp_counts)

        tmp_counts = []
        tmp_err = []  # Temporary list to hold the err counts per chan

        if self._poly_fit_exists:

            if not self._poly_fit_exists:
                raise RuntimeError(
                    "A polynomial fit to the channels does not exist!")

            # for chan in range(self._first_channel, self._n_channels + self._first_channel):
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

        # Live time correction

        exposure = 0.0
        total_real_time = 0.0
        for interval in self._time_intervals:
            total_real_time += interval.duration
            exposure += self.exposure_over_interval(
                interval.start_time, interval.stop_time
            )

        # In this case the exposure is the total live time

        self._exposure = exposure
        self._active_dead_time = total_real_time - exposure
