__author__='grburgess'

import collections
import copy
import os

import numpy as np
import pandas as pd
from pandas import HDFStore

from threeML.config.config import threeML_config
from threeML.exceptions.custom_exceptions import custom_warnings
from threeML.io.file_utils import sanitize_filename
from threeML.io.progress_bar import progress_bar
from threeML.io.rich_display import display
from threeML.utils.binner import TemporalBinner
from threeML.utils.time_interval import TimeIntervalSet
from threeML.utils.time_series.polynomial import polyfit, unbinned_polyfit, Polynomial
from threeML.utils.time_series.time_series import TimeSeries

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
    def __init__(self, arrival_times, energies, n_channels, start_time=None, stop_time=None,native_quality=None,
                 first_channel=0, rsp_file=None, ra=None, dec=None, mission=None, instrument=None, verbose=True):
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
        :param  energies: list of event energies or pha channels
        :param native_quality: native pha quality flags
        :param mission:
        :param instrument:
        :param verbose:
        :param  ra:
        :param  dec:
        """


        # pass up to TimeSeries

        super(EventList,self).__init__(start_time,stop_time, n_channels ,native_quality,
                 first_channel, rsp_file, ra, dec, mission, instrument, verbose)



        self._arrival_times = np.asarray(arrival_times)
        self._energies = np.asarray(energies)



        assert self._arrival_times.shape[0] == self._energies.shape[
            0], "Arrival time (%d) and energies (%d) have different shapes" % (
            self._arrival_times.shape[0], self._energies.shape[0])




    @property
    def n_events(self):

        return self._arrival_times.shape[0]
    @property
    def arrival_times(self):

        return self._arrival_times


    @property
    def energies(self):
        return self._energies


    @property
    def bins(self):

        if self._temporal_binner is not None:

            return self._temporal_binner
        else:

            raise RuntimeError('This EventList has no binning specified')

    def bin_by_significance(self, start, stop, sigma, mask=None, min_counts=1):
        """

       Interface to the temporal binner's significance binning model

        :param start: start of the interval to bin on
        :param stop:  stop of the interval ot bin on
        :param sigma: sigma-level of the bins
        :param mask: (bool) use the energy mask to decide on significance
        :param min_counts:  minimum number of counts per bin
        :return:
        """

        if mask is not None:

            # create phas to check
            phas = np.arange(self._first_channel, self._n_channels)[mask]

            this_mask = np.zeros_like(self._arrival_times, dtype=np.bool)

            for channel in phas:
                this_mask = np.logical_or(this_mask, self._energies == channel)

            events = self._arrival_times[this_mask]

        else:

            events = copy.copy(self._arrival_times)

        events = events[np.logical_and(events <= stop, events >= start)]



        tmp_bkg_getter = lambda a, b: self.get_total_poly_count(a, b, mask)
        tmp_err_getter = lambda a, b: self.get_total_poly_error(a, b, mask)

        # self._temporal_binner.bin_by_significance(tmp_bkg_getter,
        #                                           background_error_getter=tmp_err_getter,
        #                                           sigma_level=sigma,
        #                                           min_counts=min_counts)

        self._temporal_binner = TemporalBinner.bin_by_significance(events,
                                                                   tmp_bkg_getter,
                                                                   background_error_getter=tmp_err_getter,
                                                                   sigma_level=sigma,
                                                                   min_counts=min_counts)

    def bin_by_constant(self, start, stop, dt=1):
        """
        Interface to the temporal binner's constant binning mode

        :param start: start time of the bins
        :param stop: stop time of the bins
        :param dt: temporal spacing of the bins
        :return:
        """

        events = self._arrival_times[np.logical_and(self._arrival_times >= start, self._arrival_times <= stop)]

        self._temporal_binner = TemporalBinner.bin_by_constant(events, dt)


    def bin_by_custom(self, start, stop):
        """
        Interface to temporal binner's custom bin mode


        :param start: start times of the bins
        :param stop:  stop times of the bins
        :return:
        """

        self._temporal_binner = TemporalBinner.bin_by_custom(start, stop)
        #self._temporal_binner.bin_by_custom(start, stop)

    def bin_by_bayesian_blocks(self, start, stop, p0, use_background=False):

        events = self._arrival_times[np.logical_and(self._arrival_times >= start, self._arrival_times <= stop)]

        #self._temporal_binner = TemporalBinner(events)

        if use_background:

            integral_background = lambda t: self.get_total_poly_count(start, t)

            self._temporal_binner = TemporalBinner.bin_by_bayesian_blocks(events,
                                                                          p0,
                                                                          bkg_integral_distribution=integral_background)

        else:

            self._temporal_binner = TemporalBinner.bin_by_bayesian_blocks(events,
                                                                          p0)



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

    def _select_events(self, start, stop):
        """
        return an index of the selected events
        :param start:
        :param stop:
        :return:
        """

        return np.logical_and(start <= self._arrival_times, self._arrival_times <= stop)




class EventListWithDeadTime(EventList):
    def __init__(self, arrival_times, energies, n_channels, start_time=None, stop_time=None, dead_time=None,
                 first_channel=0, quality=None ,rsp_file=None, ra=None, dec=None, mission=None, instrument=None, verbose=True):
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
        :param  energies: list of event energies or pha channels
        :param  mission: mission name
        :param  instrument: instrument name
        :param  verbose: verbose level
        :param  ra:
        :param  dec:
        """

        EventList.__init__(self, arrival_times, energies, n_channels, start_time, stop_time, quality,first_channel, rsp_file,
                           ra, dec,
                           mission, instrument, verbose)

        if dead_time is not None:

            self._dead_time = np.asarray(dead_time)

            assert self._arrival_times.shape[0] == self._dead_time.shape[
                0], "Arrival time (%d) and Dead Time (%d) have different shapes" % (
                self._arrival_times.shape[0], self._dead_time.shape[0])

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
        '''Set the time interval(s) to be used during the analysis.

        Specified as 'tmin-tmax'. Intervals are in seconds. Example:

        set_active_time_intervals("0.0-10.0")

        which will set the energy range 0-10. seconds.
        '''

        self._time_selection_exists = True

        interval_masks = []

        time_intervals = TimeIntervalSet.from_strings(*args)

        time_intervals.merge_intersecting_intervals(in_place=True)

        for interval in time_intervals:
            tmin = interval.start_time
            tmax = interval.stop_time

            mask = self._select_events(tmin,tmax)

            interval_masks.append(mask)

        self._time_intervals = time_intervals

        time_mask = interval_masks[0]
        if len(interval_masks) > 1:
            for mask in interval_masks[1:]:
                time_mask = np.logical_or(time_mask, mask)

        tmp_counts = []  # Temporary list to hold the total counts per chan

        for chan in range(self._first_channel, self._n_channels + self._first_channel):
            channel_mask = self._energies == chan
            counts_mask = np.logical_and(channel_mask, time_mask)
            total_counts = len(self._arrival_times[counts_mask])

            tmp_counts.append(total_counts)

        self._counts = np.array(tmp_counts)

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

            total_dead_time = self._dead_time[time_mask].sum()
        else:

            total_dead_time = 0.

        self._exposure = exposure - total_dead_time


        self._active_dead_time = total_dead_time


class EventListWithLiveTime(EventList):
    def __init__(self, arrival_times, energies, n_channels, live_time, live_time_starts, live_time_stops,
                 start_time=None, stop_time=None, quality=None,
                 first_channel=0, rsp_file=None, ra=None, dec=None, mission=None, instrument=None, verbose=True):
        """
        An EventList where the exposure is calculated via and array of livetimes per interval.



        :param  arrival_times: list of event arrival times
        :param  energies: list of event energies or pha channels
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
        :param  rsp_file: the response file corresponding to these events
        :param verbose:
        :param  ra:
        :param  dec:
        """

        EventList.__init__(self, arrival_times, energies, n_channels, start_time, stop_time,quality ,first_channel, rsp_file,
                           ra, dec,
                           mission, instrument, verbose)

        assert len(live_time) == len(
            live_time_starts), "Live time fraction (%d) and live time start (%d) have different shapes" % (
            len(live_time), len(live_time_starts))

        assert len(live_time) == len(
            live_time_stops), "Live time fraction (%d) and live time stop (%d) have different shapes" % (
            len(live_time), len(live_time_stops))

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

        inside_idx = np.logical_and(self._live_time_starts < start, stop < self._live_time_stops)

        # see if it contains elements

        if self._live_time[inside_idx]:

            # we want to take a fraction of the live time covered

            dt = self._live_time_stops[inside_idx] - self._live_time_starts[inside_idx]

            fraction = (stop - start) / dt

            total_livetime = self._live_time[inside_idx] * fraction

        else:

            # First we get the live time of bins that are fully contained in the given interval
            # We now go for the closed interval because it is possible to have overlap with other intervals
            # when a closed interval exists... but not when there is only an open interval

            full_inclusion_idx = np.logical_and(start <= self._live_time_starts, stop >= self._live_time_stops)

            full_inclusion_livetime = self._live_time[full_inclusion_idx].sum()

            # Now we get the fractional parts on the left and right

            # Get the fractional part of the left bin

            left_remainder_idx = np.logical_and(start <= self._live_time_stops, self._live_time_starts <= start)

            dt = self._live_time_stops[left_remainder_idx] - self._live_time_starts[left_remainder_idx]

            # we want the distance to the stop of this bin

            distance_from_next_bin = self._live_time_stops[left_remainder_idx] - start

            fraction = distance_from_next_bin / dt

            left_fractional_livetime = self._live_time[left_remainder_idx] * fraction

            # Get the fractional part of the right bin

            right_remainder_idx = np.logical_and(self._live_time_starts <= stop, stop <= self._live_time_stops)

            dt = self._live_time_stops[right_remainder_idx] - self._live_time_starts[right_remainder_idx]

            # we want the distance from the last full bin

            distance_from_next_bin = stop - self._live_time_starts[right_remainder_idx]

            fraction = distance_from_next_bin / dt

            right_fractional_livetime = self._live_time[right_remainder_idx] * fraction

            # sum up all the live time

            total_livetime = full_inclusion_livetime + left_fractional_livetime + right_fractional_livetime

        # the sum at the end converts all the arrays to floats

        return total_livetime.sum()

    def set_active_time_intervals(self, *args):
        '''Set the time interval(s) to be used during the analysis.

        Specified as 'tmin-tmax'. Intervals are in seconds. Example:

        set_active_time_intervals("0.0-10.0")

        which will set the energy range 0-10. seconds.
        '''

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
            channel_mask = self._energies == chan
            counts_mask = np.logical_and(channel_mask, time_mask)
            total_counts = len(self._arrival_times[counts_mask])

            tmp_counts.append(total_counts)

        self._counts = np.array(tmp_counts)

        tmp_counts = []
        tmp_err = []  # Temporary list to hold the err counts per chan

        if self._poly_fit_exists:

            if not self._poly_fit_exists:
                raise RuntimeError('A polynomial fit to the channels does not exist!')

            # for chan in range(self._first_channel, self._n_channels + self._first_channel):
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

        # Live time correction

        exposure = 0.
        total_real_time = 0.
        for interval in self._time_intervals:
            total_real_time += interval.duration
            exposure += self.exposure_over_interval(interval.start_time, interval.stop_time)

        # In this case the exposure is the total live time

        self._exposure = exposure
        self._active_dead_time = total_real_time - exposure
