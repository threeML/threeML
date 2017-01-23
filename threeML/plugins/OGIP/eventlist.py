

import numpy as np
import os
import re

import copy
import pandas as pd
from pandas import HDFStore



from threeML.io.rich_display import display
from threeML.utils.stats_tools import Significance
from threeML.io.file_utils import sanitize_filename

from threeML.parallel.parallel_client import ParallelClient
from threeML.config.config import threeML_config
from threeML.exceptions.custom_exceptions import custom_warnings
from threeML.io.progress_bar import progress_bar
from threeML.utils.binner import TemporalBinner
from threeML.utils.time_interval import TimeInterval, TimeIntervalSet


from threeML.plugins.OGIP.event_polynomial import polyfit, unbinned_polyfit, Polynomial



class ReducingNumberOfThreads(Warning):
    pass


class ReducingNumberOfSteps(Warning):
    pass


class OverLappingIntervals(RuntimeError):
    pass


# find out how many splits we need to make
def ceildiv(a, b):
    return -(-a // b)


class EventList(object):
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

        self._verbose = verbose
        self._arrival_times = np.asarray(arrival_times)
        self._energies = np.asarray(energies)
        self._n_channels = n_channels
        self._first_channel = first_channel
        self._native_quality = native_quality

        self._time_intervals = None

        assert self._arrival_times.shape[0] == self._energies.shape[
            0], "Arrival time (%d) and energies (%d) have different shapes" % (
            self._arrival_times.shape[0], self._energies.shape[0])

        if native_quality is not None:

            assert len(native_quality) == n_channels, "the native quality has lenght %d but you specified there were %d channels"%(len(native_quality), n_channels)



        if start_time is None:

            self._start_time = arrival_times.min()

        else:

            self._start_time = start_time

        if stop_time is None:

            self._stop_time = arrival_times.max()

        else:

            self._stop_time = stop_time

        if instrument is None:

            custom_warnings.warn('No instrument name is given. Setting to UNKNOWN')

            self._instrument = "UNKNOWN"

        else:

            self._instrument = instrument

        if mission is None:

            custom_warnings.warn('No mission name is given. Setting to UNKNOWN')

            self._mission = "UNKNOWN"

        else:

            self._mission = mission

        self._rsp_file = rsp_file

        self._user_poly_order = -1
        self._time_selection_exists = False
        self._poly_fit_exists = False

        self._fit_method_info = {"bin type": None, 'fit method': None}


    def set_active_time_intervals(self, *args):

        raise RuntimeError("Must be implemented in subclass")

    @property
    def n_events(self):

        return self._arrival_times.shape[0]

    @property
    def energies(self):
        return self._energies

    @property
    def poly_intervals(self):
        return self._poly_intervals

    @property
    def polynomials(self):
        """ Returns polynomial is they exist"""
        if self._poly_fit_exists:
            return self._polynomials
        else:
            RuntimeError('A polynomial fit has not been made.')

    def get_poly_info(self):
        """
        Return a pandas panel frame with the polynomial coeffcients
        and errors
        Returns:
            a DataFrame

        """

        if self._poly_fit_exists:

            coeff = []
            err = []

            for poly in self._polynomials:
                coeff.append(poly.coefficients)
                err.append(poly.error)
            df_coeff = pd.DataFrame(coeff)
            df_err = pd.DataFrame(err)

            print('Coefficients')

            display(df_coeff)

            print('Coefficient Error')

            display(df_err)

            pan = pd.Panel({'coefficients': df_coeff, 'error': df_err})

            return pan






        else:
            RuntimeError('A polynomial fit has not been made.')

    def get_total_poly_count(self, start, stop, mask=None):
        """

        Get the total poly counts

        :param start:
        :param stop:
        :return:
        """
        if mask is None:
            mask = np.ones_like(self._polynomials, dtype=np.bool)

        total_counts = 0

        for p in np.asarray(self._polynomials)[mask]:
            total_counts += p.integral(start, stop)

        return total_counts

    def get_total_poly_error(self, start, stop, mask=None):
        """

        Get the total poly error

        :param start:
        :param stop:
        :return:
        """
        if mask is None:
            mask = np.ones_like(self._polynomials, dtype=np.bool)

        total_counts = 0

        for p in np.asarray(self._polynomials)[mask]:
            total_counts += p.integral_error(start, stop) ** 2

        return np.sqrt(total_counts)

    @property
    def bins(self):

        if self._temporal_binner is not None:

            return self._temporal_binner.bins
        else:

            raise RuntimeError('This EventList has no binning specified')

    @property
    def text_bins(self):

        if self._temporal_binner is not None:

            return self._temporal_binner.text_bins
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

        self._temporal_binner = TemporalBinner(events)

        tmp_bkg_getter = lambda a, b: self.get_total_poly_count(a, b, mask)
        tmp_err_getter = lambda a, b: self.get_total_poly_error(a, b, mask)

        self._temporal_binner.bin_by_significance(tmp_bkg_getter,
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

        self._temporal_binner = TemporalBinner(events)
        self._temporal_binner.bin_by_constanst(dt)

    def bin_by_custom(self, start, stop):
        """
        Interface to temporal binner's custom bin mode


        :param start: start times of the bins
        :param stop:  stop times of the bins
        :return:
        """

        self._temporal_binner = TemporalBinner(self._arrival_times)
        self._temporal_binner.bin_by_custom(start, stop)

    def bin_by_bayesian_blocks(self, start, stop, p0, use_background=False):

        events = self._arrival_times[np.logical_and(self._arrival_times >= start, self._arrival_times <= stop)]

        self._temporal_binner = TemporalBinner(events)

        if use_background:

            integral_background = lambda t: self.get_total_poly_count(start, t)

            self._temporal_binner.bin_by_bayesian_blocks(p0, bkg_integral_distribution=integral_background)

        else:

            self._temporal_binner.bin_by_bayesian_blocks(p0)

    def __set_poly_order(self, value):
        """ Set poly order only in allowed range and redo fit """

        assert type(value) is int, "Polynomial order must be integer"

        assert -1 <= value <= 4, "Polynomial order must be 0-4 or -1 to have it determined"

        self._user_poly_order = value

        if self._poly_fit_exists:

            print('Refitting background with new polynomial order (%d) and existing selections' % value)

            if self._time_selection_exists:

                self.set_polynomial_fit_interval(*self._poly_intervals.to_string().split(','), unbinned=self._unbinned)

            else:

                RuntimeError("This is a bug. Should never get here")

    def ___set_poly_order(self, value):
        """ Indirect poly order setter """

        self.__set_poly_order(value)

    def __get_poly_order(self):
        """ get the poly order """

        return self._optimal_polynomial_grade

    def ___get_poly_order(self):
        """ Indirect poly order getter """

        return self.__get_poly_order()

    poly_order = property(___get_poly_order, ___set_poly_order,
                          doc="Get or set the polynomial order")

    @property
    def time_intervals(self):
        """
        the time intervals of the events

        :return:
        """
        return self._time_intervals

    def exposure_over_interval(self, tmin, tmax):
        """ calculate the exposure over a given interval  """

        raise RuntimeError("Must be implemented in sub class")

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

    def set_polynomial_fit_interval(self, *time_intervals, **options):
        """Set the time interval to fit the background.
        Multiple intervals can be input as separate arguments
        Specified as 'tmin-tmax'. Intervals are in seconds. Example:

        set_polynomial_fit_interval("-10.0-0.0","10.-15.")

        :param time_intervals: intervals to fit on
        :param options:

        """

        # Find out if we want to binned or unbinned.
        # TODO: add the option to config file
        if 'unbinned' in options:
            unbinned = options.pop('unbinned')
            assert type(unbinned) == bool, 'unbinned option must be True or False'

        else:

            # assuming unbinned
            # could use config file here
            # unbinned = threeML_config['ogip']['use-unbinned-poly-fitting']

            unbinned = True



        poly_intervals = TimeIntervalSet.from_strings(*time_intervals)

        for time_interval in poly_intervals:
            t1 = time_interval.start_time
            t2 = time_interval.stop_time

            if t1 < self._start_time:

                custom_warnings.warn(
                    "The time interval %f-%f started before the first arrival time (%f), so we are changing the intervals to %f-%f" % (
                    t1, t2, self._start_time, self._start_time, t2))

                t1 = self._start_time


            if t2 > self._stop_time:


                custom_warnings.warn(
                    "The time interval %f-%f ended after the last arrival time (%f), so we are changing the intervals to %f-%f" % (
                        t1, t2, self._stop_time, t1, self._stop_time))

                t2 = self._stop_time

            if  (self._stop_time <= t1) or (t2 <= self._start_time):
                custom_warnings.warn(
                    "The time interval %f-%f is out side of the arrival times and will be dropped" % (
                        t1, t2))
                continue


        self._poly_intervals = poly_intervals

        # Fit the events with the given intervals
        if unbinned:

            self._unbinned = True  # keep track!

            self._unbinned_fit_polynomials()

        else:

            self._unbinned = False

            self._fit_polynomials()

        # Since changing the poly fit will alter the counts
        # We need to recalculate the source interval

        self._poly_fit_exists = True



        if self._verbose:
            print("%s %d-order polynomial fit with the %s method" % (
                self._fit_method_info['bin type'], self._optimal_polynomial_grade, self._fit_method_info['fit method']))
            print('\n')

        if self._time_selection_exists:

            self.set_active_time_intervals(*self._time_intervals.to_string().split(','))

    def get_pha_information(self, use_poly=False):
        """
        Return a PHAContainer that can be read by the PHA class
        Args:
            use_poly: (bool) choose to build from the polynomial fits
        Returns:
        """
        if not self._time_selection_exists:
            raise RuntimeError('No time selection exists! Cannot calculate rates')

        if use_poly:
            is_poisson = False

            rate_err = self._poly_count_err / self._exposure
            rates = self._poly_counts / self._exposure

            # removing negative counts

            idx = rates < 0.

            rates[idx] = 0.
            rate_err[idx] = 0.

        else:

            is_poisson = True

            rate_err = None
            rates = self._counts / (self._exposure)

        if self._native_quality is None:

            quality = np.zeros_like(rates, dtype=int)

        else:

            quality = self._native_quality

        container_dict = {}

        container_dict['instrument'] = self._instrument
        container_dict['telescope'] = self._mission
        container_dict['tstart'] = self._time_intervals.absolute_start_time
        container_dict['telapse'] = self._time_intervals.absolute_stop_time - self._time_intervals.absolute_start_time
        container_dict['channel'] = np.arange(self._n_channels) + self._first_channel
        container_dict['rate'] = rates
        container_dict['rate error'] = rate_err
        container_dict['quality'] = quality


        # TODO: make sure the grouping makes sense
        container_dict['backfile']='NONE'
        container_dict['grouping'] = np.ones(self._n_channels)
        container_dict['exposure'] = self._exposure
        container_dict['response_file'] = self._rsp_file

        return container_dict

    def peek(self):
        """
        Examine the currently selected info as well other things.

        """

        info_dict = {}

        info_dict['Active Selections'] = zip(self._time_intervals.start_times, self._time_intervals.stop_times)
        info_dict['Active Deadtime'] = self._active_dead_time
        info_dict['Active Exposure'] = self._exposure
        info_dict['Total N. Events'] = len(self._arrival_times)
        info_dict['Active Counts'] = self._counts.sum()
        info_dict['Number of Channels'] = self._n_channels

        if self._poly_fit_exists:
            info_dict['Polynomial Selections'] = zip(self._poly_intervals.start_times, self._poly_intervals.stop_times)
            info_dict['Polynomial Order'] = self._optimal_polynomial_grade
            info_dict['Active Count Error'] = np.sqrt((self._poly_count_err ** 2).sum())
            info_dict['Active Polynomial Counts'] = self._poly_counts.sum()
            info_dict['Poly fit type'] = self._fit_method_info['bin type']
            info_dict['Poly fit method'] = self._fit_method_info['fit method']

            sig = Significance(self._counts.sum(), self._poly_counts.sum())

            bkg_sig = np.sqrt((self._poly_count_err ** 2).sum())

            # too provocative?
            info_dict['Significance'] = sig.li_and_ma_equivalent_for_gaussian_background(bkg_sig)

        info_df = pd.Series(info_dict)

        display(info_df)

    def _fit_global_and_determine_optimum_grade(self, cnts, bins, exposure):
        """
        Provides the ability to find the optimum polynomial grade for *binned* counts by fitting the
        total (all channels) to 0-4 order polynomials and then comparing them via a likelihood ratio test.


        :param cnts: counts per bin
        :param bins: the bins used
        :param exposure: exposure per bin
        :return: polynomial grade
        """

        min_grade = 0
        max_grade = 4
        log_likelihoods = []

        for grade in range(min_grade, max_grade + 1):
            polynomial, log_like = polyfit(bins, cnts, grade, exposure)

            log_likelihoods.append(log_like)

        # Found the best one
        delta_loglike = np.array(map(lambda x: 2 * (x[0] - x[1]), zip(log_likelihoods[:-1], log_likelihoods[1:])))

        # print("\ndelta log-likelihoods:")

        # for i in range(max_grade):
        #    print("%s -> %s: delta Log-likelihood = %s" % (i, i + 1, deltaLoglike[i]))

        # print("")

        delta_threshold = 9.0

        mask = (delta_loglike >= delta_threshold)

        if (len(mask.nonzero()[0]) == 0):

            # best grade is zero!
            best_grade = 0

        else:

            best_grade = mask.nonzero()[0][-1] + 1

        return best_grade

    def _unbinned_fit_global_and_determine_optimum_grade(self, events, exposure):
        """
        Provides the ability to find the optimum polynomial grade for *unbinned* events by fitting the
        total (all channels) to 0-4 order polynomials and then comparing them via a likelihood ratio test.


        :param events: an event list
        :param exposure: the exposure per event
        :return: polynomial grade
        """

        # Fit the sum of all the channels to determine the optimal polynomial
        # grade


        min_grade = 0
        max_grade = 4
        log_likelihoods = []

        t_start = self._poly_intervals.start_times
        t_stop = self._poly_intervals.stop_times


        for grade in range(min_grade, max_grade + 1):
            polynomial, log_like = unbinned_polyfit(events, grade, t_start, t_stop, exposure)

            log_likelihoods.append(log_like)

        # Found the best one
        delta_loglike = np.array(map(lambda x: 2 * (x[0] - x[1]), zip(log_likelihoods[:-1], log_likelihoods[1:])))

        delta_threshold = 9.0

        mask = (delta_loglike >= delta_threshold)

        if (len(mask.nonzero()[0]) == 0):

            # best grade is zero!
            best_grade = 0

        else:

            best_grade = mask.nonzero()[0][-1] + 1

        return best_grade

    def _fit_polynomials(self):
        """

        Binned fit to each channel. Sets the polynomial array that will be used to compute
        counts over an interval



        :return:
        """

        self._poly_fit_exists = True

        self._fit_method_info['bin type'] = 'Binned'
        self._fit_method_info['fit method'] = threeML_config['event list']['binned fit method']

        # Select all the events that are in the background regions
        # and make a mask

        all_bkg_masks = []

        for selection in self._poly_intervals:
            all_bkg_masks.append(np.logical_and(self._arrival_times >= selection.start_time,
                                                self._arrival_times <= selection.stop_time))
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

        total_poly_energies = self._energies[poly_mask]

        # This calculation removes the unselected portion of the light curve
        # so that we are not fitting zero counts. It will be used in the channel calculations
        # as well

        bin_width = 1.  # seconds
        these_bins = np.arange(self._start_time,
                               self._stop_time,
                               bin_width)

        cnts, bins = np.histogram(total_poly_events,
                                  bins=these_bins)

        # Find the mean time of the bins and calculate the exposure in each bin
        mean_time = []
        exposure_per_bin = []
        for i in xrange(len(bins) - 1):
            m = np.mean((bins[i], bins[i + 1]))
            mean_time.append(m)

            exposure_per_bin.append(self.exposure_over_interval(bins[i], bins[i + 1]))

        mean_time = np.array(mean_time)

        exposure_per_bin = np.array(exposure_per_bin)

        # Remove bins with zero counts
        all_non_zero_mask = []

        for selection in self._poly_intervals:
            all_non_zero_mask.append(np.logical_and(mean_time >= selection.start_time,
                                                    mean_time <= selection.stop_time))

        non_zero_mask = all_non_zero_mask[0]
        if len(all_non_zero_mask) > 1:
            for mask in all_non_zero_mask[1:]:
                non_zero_mask = np.logical_or(mask, non_zero_mask)

        # Now we will find the the best poly order unless the use specified one
        # The total cnts (over channels) is binned to .1 sec intervals

        if self._user_poly_order == -1:

            self._optimal_polynomial_grade = self._fit_global_and_determine_optimum_grade(cnts[non_zero_mask],
                                                                                          mean_time[non_zero_mask],
                                                                                          exposure_per_bin[
                                                                                              non_zero_mask])
            if self._verbose:
                print("Auto-determined polynomial order: %d" % self._optimal_polynomial_grade)
                print('\n')


        else:

            self._optimal_polynomial_grade = self._user_poly_order

        channels = range(self._first_channel, self._n_channels + self._first_channel)

        polynomials = []

        with progress_bar(self._n_channels) as p:
            for channel in channels:
                channel_mask = total_poly_energies == channel

                # Mask background events and current channel
                # poly_chan_mask = np.logical_and(poly_mask, channel_mask)
                # Select the masked events

                current_events = total_poly_events[channel_mask]

                # now bin the selected channel counts

                cnts, bins = np.histogram(current_events,
                                          bins=these_bins)

                # Put data to fit in an x vector and y vector

                polynomial, _ = polyfit(mean_time[non_zero_mask],
                                        cnts[non_zero_mask],
                                        self._optimal_polynomial_grade,
                                        exposure_per_bin[non_zero_mask])

                polynomials.append(polynomial)
                p.increase()



        # We are now ready to return the polynomials


        self._polynomials = polynomials

    def _unbinned_fit_polynomials(self):

        self._poly_fit_exists = True

        # inform the type of fit we have
        self._fit_method_info['bin type'] = 'Unbinned'
        self._fit_method_info['fit method'] = threeML_config['event list']['unbinned fit method']

        # Select all the events that are in the background regions
        # and make a mask

        all_bkg_masks = []

        total_duration = 0.

        poly_exposure = 0

        for selection in self._poly_intervals:
            total_duration += selection.duration

            poly_exposure += self.exposure_over_interval(selection.start_time, selection.stop_time)

            all_bkg_masks.append(np.logical_and(self._arrival_times >= selection.start_time,
                                                self._arrival_times <= selection.stop_time))
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

        total_poly_energies = self._energies[poly_mask]

        # Now we will find the the best poly order unless the use specified one
        # The total cnts (over channels) is binned to .1 sec intervals

        if self._user_poly_order == -1:

            self._optimal_polynomial_grade = self._unbinned_fit_global_and_determine_optimum_grade(total_poly_events,
                                                                                                   poly_exposure)
            if self._verbose:
                print("Auto-determined polynomial order: %d" % self._optimal_polynomial_grade)
                print('\n')


        else:

            self._optimal_polynomial_grade = self._user_poly_order

        channels = range(self._first_channel, self._n_channels + self._first_channel)

        # Check whether we are parallelizing or not

        t_start = self._poly_intervals.start_times
        t_stop = self._poly_intervals.stop_times


        polynomials = []

        with progress_bar(self._n_channels) as p:
            for channel in channels:
                channel_mask = total_poly_energies == channel

                # Mask background events and current channel
                # poly_chan_mask = np.logical_and(poly_mask, channel_mask)
                # Select the masked events

                current_events = total_poly_events[channel_mask]

                polynomial, _ = unbinned_polyfit(current_events,
                                                 self._optimal_polynomial_grade,
                                                 t_start,
                                                 t_stop,
                                                 poly_exposure)

                polynomials.append(polynomial)
                p.increase()




        # We are now ready to return the polynomials


        self._polynomials = polynomials


    def save_background(self, filename, overwrite=False):
        """
        save the background to an HD5F

        :param filename:
        :return:
        """

        # make the file name proper

        filename = os.path.splitext(filename)



        filename = "%s.h5" % filename[0]


        filename_sanitized = sanitize_filename(filename)

        # Check that it does not exists
        if os.path.exists(filename_sanitized):

            if overwrite:

                try:

                    os.remove(filename_sanitized)

                except:

                    raise IOError("The file %s already exists and cannot be removed (maybe you do not have "
                                  "permissions to do so?). " % filename_sanitized)

            else:

                raise IOError("The file %s already exists!" % filename_sanitized)

        with HDFStore(filename_sanitized) as store:

            # extract the polynomial information and save it

            if self._poly_fit_exists:

                coeff = []
                err = []

                for poly in self._polynomials:
                    coeff.append(poly.coefficients)
                    err.append(poly.covariance_matrix)
                df_coeff = pd.Series(coeff)
                df_err = pd.Series(err)

            else:

                raise RuntimeError('the polynomials have not been fit yet')

            df_coeff.to_hdf(store, 'coefficients')
            df_err.to_hdf(store, 'covariance')



            store.get_storer('coefficients').attrs.metadata = {'poly_order': self._optimal_polynomial_grade,
                                                             'poly_selections': zip(self._poly_intervals.start_times,self._poly_intervals.stop_times),
                                                             'unbinned':self._unbinned}

        if self._verbose:

            print("\nSaved fitted background to %s.\n"% filename)



    def restore_fit(self, filename):


        filename_sanitized = sanitize_filename(filename)

        with HDFStore(filename_sanitized) as store:

            coefficients = store['coefficients']



            covariance = store['covariance']

            self._polynomials = []

            # create new polynomials

            for i in range(len(coefficients)):

                coeff = np.array(coefficients.loc[i])

                # make sure we get the right order
                # pandas stores the non-needed coeff
                # as nans.

                coeff = coeff[np.isfinite(coeff)]

                cov  = covariance.loc[i]



                self._polynomials.append(Polynomial.from_previous_fit(coeff, cov))





            metadata = store.get_storer('coefficients').attrs.metadata

            self._optimal_polynomial_grade = metadata['poly_order']
            poly_selections = np.array(metadata['poly_selections'])

            self._poly_intervals = TimeIntervalSet.from_starts_and_stops(poly_selections[:,0],poly_selections[:,1])
            self._unbinned = metadata['unbinned']

        # go thru and count the counts!

        self._poly_fit_exists = True

        if self._time_selection_exists:

            self.set_active_time_intervals(*self._time_intervals.to_string().split(','))


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

                for tmin, tmax in zip(self._time_intervals.start_times, self._time_intervals.start_times):
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
