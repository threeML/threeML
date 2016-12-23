# Creates a generic event list reader that can create PHA objects on the fly

import numpy as np

import re

import copy
import pandas as pd

from threeML.io.rich_display import display
from threeML.utils.stats_tools import Significance

from threeML.parallel.parallel_client import ParallelClient
from threeML.config.config import threeML_config
from threeML.exceptions.custom_exceptions import custom_warnings
from threeML.io.progress_bar import progress_bar
from threeML.utils.binner import TemporalBinner

from event_polynomial import polyfit, unbinned_polyfit

from threeML.plugins.OGIP.pha import PHAContainer


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
    def __init__(self, arrival_times, energies, n_channels, start_time=None, stop_time=None,
                 first_channel=0, rsp_file=None, ra=None, dec=None, mission=None, instrument=None, verbose=True):
        """
        Container for event style data which are tagged with time and energy/PHA.




        :param  n_channels: Number of detector channels
        :param  start_time: start time of the event list
        :param  stop_time: stop time of the event list
        :param  first_channel: where detchans begin indexing
        :param  rsp_file: the response file corresponding to these events
        :param  arrival_times: list of event arrival times
        :param  energies: list of event energies or pha channels
        :param  ra:
        :param  dec:
        """

        self._verbose = verbose
        self._arrival_times = np.asarray(arrival_times)
        self._energies = np.asarray(energies)
        self._n_channels = n_channels
        self._first_channel = first_channel

        assert self._arrival_times.shape[0] == self._energies.shape[
            0], "Arrival time (%d) and energies (%d) have different shapes" % (
            self._arrival_times.shape[0], self._energies.shape[0])

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


    @staticmethod
    def _parse_time_interval(time_interval):
        # The following regular expression matches any two numbers, positive or negative,
        # like "-10 --5","-10 - -5", "-10-5", "5-10" and so on

        tokens = re.match('(\-?\+?[0-9]+\.?[0-9]*)\s*-\s*(\-?\+?[0-9]+\.?[0-9]*)', time_interval).groups()

        return map(float, tokens)

    def set_active_time_intervals(self, *args):

        raise RuntimeError("Must be implemented in subclass")

    @property
    def n_events(self):

        return self._arrival_times.shape[0]

    @property
    def energies(self):
        return self._energies

    @property
    def tmin_list(self):
        return self._tmin_list

    @property
    def tmax_list(self):
        return self._tmax_list

    @property
    def poly_intervals(self):
        return self._poly_time_selections

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

            print('Refitting background with new polynomial order and existing selections')
            if self._unbinned:

                self._unbinned_fit_polynomials()

            else:

                self._fit_polynomials()

    def ___set_poly_order(self, value):
        """ Indirect poly order setter """

        self.__set_poly_order(value)

    def __get_poly_order(self):
        """ get the poly order """

        return self._user_poly_order

    def ___get_poly_order(self):
        """ Indirect poly order getter """

        return self.__get_poly_order()

    poly_order = property(___get_poly_order, ___set_poly_order,
                          doc="Get or set the polynomial order")

    def _exposure_over_interval(self, tmin, tmax):
        """ calculate the exposure over a given interval  """

        raise RuntimeError("Must be implemented in sub class")


    def set_polynomial_fit_interval(self, *time_intervals, **options):
        """Set the time interval to fit the background.
        Multiple intervals can be input as separate arguments
        Specified as 'tmin-tmax'. Intervals are in seconds. Example:

        set_polynomial_fit_interval("-10.0-0.0","10.-15.")

        Args:
            *time_intervals:
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

        self._poly_time_selections = []

        for time_interval in time_intervals:
            t1, t2 = self._parse_time_interval(time_interval)

            self._poly_time_selections.append((t1, t2))

        self._poly_time_selections = np.array(self._poly_time_selections)

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

            tmp = []
            for tmin, tmax in zip(self._tmin_list, self._tmax_list):
                tmp.append("%.5f-%.5f" % (tmin, tmax))

            self.set_active_time_intervals(*tmp)

    def get_pha_container(self, use_poly=False):
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





        else:

            is_poisson = True

            rate_err = None
            rates = self._counts / (self._exposure)

        pha = PHAContainer(rates=rates,
                           rate_errors=rate_err,
                           n_channels=self._n_channels,
                           exposure=self._exposure,
                           is_poisson=is_poisson,
                           response_file=self._rsp_file,
                           mission=self._mission,
                           instrument=self._instrument,
                           quality=np.zeros_like(rates, dtype=int))  # default quality to all good

        return pha

    def peek(self):
        """
        Examine the currently selected info as well other things.

        """

        info_dict = {}

        info_dict['Active Selections'] = zip(self._tmin_list, self._tmax_list)
        info_dict['Active Deadtime'] = self._active_dead_time
        info_dict['Active Exposure'] = self._exposure
        info_dict['Total N. Events'] = len(self._arrival_times)
        info_dict['Active Counts'] = self._counts.sum()
        info_dict['Number of Channels'] = self._n_channels

        if self._poly_fit_exists:
            info_dict['Polynomial Selections'] = self._poly_time_selections
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

        t_start = self._poly_time_selections[:, 0]
        t_stop = self._poly_time_selections[:, 1]

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

        for selection in self._poly_time_selections:
            all_bkg_masks.append(np.logical_and(self._arrival_times >= selection[0],
                                                self._arrival_times <= selection[1]))
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

            exposure_per_bin.append(self._exposure_over_interval(bins[i], bins[i + 1]))

        mean_time = np.array(mean_time)

        exposure_per_bin = np.array(exposure_per_bin)

        # Remove bins with zero counts
        all_non_zero_mask = []

        for selection in self._poly_time_selections:
            all_non_zero_mask.append(np.logical_and(mean_time >= selection[0],
                                                    mean_time <= selection[1]))

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

        # Check whether we are parallelizing or not



        if not threeML_config['parallel']['use-parallel']:

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


        else:

            # With parallel computation

            # In order to distribute fairly the computation, the strategy is to parallelize the computation
            # by assigning to the engines one "line" of the grid at the time

            # Connect to the engines


            raise NotImplementedError('Coming Soon!')

            client = ParallelClient()

            # Get the number of engines

            n_engines = client.get_number_of_engines()

            if n_engines > self._n_channels:

                n_engines = int(self._n_channels)

                custom_warnings.warn(
                        "The number of engines is larger than the number of channels. Using only %s engines."
                        % n_engines, ReducingNumberOfThreads)

            chunk_size = ceildiv(self._n_channels, n_engines)

            # need to remove class ref
            grade = self._optimal_polynomial_grade

            def worker(start_index):

                polynomials = []
                channel_subset = channels[chunk_size * start_index: chunk_size * (start_index + 1)]

                for channel in channel_subset:

                    channel_mask = total_poly_energies == channel

                    # Select the masked events

                    current_events = total_poly_events[channel_mask]

                    # now bin the selected channel counts

                    cnts, _ = np.histogram(current_events, bins=these_bins)

                    # cnts = cnts / bin_width

                    # Put data to fit in an x vector and y vector

                    polynomial, _ = polyfit(mean_time[non_zero_mask],
                                            cnts[non_zero_mask],
                                            grade,
                                            exposure_per_bin[non_zero_mask])

                    polynomials.append(polynomial)

                return polynomials

            # Get a balanced view of the engines

            lview = client.load_balanced_view()
            # lview.block = True

            # Distribute the work among the engines and start it, but return immediately the control
            # to the main thread

            amr = lview.map_async(worker, range(n_engines))

            # client.wait_watching_progress(amr, 10)

            print("\n")

            res = amr.get()

            polynomials = []
            for i in range(n_engines):

                polynomials.extend(res[i])

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

        for selection in self._poly_time_selections:

            total_duration += selection[1] - selection[0]

            poly_exposure += self._exposure_over_interval(selection[0], selection[1])

            all_bkg_masks.append(np.logical_and(self._arrival_times >= selection[0],
                                                self._arrival_times <= selection[1]))
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

        # if self._dead_time is not None:
        #
        #     poly_deadtime = self._dead_time[poly_mask].sum()
        #
        # else:
        #
        #     poly_deadtime = 0
        #
        # poly_exposure = total_duration - poly_deadtime

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

        t_start = self._poly_time_selections[:, 0]
        t_stop = self._poly_time_selections[:, 1]

        if not threeML_config['parallel']['use-parallel']:

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


        else:

            raise NotImplementedError('Coming Soon!')

            # With parallel computation

            # In order to distribute fairly the computation, the strategy is to parallelize the computation
            # by assigning to the engines one "line" of the grid at the time

            # Connect to the engines

            client = ParallelClient()

            # Get the number of engines

            n_engines = client.get_number_of_engines()

            if n_engines > self._n_channels:

                n_engines = int(self._n_channels)

                custom_warnings.warn(
                        "The number of engines is larger than the number of channels. Using only %s engines."
                        % n_engines, ReducingNumberOfThreads)

            chunk_size = ceildiv(self._n_channels, n_engines)

            # need to remove class ref
            grade = self._optimal_polynomial_grade

            def worker(start_index):

                polynomials = []
                channel_subset = channels[chunk_size * start_index: chunk_size * (start_index + 1)]

                for channel in channel_subset:

                    channel_mask = total_poly_energies == channel

                    # Select the masked events

                    current_events = total_poly_events[channel_mask]

                    # now bin the selected channel counts

                    cnts, _ = np.histogram(current_events, bins=these_bins)

                    # cnts = cnts / bin_width

                    # Put data to fit in an x vector and y vector

                    polynomial, _ = polyfit(mean_time[non_zero_mask],
                                            cnts[non_zero_mask],
                                            grade,
                                            exposure_per_bin[non_zero_mask])

                    polynomials.append(polynomial)

                return polynomials

            # Get a balanced view of the engines

            lview = client.load_balanced_view()
            # lview.block = True

            # Distribute the work among the engines and start it, but return immediately the control
            # to the main thread

            amr = lview.map_async(worker, range(n_engines))

            # client.wait_watching_progress(amr, 10)

            print("\n")

            res = amr.get()

            polynomials = []
            for i in range(n_engines):

                polynomials.extend(res[i])

        # We are now ready to return the polynomials


        self._polynomials = polynomials


class EventListWithDeadTime(EventList):
    def __init__(self, arrival_times, energies, n_channels, start_time=None, stop_time=None, dead_time=None,
                 first_channel=0, rsp_file=None, ra=None, dec=None, mission=None, instrument=None, verbose=True):
        """
        Container for event style data which are tagged with time and energy/PHA.




        :param  n_channels: Number of detector channels
        :param  start_time: start time of the event list
        :param  stop_time: stop time of the event list
        :param  dead_time: an array of deadtime per event
        :param  first_channel: where detchans begin indexing
        :param  rsp_file: the response file corresponding to these events
        :param  arrival_times: list of event arrival times
        :param  energies: list of event energies or pha channels
        :param  ra:
        :param  dec:
        """

        EventList.__init__(self, arrival_times, energies, n_channels, start_time, stop_time, first_channel, rsp_file,
                           ra, dec,
                           mission, instrument, verbose)

        if dead_time is not None:

            self._dead_time = np.asarray(dead_time)

            assert self._arrival_times.shape[0] == self._dead_time.shape[
                0], "Arrival time (%d) and Dead Time (%d) have different shapes" % (
                self._arrival_times.shape[0], self._dead_time.shape[0])

        else:

            self._dead_time = None



    def _exposure_over_interval(self, tmin, tmax):
        """ calculate the exposure over a given interval  """

        mask = np.logical_and(self._arrival_times >= tmin, self._arrival_times <= tmax)

        if self._dead_time is not None:

            interval_deadtime = (self._dead_time[mask]).sum()

        else:

            interval_deadtime = 0

        return (tmax - tmin) - interval_deadtime

    def set_active_time_intervals(self, *args):
        '''Set the time interval(s) to be used during the analysis.

        Specified as 'tmin-tmax'. Intervals are in seconds. Example:

        set_active_time_intervals("0.0-10.0")

        which will set the energy range 0-10. seconds.
        '''

        self._time_selection_exists = True

        tmin_list = []
        tmax_list = []
        interval_masks = []

        for arg in args:
            tmin, tmax = self._parse_time_interval(arg)
            mask = np.logical_and(self._arrival_times >= tmin,
                                  self._arrival_times <= tmax)

            tmin_list.append(tmin)
            tmax_list.append(tmax)
            interval_masks.append(mask)

        if intervals_overlap(tmin_list, tmax_list):
            raise OverLappingIntervals('Provided intervals are overlapping and hence invalid')

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

                for tmin, tmax in zip(tmin_list, tmax_list):
                    # Now integrate the appropriate background polynomial
                    total_counts += self._polynomials[chan].integral(tmin, tmax)
                    counts_err += (self._polynomials[chan].integral_error(tmin, tmax)) ** 2

                tmp_counts.append(total_counts)

                tmp_err.append(np.sqrt(counts_err))

            self._poly_counts = np.array(tmp_counts)

            self._poly_count_err = np.array(tmp_err)

            # self._is_poisson = False

        # Dead time correction

        exposure = 0.
        for tmin, tmax in zip(tmin_list, tmax_list):
            exposure += tmax - tmin

        if self._dead_time is not None:

            total_dead_time = self._dead_time[time_mask].sum()
        else:

            total_dead_time = 0.

        self._exposure = exposure - total_dead_time
        # self._total_dead_time = total_dead_time

        self._tmin_list = tmin_list
        self._tmax_list = tmax_list

        self._active_dead_time = total_dead_time



class EventListWithLiveTime(EventList):
    def __init__(self, arrival_times, energies, n_channels, live_time, live_time_starts, live_time_stops,
                 start_time=None, stop_time=None,
                 first_channel=0, rsp_file=None, ra=None, dec=None, mission=None, instrument=None, verbose=True):
        """
        Container for event style data which are tagged with time and energy/PHA.



        :param  arrival_times: list of event arrival times
        :param  energies: list of event energies or pha channels
        :param live_time: array of livetime fractions
        :param live_time_starts: start of livetime fraction bins
        :param live_time_stops:  stop of livetime fraction bins
        :param mission:
        :param instrument:
        :param  n_channels: Number of detector channels
        :param  start_time: start time of the event list
        :param  stop_time: stop time of the event list
        :param  first_channel: where detchans begin indexing
        :param  rsp_file: the response file corresponding to these events


        :param  ra:
        :param  dec:
        """

        EventList.__init__(self, arrival_times, energies, n_channels, start_time, stop_time, first_channel, rsp_file,
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

    def _exposure_over_interval(self, tmin, tmax):
        """

        :param tmin: start time of interval
        :param tmax: stop time of interval
        :return: exposure
        """

        # First see if the interval is completely contained inside a
        # livetime interval. In this case we only compute that.

        # Note that this is explicitly on the open boundary of the
        # intervals because the closed boundary is covered in the
        # next step

        inside_idx = np.logical_and(self._live_time_starts < tmin, tmax < self._live_time_stops)

        # see if it contains elements

        if self._live_time[inside_idx]:


            # we want to take a fraction of the live time covered

            dt = self._live_time_stops[inside_idx] - self._live_time_starts[inside_idx]

            fraction = (tmax - tmin) / dt

            total_livetime = self._live_time[inside_idx] * fraction

        else:

            # First we get the live time of bins that are fully contained in the given interval
            # We now go for the closed interval because it is possible to have overlap with other intervals
            # when a closed interval exists... but not when there is only an open interval

            full_inclusion_idx = np.logical_and(tmin <= self._live_time_starts, tmax >= self._live_time_stops)

            full_inclusion_livetime = self._live_time[full_inclusion_idx].sum()

            # Now we get the fractional parts on the left and right

            # Get the fractional part of the left bin

            left_remainder_idx = np.logical_and(tmin <= self._live_time_stops, self._live_time_starts <= tmin)

            dt = self._live_time_stops[left_remainder_idx] - self._live_time_starts[left_remainder_idx]

            # we want the distance to the stop of this bin

            distance_from_next_bin = self._live_time_stops[left_remainder_idx] - tmin

            fraction = distance_from_next_bin / dt

            left_fractional_livetime = self._live_time[left_remainder_idx] * fraction

            # Get the fractional part of the right bin

            right_remainder_idx = np.logical_and(self._live_time_starts <= tmax, tmax <= self._live_time_stops)

            dt = self._live_time_stops[right_remainder_idx] - self._live_time_starts[right_remainder_idx]

            # we want the distance from the last full bin

            distance_from_next_bin = tmax - self._live_time_starts[right_remainder_idx]

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

        tmin_list = []
        tmax_list = []
        interval_masks = []

        for arg in args:
            tmin, tmax = self._parse_time_interval(arg)
            mask = np.logical_and(self._arrival_times >= tmin,
                                  self._arrival_times <= tmax)

            tmin_list.append(tmin)
            tmax_list.append(tmax)
            interval_masks.append(mask)

        if intervals_overlap(tmin_list, tmax_list):
            raise OverLappingIntervals('Provided intervals are overlapping and hence invalid')

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

            #for chan in range(self._first_channel, self._n_channels + self._first_channel):
            for chan in range(self._n_channels ):

                total_counts = 0
                counts_err = 0

                for tmin, tmax in zip(tmin_list, tmax_list):


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
        for tmin, tmax in zip(tmin_list, tmax_list):

            total_real_time += tmax - tmin
            exposure += self._exposure_over_interval(tmin, tmax)

        # In this case the exposure is the total live time

        self._exposure = exposure
        self._active_dead_time = total_real_time - exposure


        self._tmin_list = tmin_list
        self._tmax_list = tmax_list



def intervals_overlap(tmin, tmax):
    n_intervals = len(tmin)

    # Check that
    for i in range(n_intervals):
        throw_away_tmin = copy.copy(tmin)
        throw_away_tmax = copy.copy(tmax)

        this_min = throw_away_tmin.pop(i)
        this_max = throw_away_tmax.pop(i)

        for mn, mx in zip(throw_away_tmin, throw_away_tmax):

            if this_min < mn < this_max:

                return True

            elif this_min < mx < this_max:

                return True

        return False
