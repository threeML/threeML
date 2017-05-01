import numpy as np

from threeML.exceptions.custom_exceptions import custom_warnings
from threeML.io.progress_bar import progress_bar
from threeML.utils.time_interval import TimeIntervalSet
from threeML.utils.time_series.polynomial import fit_global_and_determine_optimum_grade, polyfit
from threeML.plugins.spectrum.binned_spectrum import BinnedSpectrum
#from threeML.plugins.spectrum.pha_spectrum import PHASpectrum




class BinnedSpectrumSet(object):

    def __init__(self, binned_spectrum_list, reference_time=0.0, time_intervals=None):
        """

        :param binned_spectrum_list:
        :param reference_time:
        """

        self._binned_spectrum_list = binned_spectrum_list # type: list(BinnedSpectrum)
        self._reference_time = reference_time

        # normalize the time intervals if there are any

        if time_intervals is not None:

            self._time_intervals = time_intervals  - reference_time#type: TimeIntervalSet

        else:

            self._time_intervals = None




    @property
    def reference_time(self):

        return self._reference_time

    def __getitem__(self, item):

        return self._binned_spectrum_list[item]

    def __len__(self):

        return len(self._binned_spectrum_list)


    def time_to_index(self, time):
        """
        get the index of the input time

        :param time: time to search for
        :return: integer
        """

        assert self._time_intervals is not None, 'This spectrum set has no time intervals'

        return self._time_intervals.containing_bin(time)

    @property
    def qaulity_per_bin(self):

        return np.array([spectrum.quality for spectrum in self._binned_spectrum_list])



    @property
    def n_channels(self):

        return self.counts_per_bin.shape[1]

    @property
    def counts_per_bin(self):

        return np.array([spectrum.counts for spectrum in self._binned_spectrum_list])

    @property
    def count_errors_per_bin(self):

        return np.array([spectrum.count_errors for spectrum in self._binned_spectrum_list])

    @property
    def rates_per_bin(self):

        return np.array([spectrum.rates for spectrum in self._binned_spectrum_list])

    @property
    def rate_errors_per_bin(self):

        return np.array([spectrum.rate_errors for spectrum in self._binned_spectrum_list])

    @property
    def sys_errors_per_bin(self):

        return np.array([spectrum.sys_errors for spectrum in self._binned_spectrum_list])

    @property
    def exposure_per_bin(self):

        return np.array([spectrum.exposure for spectrum in self._binned_spectrum_list])


    @property
    def time_intervals(self):

        return self._time_intervals

    def polynomial_fit(self, *fit_intervals):
        """
        fits a polynomial to all channels over the input time intervals

        :param fit_intervals: str input intervals
        :return:
        """



        assert self._time_intervals is not None, 'cannot do a temporal fit with no time intervals'

        tmp_poly_intervals = TimeIntervalSet.from_strings(*fit_intervals)

        starts = []
        stops = []


        for time_interval in tmp_poly_intervals:
            t1 = time_interval.start_time
            t2 = time_interval.stop_time

            if t1 < self._time_intervals.absolute_start:
                custom_warnings.warn(
                    "The time interval %f-%f started before the first arrival time (%f), so we are changing the intervals to %f-%f" % (
                        t1, t2, self._time_intervals.absolute_start, self._time_intervals.absolute_start, t2))

                t1 = self._time_intervals.absolute_start

            if t2 > self._time_intervals.absolute_stop:
                custom_warnings.warn(
                    "The time interval %f-%f ended after the last arrival time (%f), so we are changing the intervals to %f-%f" % (
                        t1, t2, self._time_intervals.absolute_stop, t1, self._time_intervals.absolute_stop))

                t2 = self._time_intervals.absolute_stop

            if (self._time_intervals.absolute_stop <= t1) or (t2 <= self._time_intervals.absolute_start):
                custom_warnings.warn(
                    "The time interval %f-%f is out side of the arrival times and will be dropped" % (
                        t1, t2))
                continue

            starts.append(t1)
            stops.append(t2)

        poly_intervals = TimeIntervalSet.from_starts_and_stops(starts,stops)


        selected_counts = []
        selected_exposure = []
        selected_midpoints = []

        for selection in poly_intervals:

            # get the mask of these events
            mask = self._time_intervals.containing_interval(selection.start_time,
                                                            selection.stop_time,
                                                            as_mask=True)


            selected_counts.append(self.counts_per_bin[mask])
            selected_exposure.append(self.exposure_per_bin[mask])
            selected_midpoints.append(self._time_intervals.half_times[mask])


        selected_counts = np.array(selected_counts)


        optimal_polynomial_grade =  fit_global_and_determine_optimum_grade(selected_counts.sum(axis=1),
                                                                           selected_midpoints,
                                                                           selected_exposure)
        # if self._verbose:
        #     print("Auto-determined polynomial order: %d" % optimal_polynomial_grade)
        #     print('\n')

        n_channels = selected_counts.shape[1]

        polynomials = []

        with progress_bar(n_channels, title="Fitting background" ) as p:
            for counts in selected_counts.T:


                polynomial, _ = polyfit(counts,
                                        selected_midpoints,
                                        optimal_polynomial_grade,
                                        selected_exposure)

                polynomials.append(polynomial)
                p.increase()


        estimated_counts = []
        estimated_count_errors = []


        # for internal in self._time_intervals:
        #
        #     for polynomial




