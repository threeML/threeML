import numpy as np

from threeML.plugins.spectrum.binned_spectrum import BinnedSpectrum
from threeML.utils.time_interval import TimeIntervalSet
from threeML.plugins.OGIP.event_polynomial import fit_global_and_determine_optimum_grade, polyfit
from threeML.exceptions.custom_exceptions import custom_warnings
from threeML.io.progress_bar import progress_bar
from threeML.plugins.spectrum.pha_spectrum import PHASpectrum
import astropy.io.fits as fits




class BinnedSpectrumSet(object):

    def __init__(self, binned_spectrum_list, reference_time=0.0, time_intervals=None):
        """

        :param binned_spectrum_list:
        :param reference_time:
        """

        self._binned_spectrum_list = binned_spectrum_list
        self._reference_time = reference_time

        self._time_intervals = time_intervals  - reference_time#type: TimeIntervalSet


    @property
    def reference_time(self):

        return self._reference_time

    def __getitem__(self, item):

        return self._binned_spectrum_list[item]

    def __len__(self):

        return len(self._binned_spectrum_list)


    @property
    def counts_per_bin(self):

        return np.array([spectrum.counts for spectrum in self._binned_spectrum_list])

    @property
    def exposure(self):

        return np.array([spectrum.exposure for spectrum in self._binned_spectrum_list])



    @property
    def time_intervals(self):

        return self._time_intervals

    def polynomial_fit(self, *fit_intervals):


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
            selected_exposure.append(self.exposure[mask])
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



    @classmethod
    def from_pha2_fits(cls, pha2_file,*rsp_files):

        with fits.open(pha2_file) as f:

            try:

                HDUidx = f.index_of("SPECTRUM")

            except:

                raise RuntimeError("The input file %s is not in PHA format" % (pha2_file))

            spectrum = f[HDUidx]
            data = spectrum.data

            if "COUNTS" in data.columns.names:

                has_rates = False
                data_column_name = "COUNTS"

            elif "RATE" in data.columns.names:

                has_rates = True
                data_column_name = "RATE"

            else:

                raise RuntimeError("This file does not contain a RATE nor a COUNTS column. "
                                   "This is not a valid PHA file")

                # Determine if this is a PHA I or PHA II
            if len(data.field(data_column_name).shape) == 2:

                num_spectra = data.field(data_column_name).shape[0]

            else:

                raise RuntimeError("This appears to be a PHA I and not PHA II file")


            assert len(rsp_files) == 1 or len(rsp_files) == num_spectra, 'The number of RSPs input does not math the number of spectra in the PHAII file'

            # if one rsp file is used for all spectra, then we create
            # a proper length array to account for that

            if len(rsp_files) < num_spectra:

                rsp_files = [rsp_files[0]] * num_spectra



            list_of_binned_spectra =[ PHASpectrum(pha2_file,
                                                  spectrum_number=spectrum_number,
                                                  file_type='observed',
                                                  rsp_file=rsp_files[spectrum_number-1]) for spectrum_number in range(1, num_spectra+1)]



            # now get the time intervals

            start_times = data.field('TIME')
            stop_times = data.field('ENDTIME')

            time_intervals = TimeIntervalSet.from_starts_and_stops(start_times, stop_times)

            reference_time = 0

            # see if there is a reference time in the file

            if 'TRIGTIME' in spectrum.header:

                reference_time = spectrum.header['TRIGTIME']


            for t_number in range(spectrum.header['TFIELDS']):

                if 'TZERO%d' %t_number in spectrum.header:

                    reference_time = spectrum.header['TZERO%d' %t_number]


            return cls( list_of_binned_spectra, reference_time=reference_time, time_intervals=time_intervals)









