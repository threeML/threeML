from builtins import object
import numpy as np

from threeML.utils.spectrum.binned_spectrum import BinnedSpectrum
from threeML.utils.time_interval import TimeIntervalSet


class BinnedSpectrumSet(object):
    def __init__(self, binned_spectrum_list, reference_time=0.0, time_intervals=None):
        """
        a set of binned spectra with optional time intervals

        :param binned_spectrum_list: lit of binned spectal
        :param reference_time: reference time for time intervals
        :param time_intervals: optional timeinterval set
        """

        self._binned_spectrum_list = binned_spectrum_list  # type: list(BinnedSpectrum)
        self._reference_time = reference_time

        # normalize the time intervals if there are any

        if time_intervals is not None:

            self._time_intervals = (
                time_intervals - reference_time
            )  # type: TimeIntervalSet

            assert len(time_intervals) == len(
                binned_spectrum_list
            ), "time intervals mus be the same length as binned spectra"

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

        assert (
            self._time_intervals is not None
        ), "This spectrum set has no time intervals"

        return self._time_intervals.containing_bin(time)

    def sort(self):
        """
        sort the bin spectra in place according to time
        :return:
        """

        assert (
            self._time_intervals is not None
        ), "must have time intervals to do sorting"

        # get the sorting index

        idx = self._time_intervals.argsort()

        # reorder the spectra

        self._binned_spectrum_list = self._binned_spectrum_list[idx]

        # sort the time intervals in place

        self._time_intervals.sort()

    @property
    def quality_per_bin(self):

        return np.array([spectrum.quality for spectrum in self._binned_spectrum_list])

    @property
    def n_channels(self):

        return self.counts_per_bin.shape[1]

    @property
    def counts_per_bin(self):

        return np.array([spectrum.counts for spectrum in self._binned_spectrum_list])

    @property
    def count_errors_per_bin(self):

        return np.array(
            [spectrum.count_errors for spectrum in self._binned_spectrum_list]
        )

    @property
    def rates_per_bin(self):

        return np.array([spectrum.rates for spectrum in self._binned_spectrum_list])

    @property
    def rate_errors_per_bin(self):

        return np.array(
            [spectrum.rate_errors for spectrum in self._binned_spectrum_list]
        )

    @property
    def sys_errors_per_bin(self):

        return np.array(
            [spectrum.sys_errors for spectrum in self._binned_spectrum_list]
        )

    @property
    def exposure_per_bin(self):

        return np.array([spectrum.exposure for spectrum in self._binned_spectrum_list])

    @property
    def time_intervals(self):

        return self._time_intervals
