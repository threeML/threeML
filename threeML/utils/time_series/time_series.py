from __future__ import division, print_function

from builtins import object, range, zip

__author__ = "grburgess"

import collections
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from threeML.utils.progress_bar import tqdm, trange

from threeML.config.config import threeML_config
from threeML.exceptions.custom_exceptions import custom_warnings
from threeML.io.file_utils import sanitize_filename
from threeML.io.logging import setup_logger
from threeML.parallel.parallel_client import ParallelClient
from threeML.utils.spectrum.binned_spectrum import Quality
from threeML.utils.time_interval import TimeIntervalSet
from threeML.utils.time_series.polynomial import (Polynomial, polyfit,
                                                  unbinned_polyfit)

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


class TimeSeries(object):
    def __init__(
        self,
        start_time: float,
        stop_time: float,
        n_channels: int,
        native_quality=None,
        first_channel: int = 1,
        ra: float = None,
        dec: float = None,
        mission: str = None,
        instrument: str = None,
        verbose: bool = True,
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
        :param  energies: list of event energies or pha channels
        :param native_quality: native pha quality flags
        :param edges: The histogram boundaries if not specified by a response
        :param mission:
        :param instrument:
        :param verbose:
        :param  ra:
        :param  dec:
        """

        self._verbose: bool = verbose
        self._n_channels: int = n_channels
        self._first_channel: int = first_channel
        self._native_quality = native_quality

        # we haven't made selections yet

        self._time_intervals = None
        self._poly_intervals = None
        self._counts = None
        self._exposure = None
        self._poly_counts = None
        self._poly_count_err = None
        self._poly_selected_counts = None
        self._poly_exposure = None

        # ebounds for objects w/o a response
        self._edges = edges

        if native_quality is not None:
            assert len(native_quality) == n_channels, (
                "the native quality has length %d but you specified there were %d channels"
                % (len(native_quality), n_channels)
            )

        self._start_time = start_time

        self._stop_time = stop_time

        # name the instrument if there is not one

        if instrument is None:

            log.warning("No instrument name is given. Setting to UNKNOWN")

            self._instrument = "UNKNOWN"

        else:

            self._instrument = instrument

        if mission is None:

            log.warning("No mission name is given. Setting to UNKNOWN")

            self._mission = "UNKNOWN"

        else:

            self._mission = mission

        self._user_poly_order = -1
        self._time_selection_exists = False
        self._poly_fit_exists = False

        self._fit_method_info = {"bin type": None, "fit method": None}

    def set_active_time_intervals(self, *args):

        raise RuntimeError("Must be implemented in subclass")

    @property
    def poly_fit_exists(self) -> bool:

        return self._poly_fit_exists

    @property
    def n_channels(self) -> int:

        return self._n_channels

    @property
    def poly_intervals(self):
        return self._poly_intervals

    @property
    def polynomials(self):
        """ Returns polynomial is they exist"""
        if self._poly_fit_exists:
            return self._polynomials
        else:
            RuntimeError("A polynomial fit has not been made.")

    def get_poly_info(self) -> dict:
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

            # print('Coefficients')
            #
            # display(df_coeff)
            #
            # print('Coefficient Error')
            #
            # display(df_err)

            pan = {"coefficients": df_coeff, "error": df_err}

            return pan

        else:

            log.error("A polynomial fit has not been made.")
            RuntimeError()

    def get_total_poly_count(self, start: float, stop: float, mask=None) -> int:
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

    def get_total_poly_error(self, start: float, stop: float, mask=None) -> float:
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

            return self._temporal_binner
        else:

            raise RuntimeError("This EventList has no binning specified")

    def __set_poly_order(self, value: int):
        """ Set poly order only in allowed range and redo fit """

        assert type(value) is int, "Polynomial order must be integer"

        assert (
            -1 <= value <= 4
        ), "Polynomial order must be 0-4 or -1 to have it determined"

        self._user_poly_order = value

        log.debug(f"poly order set to {value}")

        if self._poly_fit_exists:

            log.info(
                f"Refitting background with new polynomial order ({value}) and existing selections"
            )

            if self._time_selection_exists:

                log.debug("recomputing time selection")

                self.set_polynomial_fit_interval(
                    *self._poly_intervals.to_string().split(","),
                    unbinned=self._unbinned,
                )

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

    poly_order = property(
        ___get_poly_order, ___set_poly_order, doc="Get or set the polynomial order"
    )

    @property
    def time_intervals(self):
        """
        the time intervals of the events

        :return:
        """
        return self._time_intervals

    def exposure_over_interval(self, tmin, tmax) -> float:
        """ calculate the exposure over a given interval  """

        raise RuntimeError("Must be implemented in sub class")

    def counts_over_interval(self, start, stop) -> int:
        """
        return the number of counts in the selected interval
        :param start: start of interval
        :param stop:  stop of interval
        :return:
        """

        # this will be a boolean list and the sum will be the
        # number of events

        raise RuntimeError("Must be implemented in sub class")

    def count_per_channel_over_interval(self, start, stop):
        """

        :param start:
        :param stop:
        :return:
        """

        raise RuntimeError("Must be implemented in sub class")

    def set_polynomial_fit_interval(self, *time_intervals, **kwargs) -> None:
        """Set the time interval to fit the background.
        Multiple intervals can be input as separate arguments
        Specified as 'tmin-tmax'. Intervals are in seconds. Example:

        set_polynomial_fit_interval("-10.0-0.0","10.-15.")

        :param time_intervals: intervals to fit on
        :param unbinned:
        :param bayes:
        :param kwargs:

        """

        # Find out if we want to binned or unbinned.
        # TODO: add the option to config file
        if "unbinned" in kwargs:
            unbinned = kwargs.pop("unbinned")
            assert type(
                unbinned) == bool, "unbinned option must be True or False"

        else:

            # assuming unbinned
            # could use config file here
            # unbinned = threeML_config['ogip']['use-unbinned-poly-fitting']

            unbinned = True


        # check if we are doing a bayesian
        # fit and record this info
            
        if "bayes" in kwargs:
            bayes = kwargs.pop("bayes")

        else:

            bayes = False

        if bayes:

            self._fit_method_info["fit method"] = "bayes"

        else:

            self._fit_method_info["fit method"] = "bayes"
            
        # we create some time intervals

        poly_intervals = TimeIntervalSet.from_strings(*time_intervals)

        # adjust the selections to the data

        new_intervals = []

        self._poly_selected_counts = []

        self._poly_exposure = 0.0

        for i, time_interval in enumerate(poly_intervals):

            t1 = time_interval.start_time
            t2 = time_interval.stop_time

            if (self._stop_time <= t1) or (t2 <= self._start_time):
                log.warning(
                    "The time interval %f-%f is out side of the arrival times and will be dropped"
                    % (t1, t2)
                )

            else:

                if t1 < self._start_time:
                    log.warning(
                        "The time interval %f-%f started before the first arrival time (%f), so we are changing the intervals to %f-%f"
                        % (t1, t2, self._start_time, self._start_time, t2)
                    )

                    t1 = self._start_time  # + 1

                if t2 > self._stop_time:
                    log.warning(
                        "The time interval %f-%f ended after the last arrival time (%f), so we are changing the intervals to %f-%f"
                        % (t1, t2, self._stop_time, t1, self._stop_time)
                    )

                    t2 = self._stop_time  # - 1.

                new_intervals.append("%f-%f" % (t1, t2))

                self._poly_selected_counts.append(
                    self.count_per_channel_over_interval(t1, t2)
                )
                self._poly_exposure += self.exposure_over_interval(t1, t2)

        # make new intervals after checks

        poly_intervals = TimeIntervalSet.from_strings(*new_intervals)

        self._poly_selected_counts = np.sum(self._poly_selected_counts, axis=0)

        # set the poly intervals as an attribute

        self._poly_intervals = poly_intervals

        # Fit the events with the given intervals
        if unbinned:

            self._unbinned = True  # keep track!

            self._unbinned_fit_polynomials(bayes=bayes)

        else:

            self._unbinned = False

            self._fit_polynomials(bayes=bayes)

        # we have a fit now

        self._poly_fit_exists = True

        log.info(
            f"{self._fit_method_info['bin type']} {self._optimal_polynomial_grade}-order polynomial fit with the {self._fit_method_info['fit method']} method"
        )

        # recalculate the selected counts

        if self._time_selection_exists:
            self.set_active_time_intervals(
                *self._time_intervals.to_string().split(","))

    def get_information_dict(
        self, use_poly: bool = False, extract: bool = False
    ) -> dict:
        """
        Return a PHAContainer that can be read by different builders

        :param use_poly: (bool) choose to build from the polynomial fits
        """
        if not self._time_selection_exists:
            log.error("No time selection exists! Cannot calculate rates")
            raise RuntimeError()

        if extract:

            log.debug("using extract method")

            is_poisson = True

            counts_err = None
            counts = self._poly_selected_counts
            rates = self._counts / self._poly_exposure
            rate_err = None
            exposure = self._poly_exposure

        elif use_poly:

            log.debug("using poly method")

            is_poisson = False

            counts_err = self._poly_count_err
            counts = self._poly_counts
            rate_err = self._poly_count_err / self._exposure
            rates = self._poly_counts / self._exposure
            exposure = self._exposure

            # removing negative counts

            idx = counts < 0.0

            counts[idx] = 0.0
            counts_err[idx] = 0.0

            rates[idx] = 0.0
            rate_err[idx] = 0.0

        else:

            is_poisson = True

            counts_err = None
            counts = self._counts
            rates = self._counts / self._exposure
            rate_err = None

            exposure = self._exposure

        if self._native_quality is None:

            quality = np.zeros_like(counts, dtype=int)

        else:

            quality = self._native_quality

        container_dict = {}

        container_dict["instrument"] = self._instrument
        container_dict["telescope"] = self._mission
        container_dict["tstart"] = self._time_intervals.absolute_start_time
        container_dict["telapse"] = (
            self._time_intervals.absolute_stop_time
            - self._time_intervals.absolute_start_time
        )
        container_dict["channel"] = np.arange(
            self._n_channels) + self._first_channel
        container_dict["counts"] = counts
        container_dict["counts error"] = counts_err
        container_dict["rates"] = rates
        container_dict["rate error"] = rate_err

        container_dict["edges"] = self._edges

        # check to see if we already have a quality object

        if isinstance(quality, Quality):

            container_dict["quality"] = quality

        else:

            container_dict["quality"] = Quality.from_ogip(quality)

        # TODO: make sure the grouping makes sense
        container_dict["backfile"] = "NONE"
        container_dict["grouping"] = np.ones(self._n_channels)
        container_dict["exposure"] = exposure
        # container_dict['response'] = self._response

        return container_dict

    def __repr__(self):
        """
        Examine the currently selected info as well other things.

        """

        return self._output().to_string()

    def _output(self):

        info_dict = collections.OrderedDict()
        for i, interval in enumerate(self.time_intervals):
            info_dict["active selection (%d)" % (i + 1)] = interval.__repr__()

        info_dict["active deadtime"] = self._active_dead_time

        if self._poly_fit_exists:

            for i, interval in enumerate(self.poly_intervals):
                info_dict["polynomial selection (%d)" % (
                    i + 1)] = interval.__repr__()

            info_dict["polynomial order"] = self._optimal_polynomial_grade

            info_dict["polynomial fit type"] = self._fit_method_info["bin type"]
            info_dict["polynomial fit method"] = self._fit_method_info["fit method"]

        return pd.Series(info_dict, index=list(info_dict.keys()))

    def _fit_global_and_determine_optimum_grade(self, cnts, bins, exposure, bayes=False):
        """
        Provides the ability to find the optimum polynomial grade for *binned* counts by fitting the
        total (all channels) to 0-4 order polynomials and then comparing them via a likelihood ratio test.


        :param cnts: counts per bin
        :param bins: the bins used
        :param exposure: exposure per bin
        :param bayes:
        :return: polynomial grade
        """

        min_grade = 0
        max_grade = 4
        log_likelihoods = []

        log.debug("attempting to find best poly with binned data")

        if threeML_config["parallel"]["use-parallel"]:

            def worker(grade):

                polynomial, log_like = polyfit(
                    bins, cnts, grade, exposure, bayes=bayes)

                return log_like

            client = ParallelClient()

            log_likelihoods = client.execute_with_progress_bar(
                worker, list(range(min_grade, max_grade + 1)), name="Finding best polynomial Order")

        else:

            for grade in trange(min_grade, max_grade + 1, desc="Finding best polynomial Order"):
                polynomial, log_like = polyfit(
                    bins, cnts, grade, exposure, bayes=bayes)

                log_likelihoods.append(log_like)

        # Found the best one
        delta_loglike = np.array(
            [2 * (x[0] - x[1])
             for x in zip(log_likelihoods[:-1], log_likelihoods[1:])]
        )

        log.debug(f"log likes {log_likelihoods}")
        log.debug(f" delta loglikes {delta_loglike}")

        delta_threshold = 9.0

        mask = delta_loglike >= delta_threshold

        if len(mask.nonzero()[0]) == 0:

            # best grade is zero!
            best_grade = 0

        else:

            best_grade = mask.nonzero()[0][-1] + 1

        return best_grade

    def _unbinned_fit_global_and_determine_optimum_grade(self, events, exposure, bayes=False):
        """
        Provides the ability to find the optimum polynomial grade for *unbinned* events by fitting the
        total (all channels) to 0-2 order polynomials and then comparing them via a likelihood ratio test.


        :param events: an event list
        :param exposure: the exposure per event
        :return: polynomial grade
        """

        # Fit the sum of all the channels to determine the optimal polynomial
        # grade

        min_grade = 0
        max_grade = 2
        log_likelihoods = []

        t_start = self._poly_intervals.start_times
        t_stop = self._poly_intervals.stop_times

        log.debug("attempting to find best fit poly with unbinned")

        if threeML_config["parallel"]["use-parallel"]:

            def worker(grade):

                polynomial, log_like = unbinned_polyfit(
                    events, grade, t_start, t_stop, exposure, bayes=bayes
                )

                return log_like

            client = ParallelClient()

            log_likelihoods = client.execute_with_progress_bar(
                worker, list(range(min_grade, max_grade + 1)), name="Finding best polynomial Order")

        else:

            for grade in trange(min_grade, max_grade + 1, desc="Finding best polynomial Order"):
                polynomial, log_like = unbinned_polyfit(
                    events, grade, t_start, t_stop, exposure, bayes=bayes
                )

                log_likelihoods.append(log_like)

        # Found the best one
        delta_loglike = np.array(
            [2 * (x[0] - x[1])
             for x in zip(log_likelihoods[:-1], log_likelihoods[1:])]
        )

        log.debug(f"log likes {log_likelihoods}")
        log.debug(f" delta loglikes {delta_loglike}")

        delta_threshold = 9.0

        mask = delta_loglike >= delta_threshold

        if len(mask.nonzero()[0]) == 0:

            # best grade is zero!
            best_grade = 0

        else:

            best_grade = mask.nonzero()[0][-1] + 1

        return best_grade

    def _fit_polynomials(self):

        raise NotImplementedError("this must be implemented in a subclass")

    def _unbinned_fit_polynomials(self):

        raise NotImplementedError("this must be implemented in a subclass")

    def save_background(self, filename, overwrite=False):
        """
        save the background to an HD5F

        :param filename:
        :return:
        """

        # make the file name proper

        filename = os.path.splitext(filename)

        filename = "%s.h5" % filename[0]

        filename_sanitized: Path = sanitize_filename(filename)

        # Check that it does not exists
        if filename_sanitized.exists():

            if overwrite:

                try:

                    filename_sanitized.unlink()

                except:

                    log.error(
                        f"The file {filename_sanitized} already exists and cannot be removed (maybe you do not have "
                        "permissions to do so?). "
                    )

                    raise IOError()

            else:

                log.error(f"The file {filename_sanitized} already exists!")
                raise IOError()

        with h5py.File(filename_sanitized, "w") as store:

            # extract the polynomial information and save it

            if self._poly_fit_exists:

                coeff = np.empty(
                    (self._n_channels, self._optimal_polynomial_grade + 1))
                err = np.empty(
                    (
                        self._n_channels,
                        self._optimal_polynomial_grade + 1,
                        self._optimal_polynomial_grade + 1,
                    )
                )

                for i, poly in enumerate(self._polynomials):

                    coeff[i, :] = poly.coefficients

                    err[i, ...] = poly.covariance_matrix

                # df_coeff = pd.Series(coeff)
                # df_err = pd.Series(err)

            else:

                log.error("the polynomials have not been fit yet")
                raise RuntimeError()

            store.create_dataset("coefficients", data=np.array(coeff))
            store.create_dataset("covariance", data=np.array(err))

            store.attrs["poly_order"] = self._optimal_polynomial_grade
            store.attrs["poly_selections"] = list(
                zip(
                    self._poly_intervals.start_times,
                    self._poly_intervals.stop_times,
                )
            )
            store.attrs["unbinned"] = self._unbinned
            store.attrs["fit_method"] = self._fit_method_info["fit method"]

        log.info(f"Saved fitted background to {filename_sanitized}")

    def restore_fit(self, filename):

        filename_sanitized: Path = sanitize_filename(filename)

        with h5py.File(filename_sanitized, "r") as store:

            coefficients = store["coefficients"][()]

            covariance = store["covariance"][()]

            self._polynomials = []

            # create new polynomials

            for i in range(len(coefficients)):
                coeff = np.array(coefficients[i])

                # make sure we get the right order
                # pandas stores the non-needed coeff
                # as nans.

                coeff = coeff[np.isfinite(coeff)]

                cov = covariance[i]

                self._polynomials.append(
                    Polynomial.from_previous_fit(coeff, cov))

            metadata = store.attrs

            self._optimal_polynomial_grade = metadata["poly_order"]
            poly_selections = np.array(metadata["poly_selections"])

            self._poly_intervals = TimeIntervalSet.from_starts_and_stops(
                poly_selections[:, 0], poly_selections[:, 1]
            )
            self._unbinned = metadata["unbinned"]

            if self._unbinned:
                self._fit_method_info["bin type"] = "unbinned"

            else:

                self._fit_method_info["bin type"] = "binned"

            self._fit_method_info["fit method"] = metadata["fit_method"]

        # go thru and count the counts!
        log.debug("resest the poly form the file")
        self._poly_fit_exists = True

        # we must go thru and collect the polynomial exposure and counts
        # so that they be extracted if needed
        self._poly_exposure = 0.0
        self._poly_selected_counts = []
        for i, time_interval in enumerate(self._poly_intervals):

            t1 = time_interval.start_time
            t2 = time_interval.stop_time

            self._poly_selected_counts.append(
                self.count_per_channel_over_interval(t1, t2)
            )
            self._poly_exposure += self.exposure_over_interval(t1, t2)

        self._poly_selected_counts = np.sum(self._poly_selected_counts, axis=0)
        if self._time_selection_exists:
            self.set_active_time_intervals(
                *self._time_intervals.to_string().split(","))

    def view_lightcurve(self, start=-10, stop=20.0, dt=1.0, use_binner=False):

        raise NotImplementedError("must be implemented in subclass")
