import copy

import astropy.io.fits as fits
import numpy as np
import re


from threeML.exceptions.custom_exceptions import custom_warnings
from threeML.io.file_utils import file_existing_and_readable
from threeML.io.progress_bar import progress_bar
from threeML.plugins.DispersionSpectrumLike import DispersionSpectrumLike
from threeML.plugins.OGIPLike import OGIPLike
from threeML.plugins.SpectrumLike import SpectrumLike, NegativeBackground
from threeML.utils.OGIP.pha import PHAWrite
from threeML.utils.OGIP.response import (
    InstrumentResponse,
    InstrumentResponseSet,
    OGIPResponse,
)

from threeML.utils.spectrum.binned_spectrum import (
    BinnedSpectrum,
    BinnedSpectrumWithDispersion,
)
from threeML.utils.polarization.binned_polarization import BinnedModulationCurve
from threeML.utils.statistics.stats_tools import Significance
from threeML.utils.time_interval import TimeIntervalSet
from threeML.utils.time_series.binned_spectrum_series import BinnedSpectrumSeries
from threeML.utils.time_series.event_list import (
    EventListWithDeadTime,
    EventListWithLiveTime,
    EventList,
    EventListWithDeadTimeFraction,
)
from threeML.utils.time_series.time_series import TimeSeries
from threeML.utils.histogram import Histogram

from threeML.utils.data_builders.fermi.gbm_data import GBMTTEFile, GBMCdata
from threeML.utils.data_builders.fermi.lat_data import LLEFile

try:

    from polarpy.polar_data import POLARData
    from polarpy.polarlike import PolarLike
    from polarpy.polar_response import PolarResponse

    has_polarpy = True

except (ImportError):

    has_polarpy = False


class BinningMethodError(RuntimeError):
    pass


class TimeSeriesBuilder(object):
    def __init__(
        self,
        name,
        time_series,
        response=None,
        poly_order=-1,
        unbinned=True,
        verbose=True,
        restore_poly_fit=None,
        container_type=BinnedSpectrumWithDispersion,
    ):
        """
        Class for handling generic time series data including binned and event list
        series. Depending on the data, this class builds either a  SpectrumLike or
        DisperisonSpectrumLike plugin

        For specific instruments, use the TimeSeries.from() classmethods


        :param name: name for the plugin
        :param time_series: a TimeSeries instance
        :param response: options InstrumentResponse instance
        :param poly_order: the polynomial order to use for background fitting
        :param unbinned: if the background should be fit unbinned
        :param verbose: the verbosity switch
        :param restore_poly_fit: file from which to read a prefitted background
        """

        assert isinstance(time_series, TimeSeries), "must be a TimeSeries instance"

        assert issubclass(container_type, Histogram), "must be a subclass of Histogram"

        self._name = name

        self._container_type = container_type

        self._time_series = time_series  # type: TimeSeries

        # make sure we have a proper response

        if response is not None:
            assert (
                isinstance(response, InstrumentResponse)
                or isinstance(response, InstrumentResponseSet)
                or isinstance(response, str)
            ), "Response must be an instance of InstrumentResponse"

        # deal with RSP weighting if need be

        if isinstance(response, InstrumentResponseSet):

            # we have a weighted response
            self._rsp_is_weighted = True
            self._weighted_rsp = response

            # just get a dummy response for the moment
            # it will be corrected when we set the interval

            self._response = InstrumentResponse.create_dummy_response(
                response.ebounds, response.monte_carlo_energies
            )

        else:

            self._rsp_is_weighted = False
            self._weighted_rsp = None

            self._response = response

        self._verbose = verbose
        self._active_interval = None
        self._observed_spectrum = None
        self._background_spectrum = None
        self._measured_background_spectrum = None

        self._time_series.poly_order = poly_order

        self._default_unbinned = unbinned

        # try and restore the poly fit if requested

        if restore_poly_fit is not None:

            if file_existing_and_readable(restore_poly_fit):
                self._time_series.restore_fit(restore_poly_fit)

                if verbose:
                    print("Successfully restored fit from %s" % restore_poly_fit)

            else:

                custom_warnings.warn(
                    "Could not find saved background %s." % restore_poly_fit
                )

    def _output(self):

        pass
        # super_out = super(EventListLike, self)._output()
        # return super_out.append(self._time_series._output())

    def __set_poly_order(self, value):
        """Background poly order setter """

        self._time_series.poly_order = value

    def ___set_poly_order(self, value):
        """ Indirect poly order setter """

        self.__set_poly_order(value)

    def __get_poly_order(self):
        """ Get poly order """
        return self._time_series.poly_order

    def ___get_poly_order(self):
        """ Indirect poly order getter """

        return self.__get_poly_order()

    background_poly_order = property(
        ___get_poly_order,
        ___set_poly_order,
        doc="Get or set the background polynomial order",
    )

    def set_active_time_interval(self, *intervals, **kwargs):
        """
        Set the time interval to be used during the analysis.
        For now, only one interval can be selected. This may be
        updated in the future to allow for self consistent time
        resolved analysis.
        Specified as 'tmin-tmax'. Intervals are in seconds. Example:

        set_active_time_interval("0.0-10.0")

        which will set the energy range 0-10. seconds.
        :param options:
        :param intervals:
        :return:
        """

        self._time_series.set_active_time_intervals(*intervals)

        # extract a spectrum

        if self._response is None:

            self._observed_spectrum = self._container_type.from_time_series(
                self._time_series, use_poly=False
            )

        else:

            if self._rsp_is_weighted:
                self._response = self._weighted_rsp.weight_by_counts(
                    *self._time_series.time_intervals.to_string().split(",")
                )

            self._observed_spectrum = self._container_type.from_time_series(
                self._time_series, self._response, use_poly=False
            )

        self._active_interval = intervals

        # re-get the background if there was a time selection

        if self._time_series.poly_fit_exists:

            self._background_spectrum = self._container_type.from_time_series(
                self._time_series, response=self._response, use_poly=True, extract=False
            )

            self._measured_background_spectrum = self._container_type.from_time_series(
                self._time_series,
                response=self._response,
                use_poly=False,
                extract=True,
            )

        self._tstart = self._time_series.time_intervals.absolute_start_time
        self._tstop = self._time_series.time_intervals.absolute_stop_time

    def set_background_interval(self, *intervals, **options):
        """
        Set the time interval to fit the background.
        Multiple intervals can be input as separate arguments
        Specified as 'tmin-tmax'. Intervals are in seconds. Example:

        setBackgroundInterval("-10.0-0.0","10.-15.")


        :param *intervals:
        :param **options:

        :return: none

        """
        if "unbinned" in options:

            unbinned = options.pop("unbinned")
        else:

            unbinned = self._default_unbinned

        self._time_series.set_polynomial_fit_interval(*intervals, unbinned=unbinned)

        # In theory this will automatically get the poly counts if a
        # time interval already exists

        if self._active_interval is not None:

            if self._response is None:

                self._background_spectrum = self._container_type.from_time_series(
                    self._time_series, use_poly=True, extract=False
                )

                self._measured_background_spectrum = self._container_type.from_time_series(
                    self._time_series, use_poly=False, extract=True
                )

            else:

                # we do not need to worry about the interval of the response if it is a set. only the ebounds are extracted here

                self._background_spectrum = self._container_type.from_time_series(
                    self._time_series, self._response, use_poly=True, extract=False
                )

                self._measured_background_spectrum = self._container_type.from_time_series(
                    self._time_series, self._response, use_poly=False, extract=True,
                )

    def write_pha_from_binner(
        self,
        file_name,
        start=None,
        stop=None,
        overwrite=False,
        force_rsp_write=False,
        extract_measured_background=False,
    ):
        """
        Write PHA fits files from the selected bins. If writing from an event list, the
        bins are from create_time_bins. If using a pre-time binned time series, the bins are those
        native to the data. Start and stop times can be used to  control which bins are written to files

        :param file_name: the file name of the output files
        :param start: optional start time of the bins
        :param stop: optional stop time of the bins
        :param overwrite: if the fits files should be overwritten
        :param force_rsp_write: force the writing of RSPs
        :param extract_measured_background: Use the selected background rather than a polynomial fit to the background
        :return: None
        """

        # we simply create a bunch of dispersion plugins and convert them to OGIP

        ogip_list = [
            OGIPLike.from_general_dispersion_spectrum(sl)
            for sl in self.to_spectrumlike(
                from_bins=True,
                start=start,
                stop=stop,
                extract_measured_background=extract_measured_background,
            )
        ]

        # write out the PHAII file

        pha_writer = PHAWrite(*ogip_list)

        pha_writer.write(
            file_name, overwrite=overwrite, force_rsp_write=force_rsp_write
        )

    def get_background_parameters(self):
        """
        Returns a pandas DataFrame containing the background polynomial
        coefficients for each channel.

        """

        return self._time_series.get_poly_info()

    def save_background(self, filename, overwrite=False):
        """

        save the background to and HDF5 file. The filename does not need an extension.
        The filename will be saved as <filename>_bkg.h5



        :param filename: name of file to save
        :param overwrite: to overwrite or not
        :return:
        """

        self._time_series.save_background(filename, overwrite)

    def view_lightcurve(self, start=-10, stop=20.0, dt=1.0, use_binner=False):
        # type: (float, float, float, bool) -> None

        """
        :param start:
        :param stop:
        :param dt:
        :param use_binner:

        """

        return self._time_series.view_lightcurve(start, stop, dt, use_binner)

    @property
    def tstart(self):
        """
        :return: start time of the active interval
        """

        return self._tstart

    @property
    def tstop(self):
        """
        :return: stop time of the active interval
        """

        return self._tstop

    @property
    def bins(self):

        return self._time_series.bins

    @property
    def time_series(self):
        """                                                                                                                                                                                                                                                                   
        returns the time_series                                                                                                                                                                                                                                               
        :return: time_series                                                                                                                                                                                                                                                  
        """

        return self._time_series

    @property
    def significance_per_interval(self):

        if self._time_series.bins is not None:

            sig_per_interval = []

            # go thru each interval and extract the significance

            for (start, stop) in self._time_series.bins.bin_stack:

                total_counts = self._time_series.counts_over_interval(start, stop)
                bkg_counts = self._time_series.get_total_poly_count(start, stop)
                bkg_error = self._time_series.get_total_poly_error(start, stop)

                sig_calc = Significance(total_counts, bkg_counts)

                sig_per_interval.append(
                    sig_calc.li_and_ma_equivalent_for_gaussian_background(bkg_error)[0]
                )

            return np.array(sig_per_interval)

    @property
    def total_counts_per_interval(self):

        if self._time_series.bins is not None:

            total_counts = []

            for (start, stop) in self._time_series.bins.bin_stack:

                total_counts.append(self._time_series.counts_over_interval(start, stop))

            return np.array(total_counts)

    @property
    def background_counts_per_interval(self):

        if self._time_series.bins is not None:

            total_counts = []

            for (start, stop) in self._time_series.bins.bin_stack:
                total_counts.append(self._time_series.get_total_poly_count(start, stop))

            return np.array(total_counts)

    def read_bins(self, time_series_builder):
        """

        Read the temporal bins from another *binned* TimeSeriesBuilder instance
        and apply those bins to this instance

        :param time_series_builder: *binned* time series builder to copy
        :return:
        """

        other_bins = time_series_builder.bins.bin_stack
        self.create_time_bins(other_bins[:, 0], other_bins[:, 1], method="custom")

    def create_time_bins(self, start, stop, method="constant", **options):
        """

        Create time bins from start to stop with a given method (constant, siginificance, bayesblocks, custom).
        Each method has required keywords specified in the parameters. Once created, this can be used as
        a JointlikelihoodSet generator, or as input for viewing the light curve.

        :param start: start of the bins or array of start times for custom mode
        :param stop: stop of the bins or array of stop times for custom mode
        :param method: constant, significance, bayesblocks, custom

        :param dt: <constant method> delta time of the
        :param sigma: <significance> sigma level of bins
        :param min_counts: (optional) <significance> minimum number of counts per bin
        :param p0: <bayesblocks> the chance probability of having the correct bin configuration.
        :return:
        """

        assert isinstance(
            self._time_series, EventList
        ), "can only bin event lists currently"

        # if 'use_energy_mask' in options:
        #
        #     use_energy_mask = options.pop('use_energy_mask')
        #
        # else:
        #
        #     use_energy_mask = False

        if method == "constant":

            if "dt" in options:
                dt = float(options.pop("dt"))

            else:

                raise RuntimeError("constant bins requires the dt option set!")

            self._time_series.bin_by_constant(start, stop, dt)

        elif method == "significance":

            if "sigma" in options:

                sigma = options.pop("sigma")

            else:

                raise RuntimeError("significance bins require a sigma argument")

            if "min_counts" in options:

                min_counts = options.pop("min_counts")

            else:

                min_counts = 10

            # removed for now
            # should we mask the data

            # if use_energy_mask:
            #
            #     mask = self._mask
            #
            # else:
            #
            #     mask = None

            self._time_series.bin_by_significance(
                start, stop, sigma=sigma, min_counts=min_counts, mask=None
            )

        elif method == "bayesblocks":

            if "p0" in options:

                p0 = options.pop("p0")

            else:

                p0 = 0.1

            if "use_background" in options:

                use_background = options.pop("use_background")

            else:

                use_background = False

            self._time_series.bin_by_bayesian_blocks(start, stop, p0, use_background)

        elif method == "custom":

            if type(start) is not list:

                if type(start) is not np.ndarray:
                    raise RuntimeError("start must be and array in custom mode")

            if type(stop) is not list:

                if type(stop) is not np.ndarray:
                    raise RuntimeError("stop must be and array in custom mode")

            assert len(start) == len(
                stop
            ), "must have equal number of start and stop times"

            self._time_series.bin_by_custom(start, stop)

        else:

            raise BinningMethodError(
                "Only constant, significance, bayesblock, or custom method argument accepted."
            )

        if self._verbose:

            print("Created %d bins via %s" % (len(self._time_series.bins), method))

    def to_spectrumlike(
        self,
        from_bins=False,
        start=None,
        stop=None,
        interval_name="_interval",
        extract_measured_background=False,
    ):
        """
        Create plugin(s) from either the current active selection or the time bins.
        If creating from an event list, the
        bins are from create_time_bins. If using a pre-time binned time series, the bins are those
        native to the data. Start and stop times can be used to  control which bins are used.

        :param from_bins: choose to create plugins from the time bins
        :param start: optional start time of the bins
        :param stop: optional stop time of the bins
        :param extract_measured_background: Use the selected background rather than a polynomial fit to the background
        :param interval_name: the name of the interval
        :return: SpectrumLike plugin(s)
        """

        # we can use either the modeled or the measured background. In theory, all the information
        # in the background spectrum should propagate to the likelihood

        if extract_measured_background:

            this_background_spectrum = self._measured_background_spectrum

        else:

            this_background_spectrum = self._background_spectrum

        # this is for a single interval

        if not from_bins:

            assert (
                self._observed_spectrum is not None
            ), "Must have selected an active time interval"

            assert isinstance(
                self._observed_spectrum, BinnedSpectrum
            ), "You are attempting to create a SpectrumLike plugin from the wrong data type"

            if this_background_spectrum is None:

                custom_warnings.warn(
                    "No background selection has been made. This plugin will contain no background!"
                )

            if self._response is None:

                return SpectrumLike(
                    name=self._name,
                    observation=self._observed_spectrum,
                    background=this_background_spectrum,
                    verbose=self._verbose,
                    tstart=self._tstart,
                    tstop=self._tstop,
                )

            else:

                return DispersionSpectrumLike(
                    name=self._name,
                    observation=self._observed_spectrum,
                    background=this_background_spectrum,
                    verbose=self._verbose,
                    tstart=self._tstart,
                    tstop=self._tstop,
                )

        else:

            # this is for a set of intervals.

            assert (
                self._time_series.bins is not None
            ), "This time series does not have any bins!"

            # save the original interval if there is one
            old_interval = copy.copy(self._active_interval)
            old_verbose = copy.copy(self._verbose)

            # we will keep it quiet to keep from being annoying

            self._verbose = False

            list_of_speclikes = []

            # get the bins from the time series
            # for event lists, these are from created bins
            # for binned spectra sets, these are the native bines

            these_bins = self._time_series.bins  # type: TimeIntervalSet

            if start is not None:
                assert stop is not None, "must specify a start AND a stop time"

            if stop is not None:
                assert stop is not None, "must specify a start AND a stop time"

                these_bins = these_bins.containing_interval(start, stop, inner=False)

            # loop through the intervals and create spec likes

            with progress_bar(len(these_bins), title="Creating plugins") as p:

                for i, interval in enumerate(these_bins):

                    self.set_active_time_interval(interval.to_string())

                    assert isinstance(
                        self._observed_spectrum, BinnedSpectrum
                    ), "You are attempting to create a SpectrumLike plugin from the wrong data type"

                    if extract_measured_background:

                        this_background_spectrum = self._measured_background_spectrum

                    else:

                        this_background_spectrum = self._background_spectrum

                    if this_background_spectrum is None:
                        custom_warnings.warn(
                            "No bakckground selection has been made. This plugin will contain no background!"
                        )

                    try:

                        if self._response is None:

                            sl = SpectrumLike(
                                name="%s%s%d" % (self._name, interval_name, i),
                                observation=self._observed_spectrum,
                                background=this_background_spectrum,
                                verbose=self._verbose,
                                tstart=self._tstart,
                                tstop=self._tstop,
                            )

                        else:

                            sl = DispersionSpectrumLike(
                                name="%s%s%d" % (self._name, interval_name, i),
                                observation=self._observed_spectrum,
                                background=this_background_spectrum,
                                verbose=self._verbose,
                                tstart=self._tstart,
                                tstop=self._tstop,
                            )

                        list_of_speclikes.append(sl)

                    except (NegativeBackground):

                        custom_warnings.warn(
                            "Something is wrong with interval %s. skipping." % interval
                        )

                    p.increase()

            # restore the old interval

            if old_interval is not None:

                self.set_active_time_interval(*old_interval)

            else:

                self._active_interval = None

            self._verbose = old_verbose

            return list_of_speclikes

    @classmethod
    def from_gbm_tte(
        cls,
        name,
        tte_file,
        rsp_file,
        restore_background=None,
        trigger_time=None,
        poly_order=-1,
        unbinned=True,
        verbose=True,
    ):
        """
           A plugin to natively bin, view, and handle Fermi GBM TTE data.
           A TTE event file are required as well as the associated response



           Background selections are specified as
           a comma separated string e.g. "-10-0,10-20"

           Initial source selection is input as a string e.g. "0-5"

           One can choose a background polynomial order by hand (up to 4th order)
           or leave it as the default polyorder=-1 to decide by LRT test

           :param name: name for your choosing
           :param tte_file: GBM tte event file
           :param rsp_file: Associated TTE CSPEC response file
           :param trigger_time: trigger time if needed
           :param poly_order: 0-4 or -1 for auto
           :param unbinned: unbinned likelihood fit (bool)
           :param verbose: verbose (bool)



               """

        # self._default_unbinned = unbinned

        # Load the relevant information from the TTE file

        gbm_tte_file = GBMTTEFile(tte_file)

        # Set a trigger time if one has not been set

        if trigger_time is not None:
            gbm_tte_file.trigger_time = trigger_time

        # Create the the event list

        event_list = EventListWithDeadTime(
            arrival_times=gbm_tte_file.arrival_times - gbm_tte_file.trigger_time,
            measurement=gbm_tte_file.energies,
            n_channels=gbm_tte_file.n_channels,
            start_time=gbm_tte_file.tstart - gbm_tte_file.trigger_time,
            stop_time=gbm_tte_file.tstop - gbm_tte_file.trigger_time,
            dead_time=gbm_tte_file.deadtime,
            first_channel=0,
            instrument=gbm_tte_file.det_name,
            mission=gbm_tte_file.mission,
            verbose=verbose,
        )

        if isinstance(rsp_file, str) or isinstance(rsp_file, unicode):

            # we need to see if this is an RSP2

            test = re.match("^.*\.rsp2$", rsp_file)

            # some GBM RSPs that are not marked RSP2 are in fact RSP2s
            # we need to check

            if test is None:

                with fits.open(rsp_file) as f:

                    # there should only be a header, ebounds and one spec rsp extension

                    if len(f) > 3:

                        # make test a dummy value to trigger the nest loop

                        test = -1

                        custom_warnings.warn(
                            "The RSP file is marked as a single response but in fact has multiple matrices. We will treat it as an RSP2"
                        )

            if test is not None:

                rsp = InstrumentResponseSet.from_rsp2_file(
                    rsp2_file=rsp_file,
                    counts_getter=event_list.counts_over_interval,
                    exposure_getter=event_list.exposure_over_interval,
                    reference_time=gbm_tte_file.trigger_time,
                )

            else:

                rsp = OGIPResponse(rsp_file)

        else:

            assert isinstance(
                rsp_file, InstrumentResponse
            ), "The provided response is not a 3ML InstrumentResponse"
            rsp = rsp_file

        # pass to the super class

        return cls(
            name,
            event_list,
            response=rsp,
            poly_order=poly_order,
            unbinned=unbinned,
            verbose=verbose,
            restore_poly_fit=restore_background,
            container_type=BinnedSpectrumWithDispersion,
        )

    @classmethod
    def from_gbm_cspec_or_ctime(
        cls,
        name,
        cspec_or_ctime_file,
        rsp_file,
        restore_background=None,
        trigger_time=None,
        poly_order=-1,
        verbose=True,
    ):
        """
               A plugin to natively bin, view, and handle Fermi GBM TTE data.
               A TTE event file are required as well as the associated response



               Background selections are specified as
               a comma separated string e.g. "-10-0,10-20"

               Initial source selection is input as a string e.g. "0-5"

               One can choose a background polynomial order by hand (up to 4th order)
               or leave it as the default polyorder=-1 to decide by LRT test

               :param name: name for your choosing
               :param tte_file: GBM tte event file
               :param rsp_file: Associated TTE CSPEC response file
               :param trigger_time: trigger time if needed
               :param poly_order: 0-4 or -1 for auto
               :param unbinned: unbinned likelihood fit (bool)
               :param verbose: verbose (bool)



               """

        # self._default_unbinned = unbinned

        # Load the relevant information from the TTE file

        cdata = GBMCdata(cspec_or_ctime_file, rsp_file)

        # Set a trigger time if one has not been set

        if trigger_time is not None:
            cdata.trigger_time = trigger_time

        # Create the the event list

        event_list = BinnedSpectrumSeries(
            cdata.spectrum_set,
            first_channel=0,
            mission="Fermi",
            instrument=cdata.det_name,
            verbose=verbose,
        )

        # we need to see if this is an RSP2

        if isinstance(rsp_file, str) or isinstance(rsp_file, unicode):

            test = re.match("^.*\.rsp2$", rsp_file)

            # some GBM RSPs that are not marked RSP2 are in fact RSP2s
            # we need to check

            if test is None:

                with fits.open(rsp_file) as f:

                    # there should only be a header, ebounds and one spec rsp extension

                    if len(f) > 3:
                        # make test a dummy value to trigger the nest loop

                        test = -1

                        custom_warnings.warn(
                            "The RSP file is marked as a single response but in fact has multiple matrices. We will treat it as an RSP2"
                        )

            if test is not None:

                rsp = InstrumentResponseSet.from_rsp2_file(
                    rsp2_file=rsp_file,
                    counts_getter=event_list.counts_over_interval,
                    exposure_getter=event_list.exposure_over_interval,
                    reference_time=cdata.trigger_time,
                )

            else:

                rsp = OGIPResponse(rsp_file)

        else:

            assert isinstance(
                rsp_file, InstrumentResponse
            ), "The provided response is not a 3ML InstrumentResponse"
            rsp = rsp_file

        # pass to the super class

        return cls(
            name,
            event_list,
            response=rsp,
            poly_order=poly_order,
            unbinned=False,
            verbose=verbose,
            restore_poly_fit=restore_background,
            container_type=BinnedSpectrumWithDispersion,
        )

    @classmethod
    def from_lat_lle(
        cls,
        name,
        lle_file,
        ft2_file,
        rsp_file,
        restore_background=None,
        trigger_time=None,
        poly_order=-1,
        unbinned=False,
        verbose=True,
    ):

        """
               A plugin to natively bin, view, and handle Fermi LAT LLE data.
               An LLE event file and FT2 (1 sec) are required as well as the associated response



               Background selections are specified as
               a comma separated string e.g. "-10-0,10-20"

               Initial source selection is input as a string e.g. "0-5"

               One can choose a background polynomial order by hand (up to 4th order)
               or leave it as the default polyorder=-1 to decide by LRT test

               :param name: name of the plugin
               :param lle_file: lle event file
               :param ft2_file: fermi FT2 file
               :param rsp_file: lle response file
               :param trigger_time: trigger time if needed
               :param poly_order: 0-4 or -1 for auto
               :param unbinned: unbinned likelihood fit (bool)
               :param verbose: verbose (bool)


               """

        lat_lle_file = LLEFile(lle_file, ft2_file, rsp_file)

        if trigger_time is not None:
            lat_lle_file.trigger_time = trigger_time

        # Mark channels less than 50 MeV as bad

        channel_30MeV = np.searchsorted(lat_lle_file.energy_edges[0], 30000.0) - 1

        native_quality = np.zeros(lat_lle_file.n_channels, dtype=int)

        idx = np.arange(lat_lle_file.n_channels) < channel_30MeV

        native_quality[idx] = 5

        event_list = EventListWithLiveTime(
            arrival_times=lat_lle_file.arrival_times - lat_lle_file.trigger_time,
            measurement=lat_lle_file.energies,
            n_channels=lat_lle_file.n_channels,
            live_time=lat_lle_file.livetime,
            live_time_starts=lat_lle_file.livetime_start - lat_lle_file.trigger_time,
            live_time_stops=lat_lle_file.livetime_stop - lat_lle_file.trigger_time,
            start_time=lat_lle_file.tstart - lat_lle_file.trigger_time,
            stop_time=lat_lle_file.tstop - lat_lle_file.trigger_time,
            quality=native_quality,
            first_channel=1,
            # rsp_file=rsp_file,
            instrument=lat_lle_file.instrument,
            mission=lat_lle_file.mission,
            verbose=verbose,
        )

        # pass to the super class

        rsp = OGIPResponse(rsp_file)

        return cls(
            name,
            event_list,
            response=rsp,
            poly_order=poly_order,
            unbinned=unbinned,
            verbose=verbose,
            restore_poly_fit=restore_background,
            container_type=BinnedSpectrumWithDispersion,
        )

    @classmethod
    def from_phaII(cls):

        raise NotImplementedError(
            "Reading from a generic PHAII file is not yet supportedgb"
        )

    @classmethod
    def from_polar_spectrum(
        cls,
        name,
        polar_hdf5_file,
        restore_background=None,
        trigger_time=0.0,
        poly_order=-1,
        unbinned=True,
        verbose=True,
    ):

        if not has_polarpy:

            raise RuntimeError("The polarpy module is not installed")

        # self._default_unbinned = unbinned

        # extract the polar varaibles

        polar_data = POLARData(
            polar_hdf5_file, polar_hdf5_response=None, reference_time=trigger_time
        )

        # Create the the event list

        event_list = EventListWithDeadTimeFraction(
            arrival_times=polar_data.time,
            measurement=polar_data.pha,
            n_channels=polar_data.n_channels,
            start_time=polar_data.time.min(),
            stop_time=polar_data.time.max(),
            dead_time_fraction=polar_data.dead_time_fraction,
            verbose=verbose,
            first_channel=1,
            mission="Tiangong-2",
            instrument="POLAR",
        )

        return cls(
            name,
            event_list,
            response=polar_data.rsp,
            poly_order=poly_order,
            unbinned=unbinned,
            verbose=verbose,
            restore_poly_fit=restore_background,
            container_type=BinnedSpectrumWithDispersion,
        )

    @classmethod
    def from_polar_polarization(
        cls,
        name,
        polar_hdf5_file,
        polar_hdf5_response,
        restore_background=None,
        trigger_time=0.0,
        poly_order=-1,
        unbinned=True,
        verbose=True,
    ):

        if not has_polarpy:
            raise RuntimeError("The polarpy module is not installed")

        # self._default_unbinned = unbinned

        # extract the polar varaibles

        polar_data = POLARData(polar_hdf5_file, polar_hdf5_response, trigger_time)

        # Create the the event list

        event_list = EventListWithDeadTimeFraction(
            arrival_times=polar_data.scattering_angle_time,
            measurement=polar_data.scattering_angles,
            n_channels=polar_data.n_scattering_bins,
            start_time=polar_data.scattering_angle_time.min(),
            stop_time=polar_data.scattering_angle_time.max(),
            dead_time_fraction=polar_data.scattering_angle_dead_time_fraction,
            verbose=verbose,
            first_channel=1,
            mission="Tiangong-2",
            instrument="POLAR",
            edges=polar_data.scattering_edges,
        )

        return cls(
            name,
            event_list,
            response=polar_hdf5_response,
            poly_order=poly_order,
            unbinned=unbinned,
            verbose=verbose,
            restore_poly_fit=restore_background,
            container_type=BinnedModulationCurve,
        )

    def to_polarlike(
        self,
        from_bins=False,
        start=None,
        stop=None,
        interval_name="_interval",
        extract_measured_background=False,
    ):

        assert has_polarpy, "you must have the polarpy module installed"

        assert issubclass(
            self._container_type, BinnedModulationCurve
        ), "You are attempting to create a POLARLike plugin from the wrong data type"

        if extract_measured_background:

            this_background_spectrum = self._measured_background_spectrum

        else:

            this_background_spectrum = self._background_spectrum

        if isinstance(self._response, str):
            self._response = PolarResponse(self._response)

        if not from_bins:

            assert (
                self._observed_spectrum is not None
            ), "Must have selected an active time interval"

            if this_background_spectrum is None:

                custom_warnings.warn(
                    "No background selection has been made. This plugin will contain no background!"
                )

            return PolarLike(
                name=self._name,
                observation=self._observed_spectrum,
                background=this_background_spectrum,
                response=self._response,
                verbose=self._verbose,
                #                 tstart=self._tstart,
                #                 tstop=self._tstop
            )

        else:

            # this is for a set of intervals.

            assert (
                self._time_series.bins is not None
            ), "This time series does not have any bins!"

            # save the original interval if there is one
            old_interval = copy.copy(self._active_interval)
            old_verbose = copy.copy(self._verbose)

            # we will keep it quiet to keep from being annoying

            self._verbose = False

            list_of_polarlikes = []

            # now we make one response to save time

            # get the bins from the time series
            # for event lists, these are from created bins
            # for binned spectra sets, these are the native bines

            these_bins = self._time_series.bins  # type: TimeIntervalSet

            if start is not None:
                assert stop is not None, "must specify a start AND a stop time"

            if stop is not None:
                assert stop is not None, "must specify a start AND a stop time"

                these_bins = these_bins.containing_interval(start, stop, inner=False)

            # loop through the intervals and create spec likes

            with progress_bar(len(these_bins), title="Creating plugins") as p:

                for i, interval in enumerate(these_bins):

                    self.set_active_time_interval(interval.to_string())

                    if extract_measured_background:

                        this_background_spectrum = self._measured_background_spectrum

                    else:

                        this_background_spectrum = self._background_spectrum

                    if this_background_spectrum is None:
                        custom_warnings.warn(
                            "No bakckground selection has been made. This plugin will contain no background!"
                        )

                    try:

                        pl = PolarLike(
                            name="%s%s%d" % (self._name, interval_name, i),
                            observation=self._observed_spectrum,
                            background=this_background_spectrum,
                            response=self._response,
                            verbose=self._verbose,
                            #               tstart=self._tstart,
                            #               tstop=self._tstop
                        )

                        list_of_polarlikes.append(pl)

                    except (NegativeBackground):

                        custom_warnings.warn(
                            "Something is wrong with interval %s. skipping." % interval
                        )

                    p.increase()

            # restore the old interval

            if old_interval is not None:

                self.set_active_time_interval(*old_interval)

            else:

                self._active_interval = None

            self._verbose = old_verbose

            return list_of_polarlikes
