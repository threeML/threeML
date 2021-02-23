from __future__ import division
from builtins import zip
from builtins import range
from past.utils import old_div
from builtins import object
import numpy as np
import pandas as pd

from threeML.utils.OGIP.response import InstrumentResponse
from threeML.utils.histogram import Histogram
from threeML.utils.interval import Interval, IntervalSet
from threeML.utils.statistics.stats_tools import sqrt_sum_of_squares


class Channel(Interval):
    @property
    def channel_width(self):

        return self._get_width()


class ChannelSet(IntervalSet):

    INTERVAL_TYPE = Channel

    @classmethod
    def from_instrument_response(cls, instrument_response):
        """
        Build EBOUNDS interval from an instrument response


        :param instrument_response:
        :return:
        """

        new_ebounds = cls.from_list_of_edges(instrument_response.ebounds)

        return new_ebounds

    @property
    def channels_widths(self):

        return np.array([channel.channel_width for channel in self._intervals])


class Quality(object):
    def __init__(self, quality):
        """
        simple class to formalize the quality flags used in spectra
        :param quality: a quality array
        """

        # total_length = len(quality)

        quality = quality.astype(str)

        n_elements = 1
        for dim in quality.shape:

            n_elements *= dim

        good = quality == "good"
        warn = quality == "warn"
        bad = quality == "bad"

        assert n_elements == (
            good.sum() + warn.sum() + bad.sum()
        ), 'quality can only contain "good", "warn", and "bad"'

        self._good = good
        self._warn = warn
        self._bad = bad

        self._quality = quality

    def __len__(self):

        return len(self._quality)

    def get_slice(self, idx):

        return Quality(self._quality[idx, :])

    @property
    def good(self):
        return self._good

    @property
    def warn(self):
        return self._warn

    @property
    def bad(self):
        return self._bad

    @property
    def n_elements(self):
        return len(self._quality)

    @classmethod
    def from_ogip(cls, ogip_quality):
        ogip_quality = np.atleast_1d(ogip_quality)
        good = ogip_quality == 0
        warn = ogip_quality == 2
        bad = np.logical_and(~good, ~warn)

        quality = np.empty_like(ogip_quality, dtype="|S4")
        quality[:] = "good"

        # quality[good] = 'good'
        quality[warn] = "warn"
        quality[bad] = "bad"

        return cls(quality)

    def to_ogip(self):
        """
        makes a quality array following the OGIP standards:
        0 = good
        2 = warn
        5 = bad

        :return:
        """

        ogip_quality = np.zeros(self._quality.shape, dtype=np.int32)

        ogip_quality[self.warn] = 2
        ogip_quality[self.bad] = 5

        return ogip_quality

    @classmethod
    def create_all_good(cls, n_channels):
        """
        construct a quality object with all good channels
        :param n_channels:
        :return:
        """

        quality = np.array(["good" for i in range(int(n_channels))])

        return cls(quality)


class BinnedSpectrum(Histogram):

    INTERVAL_TYPE = Channel

    def __init__(
        self,
        counts,
        exposure,
        ebounds,
        count_errors=None,
        sys_errors=None,
        quality=None,
        scale_factor=1.0,
        is_poisson=False,
        mission=None,
        instrument=None,
        tstart=None,
        tstop=None,
    ):
        """
        A general binned histogram of either Poisson or non-Poisson rates. While the input is in counts, 3ML spectra work
        in rates, so this class uses the exposure to construct the rates from the counts.

        :param counts: an array of counts
        :param exposure: the exposure for the counts
        :param ebounds: the len(counts) + 1 energy edges of the histogram or an instance of EBOUNDSIntervalSet
        :param count_errors: (optional) the count errors for the spectra
        :param sys_errors: (optional) systematic errors on the spectrum
        :param quality: quality instance marking good, bad and warned channels. If not provided, all channels are assumed to be good
        :param scale_factor: scaling parameter of the spectrum
        :param is_poisson: if the histogram is Poisson
        :param mission: the mission name
        :param instrument: the instrument name
        """

        # attach the parameters ot the object

        self._is_poisson = is_poisson

        self._exposure = exposure

        self._scale_factor = scale_factor

        # if we do not have a ChannelSet,

        if not isinstance(ebounds, ChannelSet):

            # make one from the edges

            ebounds = ChannelSet.from_list_of_edges(ebounds)  # type: ChannelSet

        self._ebounds = ebounds

        if count_errors is not None:

            assert not self._is_poisson, "Read count errors but spectrum marked Poisson"

            # convert counts to rate

            rate_errors = old_div(count_errors, self._exposure)

        else:

            rate_errors = None

        if sys_errors is None:

            sys_errors = np.zeros_like(counts)

        self._sys_errors = sys_errors

        # convert rates to counts

        rates = old_div(counts, self._exposure)

        if quality is not None:

            # check that we are using the 3ML quality type

            assert isinstance(quality, Quality)

            self._quality = quality

        else:

            # if there is no quality, then assume all channels are good

            self._quality = Quality.create_all_good(len(rates))

        if mission is None:

            self._mission = "UNKNOWN"

        else:

            self._mission = mission

        if instrument is None:

            self._instrument = "UNKNOWN"

        else:

            self._instrument = instrument

        self._tstart = tstart

        self._tstop = tstop

        # pass up to the binned spectrum

        super(BinnedSpectrum, self).__init__(
            list_of_intervals=ebounds,
            contents=rates,
            errors=rate_errors,
            sys_errors=sys_errors,
            is_poisson=is_poisson,
        )

    @property
    def n_channel(self):

        return len(self)

    @property
    def rates(self):
        """
        :return: rates per channel
        """
        return self._contents

    @property
    def total_rate(self):
        """
        :return: total rate
        """

        return self._contents.sum()

    @property
    def total_rate_error(self):
        """
        :return: total rate error
        """
        assert (
            self.is_poisson == False
        ), "Cannot request errors on rates for a Poisson spectrum"

        return sqrt_sum_of_squares(self._errors)

    @property
    def counts(self):
        """
        :return: counts per channel
        """

        return self._contents * self.exposure

    @property
    def count_errors(self):
        """
        :return: count error per channel
        """

        # VS: impact of this change is unclear to me, it seems to make sense and the tests pass
        if self.is_poisson:
            return None
        else:
            return self._errors * self.exposure

    @property
    def total_count(self):
        """
        :return: total counts
        """

        return self.counts.sum()

    @property
    def total_count_error(self):
        """
        :return: total count error
        """

        # VS: impact of this change is unclear to me, it seems to make sense and the tests pass
        if self.is_poisson:
            return None
        else:
            return sqrt_sum_of_squares(self.count_errors)

    @property
    def tstart(self):

        return self._tstart

    @property
    def tstop(self):

        return self._tstop

    @property
    def is_poisson(self):

        return self._is_poisson

    @property
    def rate_errors(self):
        """
        If the spectrum has no Poisson error (POISSER is False in the header), this will return the STAT_ERR column
        :return: errors on the rates
        """

        if self.is_poisson:
            return None
        else:
            return self._errors

    @property
    def n_channels(self):

        return len(self)

    @property
    def sys_errors(self):
        """
        Systematic errors per channel. This is nonzero only if the SYS_ERR column is present in the input file.

        :return: the systematic errors stored in the input spectrum
        """
        return self._sys_errors

    @property
    def exposure(self):
        """
        Exposure in seconds

        :return: exposure
        """
        return self._exposure

    @property
    def quality(self):
        return self._quality

    @property
    def scale_factor(self):

        return self._scale_factor

    @property
    def mission(self):

        return self._mission

    @property
    def instrument(self):

        return self._instrument

    def clone(
        self,
        new_counts=None,
        new_count_errors=None,
        new_exposure=None,
        new_scale_factor=None,
    ):
        """
        make a new spectrum with new counts and errors and all other
        parameters the same


        :param new_counts: new counts for the spectrum
        :param new_count_errors: new errors from the spectrum
        :return:
        """

        if new_counts is None:
            new_counts = self.counts
            new_count_errors = self.count_errors

        if new_exposure is None:
            new_exposure = self.exposure

        if new_scale_factor is None:

            new_scale_factor = self._scale_factor

        return BinnedSpectrum(
            counts=new_counts,
            ebounds=ChannelSet.from_list_of_edges(self.edges),
            exposure=new_exposure,
            count_errors=new_count_errors,
            sys_errors=self._sys_errors,
            quality=self._quality,
            scale_factor=new_scale_factor,
            is_poisson=self._is_poisson,
            mission=self._mission,
            instrument=self._instrument,
        )

    @classmethod
    def from_pandas(
        cls,
        pandas_dataframe,
        exposure,
        scale_factor=1.0,
        is_poisson=False,
        mission=None,
        instrument=None,
    ):
        """
        Build a spectrum from data contained within a pandas data frame.

        The required columns are:

        'emin': low energy bin edge
        'emax': high energy bin edge
        'counts': the counts in each bin

        Optional column names are:

        'count_errors': errors on the counts for non-Poisson data
        'sys_errors': systematic error per channel
        'quality' list of 3ML quality flags 'good', 'warn', 'bad'


        :param pandas_dataframe: data frame containing information to be read into spectrum
        :param exposure: the exposure  of the spectrum
        :param scale_factor: the scale factor of the spectrum
        :param is_poisson: if the data are Poisson distributed
        :param mission: (optional) the mission name
        :param instrument: (optional) the instrument name
        :return:
        """

        # get the required columns

        emin = np.array(pandas_dataframe["emin"])
        emax = np.array(pandas_dataframe["emax"])
        counts = np.array(pandas_dataframe["counts"])
        ebounds = emin.tolist()
        ebounds.append(emax[-1])

        ebounds = ChannelSet.from_list_of_edges(ebounds)

        # default optional parameters
        count_errors = None
        sys_errors = None
        quality = None

        if "count_errors" in list(pandas_dataframe.keys()):

            count_errors = np.array(pandas_dataframe["count_errors"])

        if "sys_errors" in list(pandas_dataframe.keys()):
            sys_errors = np.array(pandas_dataframe["sys_errors"])

        if "quality" in list(pandas_dataframe.keys()):
            quality = Quality(np.array(pandas_dataframe["quality"]))

        return cls(
            counts=counts,
            exposure=exposure,
            ebounds=ebounds,
            count_errors=count_errors,
            sys_errors=sys_errors,
            quality=quality,
            scale_factor=scale_factor,
            is_poisson=is_poisson,
            mission=mission,
            instrument=instrument,
        )

    def to_pandas(self, use_rate=True):
        """
        make a pandas table from the spectrum.

        :param use_rate: if the table should use rates or counts
        :return:
        """

        if use_rate:

            out_name = "rates"
            out_values = self.rates

        else:

            out_name = "counts"
            out_values = self.rates * self.exposure

        out_dict = {
            "emin": self.starts,
            "emax": self.stops,
            out_name: out_values,
            "quality": self.quality,
        }

        if self.rate_errors is not None:

            if use_rate:

                out_dict["rate_errors"] = self.rate_errors

            else:

                out_dict["count_errors"] = self.rate_errors * self.exposure

        if self.sys_errors is not None:

            out_dict["sys_errors"] = None

        return pd.DataFrame(out_dict)

    @classmethod
    def from_time_series(cls, time_series, use_poly=False, from_model=False, **kwargs):
        """

        :param time_series:
        :param use_poly:
        :return:
        """

        pha_information = time_series.get_information_dict(use_poly)

        is_poisson = True

        if use_poly:
            is_poisson = False

        return cls(
            instrument=pha_information["instrument"],
            mission=pha_information["telescope"],
            tstart=pha_information["tstart"],
            tstop=pha_information["tstart"]+pha_information["telapse"],
            #telapse=pha_information["telapse"],
            # channel=pha_information['channel'],
            counts=pha_information["counts"],
            count_errors=pha_information["counts error"],
            quality=pha_information["quality"],
            #grouping=pha_information["grouping"],
            exposure=pha_information["exposure"],
            #backscale=1.0,
            is_poisson=is_poisson,
            ebounds=pha_information["edges"]
        )

    def __add__(self, other):
        assert self == other, "The bins are not equal"

        new_sys_errors = self.sys_errors
        if new_sys_errors is None:
            new_sys_errors = other.sys_errors
        elif other.sys_errors is not None:
            new_sys_errors += other.sys_errors

        new_exposure = self.exposure + other.exposure

        if self.count_errors is None and other.count_errors is None:
            new_count_errors = None
        else:
            assert (
                self.count_errors is not None or other.count_errors is not None
            ), "only one of the two spectra have errors, can not add!"
            new_count_errors = (self.count_errors ** 2 + other.count_errors ** 2) ** 0.5

        new_counts = self.counts + other.counts

        new_spectrum = self.clone(
            new_counts=new_counts,
            new_count_errors=new_count_errors,
            new_exposure=new_exposure,
        )

        if self.tstart is None:
            if other.tstart is None:
                new_spectrum._tstart = None

            else:

                new_spectrum._tstart = other.tstart
        elif other.tstart is None:

            new_spectrum._tstart = self.tstart

        else:

            new_spectrum._tstart = min(self.tstart, other.tstart)

        if self.tstop is None:
            if other.tstop is None:
                new_spectrum._tstop = None

            else:

                new_spectrum._tstop = other.tstop
        elif other.tstop is None:

            new_spectrum._tstop = self.tstop

        else:

            new_spectrum._tstop = min(self.tstop, other.tstop)

        return new_spectrum

    def add_inverse_variance_weighted(self, other):
        assert self == other, "The bins are not equal"

        if self.is_poisson or other.is_poisson:
            raise Exception("Inverse_variance_weighting not implemented for poisson")

        new_sys_errors = self.sys_errors
        if new_sys_errors is None:
            new_sys_errors = other.sys_errors
        elif other.sys_errors is not None:
            new_sys_errors += other.sys_errors

        new_exposure = self.exposure + other.exposure

        new_rate_errors = np.array(
            [
                (e1 ** -2 + e2 ** -2) ** -0.5
                for e1, e2 in zip(self.rate_errors, other._errors)
            ]
        )
        new_rates = (
            np.array(
                [
                    (c1 * e1 ** -2 + c2 * e2 ** -2)
                    for c1, e1, c2, e2 in zip(
                        self.rates, self._errors, other.rates, other._errors
                    )
                ]
            )
            * new_rate_errors ** 2
        )

        new_count_errors = new_rate_errors * new_exposure
        new_counts = new_rates * new_exposure

        new_counts[np.isnan(new_counts)] = 0
        new_count_errors[np.isnan(new_count_errors)] = 0

        new_spectrum = self.clone(
            new_counts=new_counts, new_count_errors=new_count_errors
        )

        new_spectrum._exposure = new_exposure

        if self.tstart is None:
            if other.tstart is None:
                new_spectrum._tstart = None

            else:

                new_spectrum._tstart = other.tstart
        elif other.tstart is None:

            new_spectrum._tstart = self.tstart

        else:

            new_spectrum._tstart = min(self.tstart, other.tstart)

        if self.tstop is None:
            if other.tstop is None:
                new_spectrum._tstop = None

            else:

                new_spectrum._tstop = other.tstop
        elif other.tstop is None:

            new_spectrum._tstop = self.tstop

        else:

            new_spectrum._tstop = min(self.tstop, other.tstop)

        return new_spectrum


class BinnedSpectrumWithDispersion(BinnedSpectrum):
    def __init__(
        self,
        counts,
        exposure,
        response,
        count_errors=None,
        sys_errors=None,
        quality=None,
        scale_factor=1.0,
        is_poisson=False,
        mission=None,
        instrument=None,
        tstart=None,
        tstop=None,
    ):
        """
        A binned spectrum that must be deconvolved via a dispersion or response matrix


        :param counts:
        :param exposure:
        :param response:
        :param count_errors:
        :param sys_errors:
        :param quality:
        :param scale_factor:
        :param is_poisson:
        :param mission:
        :param instrument:
        """

        assert isinstance(
            response, InstrumentResponse
        ), "The response is not a valid instance of InstrumentResponse"

        self._rsp = response

        ebounds = ChannelSet.from_instrument_response(response)

        super(BinnedSpectrumWithDispersion, self).__init__(
            counts=counts,
            exposure=exposure,
            ebounds=ebounds,
            count_errors=count_errors,
            sys_errors=sys_errors,
            quality=quality,
            scale_factor=scale_factor,
            is_poisson=is_poisson,
            mission=mission,
            instrument=instrument,
            tstart=tstart,
            tstop=tstop,
        )

    @property
    def response(self):

        return self._rsp

    @classmethod
    def from_time_series(
        cls, time_series, response=None, use_poly=False, extract=False
    ):
        """

        :param time_series:
        :param use_poly:
        :return:
        """

        assert not (
            use_poly and extract
        ), "cannot extract background counts and use the poly"

        pha_information = time_series.get_information_dict(use_poly, extract)

        is_poisson = True

        if use_poly:
            is_poisson = False

        return cls(
            instrument=pha_information["instrument"],
            mission=pha_information["telescope"],
            tstart=pha_information["tstart"],
            tstop=pha_information["tstart"] + pha_information["telapse"],
            # channel=pha_information['channel'],
            counts=pha_information["counts"],
            count_errors=pha_information["counts error"],
            quality=pha_information["quality"],
            # grouping=pha_information['grouping'],
            exposure=pha_information["exposure"],
            response=response,
            scale_factor=1.0,
            is_poisson=is_poisson,
        )

    def clone(
        self,
        new_counts=None,
        new_count_errors=None,
        new_sys_errors=None,
        new_exposure=None,
        new_scale_factor=None,
    ):
        """
        make a new spectrum with new counts and errors and all other
        parameters the same


        :param new_sys_errors:
        :param new_exposure:
        :param new_scale_factor:
        :param new_counts: new counts for the spectrum
        :param new_count_errors: new errors from the spectrum
        :return:
        """

        if new_counts is None:
            new_counts = self.counts
            new_count_errors = self.count_errors

        if new_sys_errors is None:
            new_sys_errors = self.sys_errors

        if new_exposure is None:
            new_exposure = self.exposure

        if new_scale_factor is None:

            new_scale_factor = self._scale_factor

        return BinnedSpectrumWithDispersion(
            counts=new_counts,
            exposure=new_exposure,
            response=self._rsp,
            count_errors=new_count_errors,
            sys_errors=new_sys_errors,
            quality=self._quality,
            scale_factor=new_scale_factor,
            is_poisson=self._is_poisson,
            mission=self._mission,
            instrument=self._instrument,
        )

    def __add__(self, other):
        # TODO implement equality in InstrumentResponse class
        assert self.response is other.response

        new_spectrum = super(BinnedSpectrumWithDispersion, self).__add__(other)

        return new_spectrum
