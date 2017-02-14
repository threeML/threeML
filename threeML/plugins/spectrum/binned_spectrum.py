import numpy as np
import pandas as pd

from threeML.utils.histogram import Histogram
from threeML.utils.interval import Interval, IntervalSet
from threeML.plugins.OGIP.response import InstrumentResponse
from threeML.utils.stats_tools import sqrt_sum_of_squares

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

        return np.array([channel.channel_width for channel in self._intervals ])

class Quality(object):
    def __init__(self, quality):
        """
        simple class to formalize the quality flags used in spectra
        :param quality:
        """

        total_length = len(quality)

        good = quality == 'good'
        warn = quality == 'warn'
        bad  = quality == 'bad'

        assert total_length == sum(good) + sum(warn) +sum(bad), 'quality can only contain "good", "warn", and "bad"'

        self._good = good
        self._warn = warn
        self._bad = bad

        self._quality = quality

    def __len__(self):

        return len(self._quality)

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
        good = ogip_quality == 0
        warn = ogip_quality == 2
        bad = np.logical_and(~good, ~warn)

        quality = np.array(['good' for i in xrange(len(ogip_quality))])

        #quality[good] = 'good'
        quality[warn] = 'warn'
        quality[bad] = 'bad'

        return cls(quality)

    def to_ogip(self):
        """
        makes a quality array following the OGIP standards:
        0 = good
        2 = warn
        5 = bad

        :return:
        """

        ogip_quality = np.zeros(self._quality.shape,dtype=np.int32)

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

        quality = np.array(['good' for i in xrange(int(n_channels))])

        return cls(quality)



class BinnedSpectrum(Histogram):

    INTERVAL_TYPE = Channel

    def __init__(self, counts, exposure, ebounds, count_errors=None, sys_errors=None, quality=None, scale_factor=1., is_poisson=False, mission=None, instrument=None):
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

            ebounds = ChannelSet.from_list_of_edges(ebounds) #type: ChannelSet


        if count_errors is not None:

            assert not self._is_poisson, "Read count errors but spectrum marked Poisson"

            # convert counts to rate

            rate_errors = count_errors / self._exposure

        else:

            rate_errors = None

        if sys_errors is None:

            sys_errors = np.zeros_like(counts)

        self._sys_errors = sys_errors

        # convert rates to counts

        rates = counts / self._exposure


        if quality is not None:

            # check that we are using the 3ML quality type

            assert isinstance(quality, Quality)

            self._quality = quality

        else:

            # if there is no quality, then assume all channels are good

            self._quality = Quality.create_all_good(len(rates))


        if mission is None:

            self._mission = 'UNKNOWN'

        else:

            self._mission = mission

        if instrument is None:

            self._instrument = 'UNKNOWN'

        else:

            self._instrument = instrument


        # pass up to the binned spectrum

        super(BinnedSpectrum, self).__init__(list_of_intervals=ebounds,
                                             contents=rates,
                                             errors=rate_errors,
                                             sys_errors=sys_errors,
                                             is_poisson=is_poisson)

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
        assert self.is_poisson == False, "Cannot request errors on rates for a Poisson spectrum"

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

        assert self.is_poisson == False, "Cannot request errors on rates for a Poisson spectrum"

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

        assert self.is_poisson == False, "Cannot request errors on rates for a Poisson spectrum"

        return sqrt_sum_of_squares(self.count_errors)

    @property
    def is_poisson(self):

        return self._is_poisson

    @property
    def rate_errors(self):
        """
        If the spectrum has no Poisson error (POISSER is False in the header), this will return the STAT_ERR column
        :return: errors on the rates
        """

        assert self.is_poisson == False, "Cannot request errors on rates for a Poisson spectrum"

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

    def clone(self, new_counts=None, new_count_errors=None):
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

        return BinnedSpectrum(counts=new_counts,
                              ebounds=ChannelSet.from_list_of_edges(self.edges),
                              exposure=self._exposure,
                              count_errors=new_count_errors,
                              sys_errors=self._sys_errors,
                              quality=self._quality,
                              scale_factor=self._scale_factor,
                              is_poisson=self._is_poisson,
                              mission=self._mission,
                              instrument=self._instrument)

    @classmethod
    def from_pandas(cls,pandas_dataframe,exposure,scale_factor=1.,is_poisson=False,mission=None,instrument=None):
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

        emin = np.array(pandas_dataframe['emin'])
        emax = np.array(pandas_dataframe['emax'])
        counts = np.array(pandas_dataframe['counts'])
        ebounds = emin.tolist()
        ebounds.append(emax[-1])


        ebounds = ChannelSet.from_list_of_edges(ebounds)

        # default optional parameters
        count_errors = None
        sys_errors = None
        quality = None

        if 'count_errors' in pandas_dataframe.keys():

            count_errors = np.array(pandas_dataframe['count_errors'])

        if 'sys_errors' in pandas_dataframe.keys():
            sys_errors = np.array(pandas_dataframe['sys_errors'])

        if 'quality' in pandas_dataframe.keys():
            quality = Quality(np.array(pandas_dataframe['quality']))

        return cls(counts=counts,
                   exposure=exposure,
                   ebounds=ebounds,
                   count_errors=count_errors,
                   sys_errors=sys_errors,
                   quality=quality,
                   scale_factor=scale_factor,
                   is_poisson=is_poisson,
                   mission=mission,
                   instrument=instrument)

    def to_pandas(self,use_rate=True):
        """
        make a pandas table from the spectrum.

        :param use_rate: if the table should use rates or counts
        :return:
        """



        if use_rate:

            out_name = 'rates'
            out_values = self.rates

        else:

            out_name = 'counts'
            out_values = self.rates * self.exposure

        out_dict = {'emin': self.starts, 'emax': self.stops,out_name:out_values, 'quality': self.quality}

        if self.rate_errors is not None:

            if use_rate:

                out_dict['rate_errors'] = self.rate_errors

            else:

                out_dict['count_errors'] =self.rate_errors * self.exposure

        if self.sys_errors is not None:

            out_dict['sys_errors'] = None


        return pd.DataFrame(out_dict)


class BinnedSpectrumWithDispersion(BinnedSpectrum):

    def __init__(self, counts, exposure, response, count_errors=None, sys_errors=None, quality=None, scale_factor=1., is_poisson=False, mission=None, instrument=None ):
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


        assert isinstance(response, InstrumentResponse), 'The response is not a valid instance of InstrumentResponse'

        self._rsp = response

        ebounds = ChannelSet.from_instrument_response(response)



        super(BinnedSpectrumWithDispersion, self).__init__(counts=counts,
                                                           exposure=exposure,
                                                           ebounds=ebounds,
                                                           count_errors=count_errors,
                                                           sys_errors=sys_errors,
                                                           quality=quality,
                                                           scale_factor=scale_factor,
                                                           is_poisson=is_poisson,
                                                           mission=mission,
                                                           instrument=instrument)


    @property
    def response(self):

        return self._rsp

    def clone(self, new_counts=None, new_count_errors=None):
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

        return BinnedSpectrumWithDispersion(counts=new_counts,
                                            response=self._rsp,
                                            count_errors=new_count_errors,
                                            sys_errors=self._sys_errors,
                                            quality=self._quality,
                                            scale_factor=self._scale_factor,
                                            is_poisson=self._is_poisson,
                                            mission=self._mission,
                                            instrument=self._instrument)











