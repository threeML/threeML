import numpy as np
import os
import warnings

import pandas as pd

from threeML.utils.histogram import Histogram
from threeML.utils.interval import Interval, IntervalSet
from threeML.plugins.OGIP.response import InstrumentResponse

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

        self._quality = quality

    @property
    def good(self):
        return self._quality == 'good'

    def warn(self):
        return self._quality == 'warn'

    @property
    def bad(self):
        return self._quality == 'bad'

    @property
    def n_elements(self):
        return len(self._quality)

    @classmethod
    def from_ogip(cls, ogip_quality):
        good = ogip_quality == 0
        warn = ogip_quality == 2
        bad = np.logical_and(~good, ~warn)

        quality = np.empty_like(ogip_quality, dtype=str)

        quality[good] = 'good'
        quality[warn] = 'warn'
        quality[bad] = 'bad'

        return cls(quality)

    @classmethod
    def all_good(cls, n_channels):
        """
        construct a quality object with all good channels
        :param n_channels:
        :return:
        """

        quality = np.array(['good']* int(n_channels))

        return cls(quality)



class BinnedSpectrum(Histogram):

    INTERVAL_TYPE = Channel

    def __init__(self, counts, exposure, ebounds, count_errors=None, sys_errors=None, quality=None, scale_factor=1., is_poisson=False):
        """
        A general binned histogram of either Poisson or non-Poisson rates. While the input is in counts, 3ML spectra work
        in rates, so this class uses the exposure to construct the rates from the counts. While it is possible to
        construct a histogram directly, this class provides methods to construct histograms from PHA and ROOT (not yet!)
        files directly.



        :param counts: an array of counts
        :param exposure: the exposure for the counts
        :param ebounds: the len(counts) + 1 energy edges of the histogram or an instance of EBOUNDSIntervalSet
        :param is_poisson: if the histogram is Poisson
        :param count_errors: (optional) count errors for non-Poisson data
        :param sys_errors: (optional) systematic error per channel
        """

        self._is_poisson = is_poisson
        self._exposure = exposure


        if not isinstance(ebounds, ChannelSet):

            ebounds = ChannelSet.from_list_of_edges(ebounds)


        if count_errors is not None:

            assert not self._is_poisson, "Read count errors but spectrum marked Poisson"

            # convert counts to rate

            rate_errors = count_errors / self._exposure

        else:

            rate_errors = None

        if sys_errors is None:

            sys_errors = np.zeros_like(counts)

        # convert rates to counts

        rates = counts / self._exposure


        if quality is not None:

            # check that we are using the 3ML quality type

            assert isinstance(quality, Quality)

            self._quality = quality

        else:

            # if there is no quality, then assume all channels are good

            self._quality = Quality.all_good(len(rates))




        self._scale_factor = scale_factor


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



    @classmethod
    def from_text_file(cls, file_name, **kwargs):


        file_df = pd.read_table(file_name,**kwargs)

    @classmethod
    def from_pandas(cls,pandas_dataframe):

        pass

    @classmethod
    def from_astropy_table(cls, astropy_table):

        pass

    def to_text(self, file_name):

        pass

    def to_astropy(self):

        pass

    def to_pandas(self):

        pass



class BinnedSpectrumWithDispersion(BinnedSpectrum):

    def __init__(self, counts, exposure, response, count_errors=None, sys_errors=None, quality=None, scale_factor=1., is_poisson=False ):
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
                                                           is_poisson=is_poisson)


    @property
    def response(self):

        return self._rsp










