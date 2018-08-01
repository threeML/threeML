import numpy as np
import pandas as pd

#from threeML.utils.OGIP.response import InstrumentResponse
from threeML.utils.spectrum.binned_spectrum import BinnedSpectrum
from threeML.utils.histogram import Histogram
from threeML.utils.interval import Interval, IntervalSet
from threeML.utils.statistics.stats_tools import sqrt_sum_of_squares


class ScatteringChannel(Interval):

    @property
    def channel_width(self):

        return self._get_width()


class ScatteringChannelSet(IntervalSet):

    INTERVAL_TYPE = ScatteringChannel

    @classmethod
    def from_instrument_response(cls, instrument_response):
        """
        Build EBOUNDS interval from an instrument response


        :param instrument_response:
        :return:
        """
        raise NotImplementedError('Under Construction')

        new_ebounds = cls.from_list_of_edges(instrument_response.ebounds)

        return new_ebounds

    @property
    def channels_widths(self):

        return np.array([channel.channel_width for channel in self._intervals ])


class BinnedModulationCurve(BinnedSpectrum):

    INTERVAL_TYPE = ScatteringChannel

    def __init__(self, counts, exposure, abounds, count_errors=None, sys_errors=None, quality=None, scale_factor=1.,
                 is_poisson=False, mission=None, instrument=None, tstart=None, tstop=None):
        """
        A general binned histogram of either Poisson or non-Poisson rates. While the input is in counts, 3ML spectra work
        in rates, so this class uses the exposure to construct the rates from the counts.

        :param counts: an array of counts
        :param exposure: the exposure for the counts
        :param abounds: the len(counts) + 1 energy edges of the histogram or an instance of EBOUNDSIntervalSet
        :param count_errors: (optional) the count errors for the spectra
        :param sys_errors: (optional) systematic errors on the spectrum
        :param quality: quality instance marking good, bad and warned channels. If not provided, all channels are assumed to be good
        :param scale_factor: scaling parameter of the spectrum
        :param is_poisson: if the histogram is Poisson
        :param mission: the mission name
        :param instrument: the instrument name
        """


        super(BinnedModulationCurve, self).__init__(counts,
                                                    exposure,
                                                    abounds,
                                                    count_errors=None,
                                                    sys_errors=None,
                                                    quality=None,
                                                    scale_factor=1.,
                                                    is_poisson=False,
                                                    mission=None,
                                                    instrument=None,
                                                    tstart=None,
                                                    tstop=None)

    @classmethod
    def from_time_series(cls, time_series, use_poly=False):
        """

        :param time_series:
        :param use_poly:
        :return:
        """


        pha_information = time_series.get_information_dict(use_poly)

        is_poisson = True

        if use_poly:
            is_poisson = False

        return cls(instrument=pha_information['instrument'],
                   mission=pha_information['telescope'],
                   tstart=pha_information['tstart'],
                   tstop=pha_information['tstart'] + pha_information['telapse'],
                   #channel=pha_information['channel'],
                   counts =pha_information['counts'],
                   count_errors=pha_information['counts error'],
                   quality=pha_information['quality'],
                   #grouping=pha_information['grouping'],
                   exposure=pha_information['exposure'],
                   response=response,
                   scale_factor=1.,
                   is_poisson=is_poisson)



        
    def clone(self, new_counts=None, new_count_errors=None, new_exposure=None, new_scale_factor=None):
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

        return BinnedModulationCurve(counts=new_counts,
                              abounds=ScatteringChannelSet.from_list_of_edges(self.edges),
                              exposure=new_exposure,
                              count_errors=new_count_errors,
                              sys_errors=self._sys_errors,
                              quality=self._quality,
                              scale_factor=new_scale_factor,
                              is_poisson=self._is_poisson,
                              mission=self._mission,
                              instrument=self._instrument)


