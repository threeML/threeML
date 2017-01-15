import numpy as np


class Quality(object):
    def __init__(self, quality):
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

        return Quality(quality)


class BinnedSpectrum(object):
    def __init__(self, counts, exposure, ebounds, is_poisson=False, count_errors=None, sys_errors=None, response=None,
                 ancillary_file=None, telescope=None, instrument=None, quality=None, scale_factor=None, background=None,
                 file_name=None):
        """
        A general binned histogram of either Poisson or non-Poisson rates. While the input is in counts, 3ML spectra work
        in rates, so this class uses the exposure to construct the rates from the counts. While it is possible to
        construct a histogram directly, this class provides methods to construct histograms from PHA and ROOT (not yet!)
        files directly.



        :param counts: an array of counts
        :param exposure: the exposure for the counts
        :param ebounds: the len(counts) + 1 energy edges of the histogram
        :param is_poisson: if the histogram is Poisson
        :param count_errors: (optional) count errors for non-Poisson data
        :param sys_errors: (options) systematic error per channel
        :param response: (optional) instance of InstrumentResponse
        :param ancillary_file: (optional) ancillary file
        :param telescope: (optional) telescope name
        :param instrument: (optional) instrument name
        :param quality: (optional) Quality object to specify quality per channel
        :param scale_factor: (optional) background scale factor
        :param background: (optional) BinnedBackgroundSpectrum object
        :param file_name: (optional) file name associated to the spectrum
        """

        self._n_channels = len(counts)
        self._is_poisson = is_poisson
        self._ebounds = ebounds
        self._exposure = exposure

        assert self._n_channels == len(
            self._ebounds) + 1, "read %d channels but %s energy boundaries. Should be n+1" % (
            self._n_channels, len(self._ebounds))

        if count_errors is not None:
            assert self._n_channels == len(count_errors), "read %d channels but %s count errors. Should be equal" % (
                self._n_channels, len(count_errors))

            assert not self._is_poisson, "Read count errors but spectrum marked Poisson"

            # convert counts to rate

            self._rate_errors = count_errors / self._exposure

        if sys_errors is not None:
            assert self._n_channels == len(sys_errors), "read %d channels but %s sys errors. Should be equal" % (
                self._n_channels, len(sys_errors))

            self._sys_errors = sys_errors

        else:

            self._sys_errors = np.zeros_like(counts)

        # convert rates to counts

        self._rates = counts / self._exposure

        self._scale_factor = scale_factor

        self._response = response

        self._ancillary_file = ancillary_file

        # Quality comes in 3 categories: good, warn, bad
        # therefore, any quality flags in files must be
        # converted to these values.
        #

        if quality is None:

            self._quality = Quality(np.array(['good'] * len(self._rates)))

        else:

            assert isinstance(quality, Quality), "quality argument must be of type qaulity"

            self._quality = quality

        if background is not None:

            assert isinstance(background, BinnedBackgroundSpectrum)

        else:

            self._background = None

        self._file_name = file_name

        self._telescope = telescope

        self._instrument = instrument

    @classmethod
    def from_pha_file(cls, pha_file_name, spectrum_number, file_type='observed'):

        pass

    @classmethod
    def from_pha_instance(cls, pha_instance, spectrum_number, file_type):

        pass

    @classmethod
    def from_ROOT(cls):

        raise NotImplementedError("Instantiation from ROOT files is not yet implemented")

    @property
    def rates(self):
        """
        :return: rates per channel
        """
        return self._rates

    @property
    def rate_errors(self):
        """
        If the spectrum has no Poisson error (POISSER is False in the header), this will return the STAT_ERR column
        :return: errors on the rates
        """

        assert self.is_poisson() == False, "Cannot request errors on rates for a Poisson spectrum"

        return self._rate_errors

    @property
    def n_channels(self):

        return self._n_channels

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
    def background(self):
        """
        Returns the background file definied in the header, or None if there is none defined
p
        :return: a path to a file, or None
        """

        return self._background

    @property
    def scale_factor(self):
        """
        This is a scale factor (in the BACKSCAL keyword) which must be used to rescale background and source
        regions

        :return:
        """
        return self._scale_factor

    @property
    def response_file(self):
        """
            Returns the response file definied in the header, or None if there is none defined

            :return: a path to a file, or None
            """
        return self._response

    @property
    def ancillary_file(self):
        """
            Returns the ancillary file defined in the header, or None if there is none defined

            :return: a path to a file, or None
            """
        return self._ancillary_file

    @property
    def instrument(self):
        """
        Returns the name of the mission used to make the observation
        :return: a string
        """

        if self._instrument is None:
            return 'UNKNOWN'

        return self._instrument

    @property
    def instrument(self):
        """
        Returns the name of the instrument used to make the observation
        :return: a string
        """

        if self._instrument is None:
            return 'UNKNOWN'

        return self._instrument

    def is_poisson(self):
        """
        Returns whether the spectrum has Poisson errors or not

        :return: True or False
        """

        return self._is_poisson

    @property
    def quality(self):
        """
        Return the native quality of the PHA file
        :return:
        """

        return self._quality


class BinnedBackgroundSpectrum(BinnedSpectrum):
    def __init__(self, counts, exposure, ebounds, is_poisson=False, count_errors=None, sys_errors=None,
                 ancillary_file=None, telescope=None, instrument=None, quality=None,
                 file_name=None):
        """
        A general binned histogram of either Poisson or non-Poisson rates that are background. While the input is in counts, 3ML spectra work
        in rates, so this class uses the exposure to construct the rates from the counts. While it is possible to
        construct a histogram directly, this class provides methods to construct histograms from PHA and ROOT (not yet!)
        files directly.

        This class is a simply a wrapper around the BinnedSpectrum Class that automates setting the non-needed keywords to their correct values
        for a background spectrum. A similar object can be constructed from the BinnedSpectrum alone, but the associated background object
        of a BinnedSpectrum instantiation MUST be a subclass of this type.

        :param counts: an array of counts
        :param exposure: the exposure for the counts
        :param ebounds: the len(counts) + 1 energy edges of the histogram
        :param is_poisson: if the histogram is Poisson
        :param count_errors: (optional) count errors for non-Poisson data
        :param sys_errors: (options) systematic error per channel
        :param ancillary_file: (optional) ancillary file
        :param telescope: (optional) telescope name
        :param instrument: (optional) instrument name
        :param quality: (optional) Quality object to specify quality per channel
        :param file_name: (optional) file name associated to the spectrum
        """

        super(BinnedBackgroundSpectrum, self).__init__(counts,
                                                       exposure,
                                                       ebounds,
                                                       is_poisson,
                                                       count_errors,
                                                       sys_errors,
                                                       response=None,
                                                       ancillary_file=ancillary_file,
                                                       telescope=telescope,
                                                       instrument=instrument,
                                                       quality=quality,
                                                       scale_factor=None,
                                                       background=None,
                                                       file_name=file_name)


class BinnedPoissonSpectrum(BinnedSpectrum):
    def __init__(self, counts, exposure, ebounds, sys_errors=None, response=None,
                 ancillary_file=None, telescope=None, instrument=None, quality=None, scale_factor=None, background=None,
                 file_name=None):
        """
           A general binned histogram of Poisson rates. While the input is in counts, 3ML spectra work
           in rates, so this class uses the exposure to construct the rates from the counts. While it is possible to
           construct a histogram directly, this class provides methods to construct histograms from PHA and ROOT (not yet!)
           files directly.

           This is a subclass of the BinnedSpectrum class that automates setting the keywords for a Poisson spectrum



           :param counts: an array of counts
           :param exposure: the exposure for the counts
           :param ebounds: the len(counts) + 1 energy edges of the histogram
           :param sys_errors: (options) systematic error per channel
           :param response: (optional) instance of InstrumentResponse
           :param ancillary_file: (optional) ancillary file
           :param telescope: (optional) telescope name
           :param instrument: (optional) instrument name
           :param quality: (optional) Quality object to specify quality per channel
           :param scale_factor: (optional) background scale factor
           :param background: (optional) BinnedBackgroundSpectrum object
           :param file_name: (optional) file name associated to the spectrum
           """

        super(BinnedPoissonSpectrum, self).__init__(self,
                                                    counts,
                                                    exposure,
                                                    ebounds,
                                                    is_poisson=True,
                                                    count_errors=None,
                                                    sys_errors=sys_errors,
                                                    response=response,
                                                    ancillary_file=ancillary_file,
                                                    telescope=telescope,
                                                    instrument=instrument,
                                                    quality=quality,
                                                    scale_factor=scale_factor,
                                                    background=background,
                                                    file_name=file_name)


class BinnedPoissonBackgroundSpectrum(BinnedBackgroundSpectrum):
    def __init__(self, counts, exposure, ebounds, sys_errors=None, response=None,
                 ancillary_file=None, telescope=None, instrument=None, quality=None, file_name=None):
        """
                  A general binned histogram of either Poisson background rates. While the input is in counts, 3ML spectra work
                  in rates, so this class uses the exposure to construct the rates from the counts. While it is possible to
                  construct a histogram directly, this class provides methods to construct histograms from PHA and ROOT (not yet!)
                  files directly.

                  This is a subclass of the BinnedSpectrum class that automates setting the keywords for a Poisson spectrum that is
                  background



                  :param counts: an array of counts
                  :param exposure: the exposure for the counts
                  :param ebounds: the len(counts) + 1 energy edges of the histogram
                  :param sys_errors: (options) systematic error per channel
                  :param response: (optional) instance of InstrumentResponse
                  :param ancillary_file: (optional) ancillary file
                  :param telescope: (optional) telescope name
                  :param instrument: (optional) instrument name
                  :param quality: (optional) Quality object to specify quality per channel
                  :param file_name: (optional) file name associated to the spectrum
                  """

        super(BinnedPoissonSpectrum, self).__init__(self,
                                                    counts,
                                                    exposure,
                                                    ebounds,
                                                    is_poisson=True,
                                                    count_errors=None,
                                                    sys_errors=sys_errors,
                                                    response=response,
                                                    ancillary_file=ancillary_file,
                                                    telescope=telescope,
                                                    instrument=instrument,
                                                    quality=quality,
                                                    scale_factor=None,
                                                    file_name=file_name)
