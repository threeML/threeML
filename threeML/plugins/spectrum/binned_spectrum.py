import numpy as np
import astropy.io.fits as fits
import os
import warnings



class BinnedSpectrum(object):
    def __init__(self, counts, exposure, ebounds, is_poisson=False, count_errors=None, sys_errors=None, response_file=None,
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
        :param response_file: (optional) response file
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

        self._response = response_file

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
    def from_fits_file(cls, phafile, spectrum_number,file_type='observed'):

        _required_keywords = {}
        _required_keywords['observed'] = ("mission:TELESCOP,instrument:INSTRUME,filter:FILTER," +
                                          "exposure:EXPOSURE,backfile:BACKFILE," +
                                          "respfile:RESPFILE," +
                                          "ancrfile:ANCRFILE,hduclass:HDUCLASS," +
                                          "hduclas1:HDUCLAS1,poisserr:POISSERR," +
                                          "chantype:CHANTYPE,detchans:DETCHANS,"
                                          "backscal:BACKSCAL").split(",")

        # hduvers:HDUVERS

        _required_keywords['background'] = ("telescope:TELESCOP,instrument:INSTRUME,filter:FILTER," +
                                            "exposure:EXPOSURE," +
                                            "hduclass:HDUCLASS," +
                                            "hduclas1:HDUCLAS1,poisserr:POISSERR," +
                                            "chantype:CHANTYPE,detchans:DETCHANS,"
                                            "backscal:BACKSCAL").split(",")

        # hduvers:HDUVERS

        _might_be_columns = {}
        _might_be_columns['observed'] = ("EXPOSURE,BACKFILE," +
                                         "CORRFILE,CORRSCAL," +
                                         "RESPFILE,ANCRFILE,"
                                         "BACKSCAL").split(",")
        _might_be_columns['background'] = ("EXPOSURE,BACKSCAL").split(",")

        assert file_type.lower() in ['observed', 'background'], "Unrecognized filetype keyword value"

        file_type = file_type.lower()

        # Allow the use of a syntax like "mySpectrum.pha{1}" to specify the spectrum
        # number in PHA II files

        ext = os.path.splitext(phafile)[-1]

        if '{' in ext:
            spectrum_number = int(ext.split('{')[-1].replace('}', ''))

            phafile = phafile.split('{')[0]

        # Read the data

        with fits.open(phafile) as f:

            try:

                HDUidx = f.index_of("SPECTRUM")

            except:

                raise RuntimeError("The input file %s is not in PHA format" % (phafile))



            spectrum = f[HDUidx]
            data = spectrum.data
            header = spectrum.header

            # We don't support yet the rescaling

            if "CORRFILE" in header:

                if header.get("CORRFILE").upper().strip() != "NONE":
                    raise RuntimeError("CORRFILE is not yet supported")

            # See if there is there is a QUALITY==0 in the header

            if "QUALITY" in header:

                has_quality_column = False

                if header["QUALITY"] == 0:

                    is_all_data_good = True

                else:

                    is_all_data_good = False


            else:

                if "QUALITY" in data.columns.names:

                    has_quality_column = True

                    is_all_data_good = False

                else:

                    has_quality_column = False

                    is_all_data_good = True

                    warnings.warn(
                        'Could not find QUALITY in columns or header of PHA file. This is not a valid OGIP file. Assuming QUALITY =0 (good)')

            # Determine if this file contains COUNTS or RATES

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

                typeII = True

                if spectrum_number == None:
                    raise RuntimeError("This is a PHA Type II file. You have to provide a spectrum number")

            else:

                typeII = False

            # Collect information from mandatory keywords

            keys = _required_keywords[file_type]

            gathered_keywords = {}

            for k in keys:

                internal_name, keyname = k.split(":")

                key_has_been_collected = False

                if keyname in header:
                    gathered_keywords[internal_name] = header.get(keyname)

                    # Fix "NONE" in None
                    if gathered_keywords[internal_name] == "NONE" or \
                                    gathered_keywords[internal_name] == 'none':
                        gathered_keywords[internal_name] = None

                    key_has_been_collected = True

                # Note that we check again because the content of the column can override the content of the header

                if keyname in _might_be_columns[file_type] and typeII:

                    # Check if there is a column with this name

                    if keyname in data.columns.names:
                        # This will set the exposure, among other things

                        gathered_keywords[internal_name] = data[keyname][spectrum_number - 1]

                        # Fix "NONE" in None
                        if gathered_keywords[internal_name] == "NONE" or \
                                        gathered_keywords[internal_name] == 'none':
                            gathered_keywords[internal_name] = None

                        key_has_been_collected = True

                if not key_has_been_collected:

                    # The keyword POISSERR is a special case, because even if it is missing,
                    # it is assumed to be False if there is a STAT_ERR column in the file

                    if keyname == "POISSERR" and "STAT_ERR" in data.columns.names:

                        warnings.warn("POISSERR is not set. Assuming non-poisson errors as given in the "
                                      "STAT_ERR column")

                        is_poisson = False

                    elif keyname == "ANCRFILE":

                        # Some non-compliant files have no ARF because they don't need one. Don't fail, but issue a
                        # warning

                        warnings.warn("ANCRFILE is not set. This is not a compliant OGIP file. Assuming no ARF.")

                        gathered_keywords['ancrfile'] = None

                    else:

                        raise RuntimeError("Keyword %s not found. File %s is not a proper PHA "
                                           "file" % (keyname, phafile))

            # Now get the data (counts or rates) and their errors. If counts, transform them in rates

            exposure = gathered_keywords['exposure']
            response_file = gathered_keywords['response_file']
            ancillary_file = gathered_keywords['ancrfile']
            telescope = gathered_keywords['telescope']
            instrument = gathered_keywords['instrument']
            scale_factor = gathered_keywords['scale_factor']
            is_poisson - gathered_keywords['poisserr']


            if typeII:

                # PHA II file
                if has_rates:

                    counts = data.field(data_column_name)[spectrum_number - 1, :] * exposure

                    if not is_poisson:

                        count_errors = data.field("STAT_ERR")[spectrum_number - 1, :] * exposure

                else:

                    rates = data.field(data_column_name)[spectrum_number - 1, :]

                    if not is_poisson:
                        count_errors = data.field("STAT_ERR")[spectrum_number - 1, :]

                if "SYS_ERR" in data.columns.names:

                    sys_errors = data.field("SYS_ERR")[spectrum_number - 1, :]
                else:

                    sys_errors = np.zeros(rates.shape)

                if has_quality_column:

                    quality = data.field("QUALITY")[spectrum_number - 1, :]

                else:

                    if is_all_data_good:

                        quality = np.zeros_like(rates, dtype=int)

                    else:

                        quality = np.zeros_like(rates, dtype=int) + 5



            elif typeII == False:

                # PHA 1 file
                if has_rates:

                    counts = data.field(data_column_name) * exposure

                    if not is_poisson:

                        count_errors = data.field("STAT_ERR") * exposure

                else:

                    counts = data.field(data_column_name)

                    if not is_poisson:

                        count_errors = data.field("STAT_ERR")

                if "SYS_ERR" in data.columns.names:

                    sys_errors = data.field("SYS_ERR")

                else:

                    sys_errors = np.zeros(counts.shape)

                if has_quality_column:

                    quality = data.field("QUALITY")

                else:

                    if is_all_data_good:

                        quality = np.zeros_like(counts, dtype=int)

                    else:

                        quality = np.zeros_like(counts, dtype=int) + 5

                        # Now that we have read it, some safety checks



        if 'backfile' in gathered_keywords:

            background = BinnedSpectrum.from_fits_file(gathered_keywords['backfile'],
                                                       spectrum_number,
                                                       file_type='background')

        else:

            background = None



        # Now that we have read it, some safety checks

        quality = Quality.from_ogip(quality)

        assert rates.shape[0] == gathered_keywords['detchans'], \
            "The data column (RATES or COUNTS) has a different number of entries than the " \
            "DETCHANS declared in the header"

        if file_type == 'observed':

            if is_poisson:

                return BinnedPoissonSpectrum(counts,
                                             exposure,
                                             count_errors,
                                             sys_errors,
                                             response_file,
                                             ancillary_file,
                                             telescope,
                                             instrument,
                                             quality,
                                             scale_factor,
                                             background,
                                             phafile
                                             )

            else:

                return BinnedSpectrum(counts,
                                      exposure,
                                      sys_errors,
                                      response_file,
                                      ancillary_file,
                                      telescope,
                                      instrument,
                                      quality,
                                      scale_factor,
                                      background,
                                      phafile)

        else:

            if is_poisson:

                return BinnedPoissonBackgroundSpectrum(counts,
                                                       exposure,
                                                       sys_errors,
                                                       response_file,
                                                       ancillary_file,
                                                       telescope,
                                                       instrument,
                                                       quality,
                                                       scale_factor,
                                                       phafile)

            else:

                return BinnedBackgroundSpectrum(counts,
                                                exposure,
                                                count_errors,
                                                sys_errors,
                                                response_file,
                                                ancillary_file,
                                                telescope,
                                                instrument,
                                                quality,
                                                scale_factor,
                                                phafile)





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
    def telescope(self):
        """
        Returns the name of the mission used to make the observation
        :return: a string
        """

        if self._telescope is None:
            return 'UNKNOWN'

        return self._telescope

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
                                                       response_file=None,
                                                       ancillary_file=ancillary_file,
                                                       telescope=telescope,
                                                       instrument=instrument,
                                                       quality=quality,
                                                       scale_factor=None,
                                                       background=None,
                                                       file_name=file_name)


class BinnedPoissonSpectrum(BinnedSpectrum):
    def __init__(self, counts, exposure, ebounds, sys_errors=None, response_file=None,
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
           :param response_file: (optional) instance of InstrumentResponse
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
                                                    response_file=response_file,
                                                    ancillary_file=ancillary_file,
                                                    telescope=telescope,
                                                    instrument=instrument,
                                                    quality=quality,
                                                    scale_factor=scale_factor,
                                                    background=background,
                                                    file_name=file_name)


class BinnedPoissonBackgroundSpectrum(BinnedBackgroundSpectrum):
    def __init__(self, counts, exposure, ebounds, sys_errors=None, response_file=None,
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
                  :param response_file: (optional) instance of InstrumentResponse
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
                                                    response=response_file,
                                                    ancillary_file=ancillary_file,
                                                    telescope=telescope,
                                                    instrument=instrument,
                                                    quality=quality,
                                                    scale_factor=None,
                                                    file_name=file_name)


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
