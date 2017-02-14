import numpy as np
import os
import warnings


from threeML.plugins.spectrum.binned_spectrum import BinnedSpectrumWithDispersion, Quality
from threeML.plugins.OGIP.pha import PHAII
from threeML.plugins.OGIP.response import OGIPResponse, InstrumentResponse

_required_keywords = {}
_required_keywords['observed'] = ("mission:TELESCOP,instrument:INSTRUME,filter:FILTER," +
                                  "exposure:EXPOSURE,backfile:BACKFILE," +
                                  "respfile:RESPFILE," +
                                  "ancrfile:ANCRFILE,hduclass:HDUCLASS," +
                                  "hduclas1:HDUCLAS1,poisserr:POISSERR," +
                                  "chantype:CHANTYPE,detchans:DETCHANS,"
                                  "backscal:BACKSCAL").split(",")

# hduvers:HDUVERS

_required_keywords['background'] = ("mission:TELESCOP,instrument:INSTRUME,filter:FILTER," +
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

class PHASpectrum(BinnedSpectrumWithDispersion):

    def __init__(self, pha_file_or_instance, spectrum_number=None, file_type='observed',rsp_file=None, arf_file=None):
        """
        A spectrum with dispersion build from an OGIP-compliant PHA FITS file. Both Type I & II files can be read. Type II
        spectra are selected either by specifying the spectrum_number or via the {spectrum_number} file name convention used
        in XSPEC. If the file_type is background, a 3ML InstrumentResponse or subclass must be passed so that the energy
        bounds can be obtained.


        :param pha_file_or_instance: either a PHA file name or threeML.plugins.OGIP.pha.PHAII instance
        :param spectrum_number: (optional) the spectrum number of the TypeII file to be used
        :param file_type: observed or background
        :param rsp_file: RMF filename or threeML.plugins.OGIP.response.InstrumentResponse instance
        :param arf_file: (optional) and ARF filen ame
        """

        # extract the spectrum number if needed



        assert isinstance(pha_file_or_instance, str) or isinstance(pha_file_or_instance,
                                                                   PHAII), 'Must provide a FITS file name or PHAII instance'



        if isinstance(pha_file_or_instance, str):

            ext = os.path.splitext(pha_file_or_instance)[-1]

            if '{' in ext:
                spectrum_number = int(ext.split('{')[-1].replace('}', ''))

                pha_file_or_instance = pha_file_or_instance.split('{')[0]

            # Read the data

            filename = pha_file_or_instance



            # create a FITS_FILE instance

            pha_file_or_instance = PHAII.from_fits_file(pha_file_or_instance)

        # If this is already a FITS_FILE instance,

        elif isinstance(pha_file_or_instance, PHAII):

            # we simply create a dummy filename

            filename = 'pha_instance'


        else:

            raise RuntimeError('This is a bug')

        self._file_name = filename

        assert file_type.lower() in ['observed', 'background'], "Unrecognized filetype keyword value"

        file_type = file_type.lower()

        try:

            HDUidx = pha_file_or_instance.index_of("SPECTRUM")

        except:

            raise RuntimeError("The input file %s is not in PHA format" % (pha_file_or_instance))

        #spectrum_number = spectrum_number

        spectrum = pha_file_or_instance[HDUidx]


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

                    gathered_keywords['poisserr'] = False

                elif keyname == "ANCRFILE":

                    # Some non-compliant files have no ARF because they don't need one. Don't fail, but issue a
                    # warning

                    warnings.warn("ANCRFILE is not set. This is not a compliant OGIP file. Assuming no ARF.")

                    gathered_keywords['ancrfile'] = None

                else:

                    raise RuntimeError("Keyword %s not found. File %s is not a proper PHA "
                                       "file" % (keyname, filename))


        is_poisson = gathered_keywords['poisserr']

        exposure = gathered_keywords['exposure']

        # now we need to get the response file so that we can extract the EBOUNDS

        if file_type == 'observed':

            if rsp_file is None:

                # this means it should be specified in the header
                rsp_file = gathered_keywords['respfile']

                if arf_file is  None:

                    arf_file = gathered_keywords['ancrfile']

                    # Read in the response

            if isinstance(rsp_file, str) or isinstance(rsp_file, unicode):
                rsp = OGIPResponse(rsp_file, arf_file=arf_file)

            else:

                # assume a fully formed OGIPResponse
                rsp = rsp_file




        if file_type == 'background':

            # we need the rsp ebounds from response to build the histogram

            assert isinstance(rsp_file,InstrumentResponse), 'You must supply and OGIPResponse to extract the energy bounds'

            rsp = rsp_file


        # Now get the data (counts or rates) and their errors. If counts, transform them in rates

        if typeII:

            # PHA II file
            if has_rates:

                rates = data.field(data_column_name)[spectrum_number - 1, :]

                if not is_poisson:
                    rate_errors = data.field("STAT_ERR")[spectrum_number - 1, :]

            else:

                rates = data.field(data_column_name)[spectrum_number - 1, :] / exposure

                if not is_poisson:
                    rate_errors = data.field("STAT_ERR")[spectrum_number - 1, :] / exposure

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

                rates = data.field(data_column_name)

                if not is_poisson:
                    rate_errors = data.field("STAT_ERR")

            else:

                rates = data.field(data_column_name) / exposure

                if not is_poisson:
                    rate_errors = data.field("STAT_ERR") / exposure

            if "SYS_ERR" in data.columns.names:

                sys_errors = data.field("SYS_ERR")

            else:

                sys_errors = np.zeros(rates.shape)

            if has_quality_column:

                quality = data.field("QUALITY")

            else:

                if is_all_data_good:

                    quality = np.zeros_like(rates, dtype=int)

                else:

                    quality = np.zeros_like(rates, dtype=int) + 5

                    # Now that we have read it, some safety checks

            assert rates.shape[0] == gathered_keywords['detchans'], \
                "The data column (RATES or COUNTS) has a different number of entries than the " \
                "DETCHANS declared in the header"


        quality = Quality.from_ogip(quality)

        counts = rates*exposure

        if not is_poisson:

            count_errors = rate_errors * exposure

        else:

            count_errors = None

        # default the grouping to all open bins
        # this will only be altered if the spectrum is rebinned
        self._grouping = np.ones_like(counts)

        # this saves the extra properties to the class

        self._gathered_keywords = gathered_keywords

        self._file_type = file_type

        # pass the needed spectrum values back up
        # remember that Spectrum reads counts, but returns
        # rates!


        super(PHASpectrum, self).__init__(counts=counts,
                                          exposure=exposure,
                                          response=rsp,
                                          count_errors=count_errors,
                                          sys_errors=sys_errors,
                                          is_poisson=is_poisson,
                                          quality=quality,
                                          mission=gathered_keywords['mission'],
                                          instrument=gathered_keywords['instrument'])

    def _return_file(self, key):

        if key in self._gathered_keywords:

            return self._gathered_keywords[key]

        else:

            return None


    def set_ogip_grouping(self,grouping):
        """
        If the counts are rebinned, this updates the grouping
        :param grouping:

        """

        self._grouping = grouping

    @property
    def filename(self):

        return self._file_name

    @property
    def background_file(self):
        """
        Returns the background file definied in the header, or None if there is none defined
p
        :return: a path to a file, or None
        """

        return self._return_file('backfile')

    @property
    def scale_factor(self):
        """
        This is a scale factor (in the BACKSCAL keyword) which must be used to rescale background and source
        regions

        :return:
        """
        return self._gathered_keywords['backscal']

    @property
    def response_file(self):
        """
            Returns the response file definied in the header, or None if there is none defined

            :return: a path to a file, or None
            """
        return self._return_file('respfile')

    @property
    def ancillary_file(self):
        """
            Returns the ancillary file definied in the header, or None if there is none defined

            :return: a path to a file, or None
            """
        return self._return_file('ancrfile')

    @property
    def grouping(self):

        return self._grouping

    def clone(self, new_counts=None, new_count_errors=None, ):
        """
        make a new spectrum with new counts and errors and all other
        parameters the same


        :param new_counts: new counts for the spectrum
        :param new_count_errors: new errors from the spectrum
        :return: new pha spectrum
        """

        if new_counts is None:
            new_counts = self.counts
            new_count_errors = self.count_errors


        if new_count_errors is None:
            stat_err = None

        else:

            stat_err = new_count_errors/self.exposure

        # create a new PHAII instance

        pha = PHAII(instrument_name=self.instrument,
                    telescope_name=self.mission,
                    tstart=0,
                    telapse=self.exposure,
                    channel=range(1,len(self)+1),
                    rate=new_counts/self.exposure,
                    stat_err=stat_err,
                    quality=self.quality.to_ogip(),
                    grouping=self.grouping,
                    exposure=self.exposure,
                    backscale=self.scale_factor,
                    respfile=None,
                    ancrfile=None,
                    is_poisson=self.is_poisson)


        return pha

    @classmethod
    def from_dispersion_spectrum(cls, dispersion_spectrum, file_type='observed'):
        # type: (BinnedSpectrumWithDispersion, str) -> PHASpectrum



        if dispersion_spectrum.is_poisson:

            rate_errors = None

        else:

            rate_errors = dispersion_spectrum.rate_errors

        pha = PHAII(instrument_name=dispersion_spectrum.instrument,
                    telescope_name=dispersion_spectrum.mission,
                    tstart=0,  # TODO: add this in so that we have proper time!
                    telapse=dispersion_spectrum.exposure,
                    channel=range(1, len(dispersion_spectrum) + 1),
                    rate=dispersion_spectrum.rates,
                    stat_err=rate_errors,
                    quality=dispersion_spectrum.quality.to_ogip(),
                    grouping=np.ones(len(dispersion_spectrum)),
                    exposure=dispersion_spectrum.exposure,
                    backscale=dispersion_spectrum.scale_factor,
                    respfile=None,
                    ancrfile=None,
                    is_poisson=dispersion_spectrum.is_poisson)

        return cls(pha_file_or_instance=pha, spectrum_number=1, file_type=file_type,
                   rsp_file=dispersion_spectrum.response)
