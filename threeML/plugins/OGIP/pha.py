import os
import astropy.io.fits as fits
import numpy as np
import warnings
from collections import MutableMapping
from copy import copy
from threeML.plugins.OGIP.response import EBOUNDS, SPECRESP_MATRIX
from threeML.io.fits_file import FITSExtension, FITSFile
import astropy.units as u

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


class PHA(object):
    """
    Represents a PHA spectrum from the OGIP format. While this class will always represents only one spectrum, it can
    be instances with both a Type I and Type II file.

    Despite the input (counts or rates), this will always have data expressed in rates.
    """

    def __init__(self, phafile, spectrum_number=None, file_type='observed', ):

        if isinstance(phafile, str):

            ext = os.path.splitext(phafile)[-1]

            if '{' in ext:
                spectrum_number = int(ext.split('{')[-1].replace('}', ''))

                phafile = phafile.split('{')[0]

            # Read the data

            filename = phafile

            # create a FITS_FILE instance

            phafile = PHAII.from_fits_file(phafile)


        elif isinstance(phafile, PHAII):

            # we simply create a dummy filename

            filename = 'pha_instance'



        spectrum = self._extract_pha_information(phafile, spectrum_number, file_type, filename)



        data = spectrum.data
        header = spectrum.header

        # We don't support yet the rescaling

        if "CORRFILE" in header:

            if header.get("CORRFILE").upper().strip() != "NONE":
                raise RuntimeError("CORRFILE is not yet supported")

        # See if there is there is a QUALITY==0 in the header

        if "QUALITY" in header:

            self._has_quality_column = False

            if header["QUALITY"] == 0:

                self._is_all_data_good = True

            else:

                self._is_all_data_good = False


        else:

            if "QUALITY" in data.columns.names:

                self._has_quality_column = True

                self._is_all_data_good = False

            else:

                self._has_quality_column = False

                self._is_all_data_good = True

                warnings.warn(
                    'Could not find QUALITY in columns or header of PHA file. This is not a valid OGIP file. Assuming QUALITY =0 (good)')

        # Determine if this file contains COUNTS or RATES

        if "COUNTS" in data.columns.names:

            self._has_rates = False
            self._data_column_name = "COUNTS"

        elif "RATE" in data.columns.names:

            self._has_rates = True
            self._data_column_name = "RATE"

        else:

            raise RuntimeError("This file does not contain a RATE nor a COUNTS column. "
                               "This is not a valid PHA file")

        # Determine if this is a PHA I or PHA II
        if len(data.field(self._data_column_name).shape) == 2:

            self._typeII = True

            if self._spectrum_number == None:
                raise RuntimeError("This is a PHA Type II file. You have to provide a spectrum number")

        else:

            self._typeII = False

        # Collect information from mandatory keywords

        keys = _required_keywords[self._file_type]

        self._gathered_keywords = {}

        for k in keys:

            internal_name, keyname = k.split(":")

            key_has_been_collected = False

            if keyname in header:
                self._gathered_keywords[internal_name] = header.get(keyname)

                # Fix "NONE" in None
                if self._gathered_keywords[internal_name] == "NONE" or \
                                self._gathered_keywords[internal_name] == 'none':
                    self._gathered_keywords[internal_name] = None

                key_has_been_collected = True

            # Note that we check again because the content of the column can override the content of the header

            if keyname in _might_be_columns[self._file_type] and self._typeII:

                # Check if there is a column with this name

                if keyname in data.columns.names:
                    # This will set the exposure, among other things

                    self._gathered_keywords[internal_name] = data[keyname][self._spectrum_number - 1]

                    # Fix "NONE" in None
                    if self._gathered_keywords[internal_name] == "NONE" or \
                                    self._gathered_keywords[internal_name] == 'none':
                        self._gathered_keywords[internal_name] = None

                    key_has_been_collected = True

            if not key_has_been_collected:

                # The keyword POISSERR is a special case, because even if it is missing,
                # it is assumed to be False if there is a STAT_ERR column in the file

                if keyname == "POISSERR" and "STAT_ERR" in data.columns.names:

                    warnings.warn("POISSERR is not set. Assuming non-poisson errors as given in the "
                                  "STAT_ERR column")

                    self._gathered_keywords['poisserr'] = False

                elif keyname == "ANCRFILE":

                    # Some non-compliant files have no ARF because they don't need one. Don't fail, but issue a
                    # warning

                    warnings.warn("ANCRFILE is not set. This is not a compliant OGIP file. Assuming no ARF.")

                    self._gathered_keywords['ancrfile'] = None

                else:

                    raise RuntimeError("Keyword %s not found. File %s is not a proper PHA "
                                       "file" % (keyname, filename))

        # Now get the data (counts or rates) and their errors. If counts, transform them in rates

        if self._typeII:

            # PHA II file
            if self._has_rates:

                self._rates = data.field(self._data_column_name)[self._spectrum_number - 1, :]

                if not self.is_poisson():
                    self._rate_errors = data.field("STAT_ERR")[self._spectrum_number - 1, :]

            else:

                self._rates = data.field(self._data_column_name)[self._spectrum_number - 1, :] / self.exposure

                if not self.is_poisson():
                    self._rate_errors = data.field("STAT_ERR")[self._spectrum_number - 1, :] / self.exposure

            if "SYS_ERR" in data.columns.names:

                self._sys_errors = data.field("SYS_ERR")[self._spectrum_number - 1, :]
            else:

                self._sys_errors = np.zeros(self._rates.shape)

            if self._has_quality_column:

                self._quality = data.field("QUALITY")[self._spectrum_number - 1, :]

            else:

                if self._is_all_data_good:

                    self._quality = np.zeros_like(self._rates, dtype=int)

                else:

                    self._quality = np.zeros_like(self._rates, dtype=int) + 5



        elif self._typeII == False:

            # PHA 1 file
            if self._has_rates:

                self._rates = data.field(self._data_column_name)

                if not self.is_poisson():
                    self._rate_errors = data.field("STAT_ERR")

            else:

                self._rates = data.field(self._data_column_name) / self.exposure

                if not self.is_poisson():
                    self._rate_errors = data.field("STAT_ERR") / self.exposure

            if "SYS_ERR" in data.columns.names:

                self._sys_errors = data.field("SYS_ERR")

            else:

                self._sys_errors = np.zeros(self._rates.shape)

            if self._has_quality_column:

                self._quality = data.field("QUALITY")

            else:

                if self._is_all_data_good:

                    self._quality = np.zeros_like(self._rates, dtype=int)

                else:

                    self._quality = np.zeros_like(self._rates, dtype=int) + 5

                    # Now that we have read it, some safety checks

            assert self._rates.shape[0] == self._gathered_keywords['detchans'], \
                "The data column (RATES or COUNTS) has a different number of entries than the " \
                "DETCHANS declared in the header"


    def _extract_pha_information(self,pha_instance, spectrum_number, file_type, phafile = None):

        assert file_type.lower() in ['observed', 'background'], "Unrecognized filetype keyword value"

        self._file_type = file_type.lower()

        try:

            HDUidx = pha_instance.index_of("SPECTRUM")

        except:

            raise RuntimeError("The input file %s is not in PHA format" % (phafile))

        self._spectrum_number = spectrum_number

        spectrum = pha_instance[HDUidx]

        return spectrum




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

        return self._gathered_keywords['detchans']

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
        return self._gathered_keywords['exposure']

    def _return_file(self, key):

        if key in self._gathered_keywords:

            return self._gathered_keywords[key]

        else:

            return None

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
    def mission(self):
        """
        Returns the name of the mission used to make the observation
        :return: a string
        """

        if self._gathered_keywords['mission'] is None:

            return 'UNKNOWN'

        return self._gathered_keywords['mission']

    @property
    def instrument(self):
        """
        Returns the name of the instrument used to make the observation
        :return: a string
        """

        if self._gathered_keywords['instrument'] is None:

            return 'UNKNOWN'

        return self._gathered_keywords['instrument']

    def is_poisson(self):
        """
        Returns whether the spectrum has Poisson errors or not

        :return: True or False
        """

        return self._gathered_keywords['poisserr']

    @property
    def quality(self):
        """
        Return the native quality of the PHA file
        :return:
        """

        return self._quality


class PHAWrite(object):
    def __init__(self, *ogiplike):
        """
        This class handles writing of PHA files from OGIPLike style plugins. It takes an arbitrary number of plugins as
        input. While OGIPLike provides a write_pha method, it is only for writing the given instance to disk. The class
         in general can be used to save an entire series of OGIPLikes to PHAs which can be used for time-resolved style
         plugins. An example implentation is given in FermiGBMTTELike.


        :param ogiplike: OGIPLike plugin(s) to be written to disk
        """

        self._ogiplike = ogiplike

        self._n_spectra = len(ogiplike)

        # The following lists corresponds to the different columns in the PHA/CSPEC
        # formats, and they will be filled up by addSpectrum()

        self._tstart = {'pha': [], 'bak': []}
        self._tstop = {'pha': [], 'bak': []}
        self._channel = {'pha': [], 'bak': []}
        self._rate = {'pha': [], 'bak': []}
        self._stat_err = {'pha': [], 'bak': []}
        self._sys_err = {'pha': [], 'bak': []}
        self._backscal = {'pha': [], 'bak': []}
        self._quality = {'pha': [], 'bak': []}
        self._grouping = {'pha': [], 'bak': []}
        self._exposure = {'pha': [], 'bak': []}
        self._backfile = {'pha': [], 'bak': []}
        self._respfile = {'pha': [], 'bak': []}
        self._ancrfile = {'pha': [], 'bak': []}
        self._mission = {'pha': [], 'bak': []}
        self._instrument = {'pha': [], 'bak': []}

        # If the PHAs have existing background files
        # then it is assumed that we will not need to write them
        # out. THe most likely case is that the background file does not
        # exist i.e. these are simulations are from EventList object
        # Just one instance of no background file existing cause the write
        self._write_bak_file = False

        # Assuming all entries will have one answer
        self._is_poisson = {'pha': True, 'bak': True}

        self._pseudo_time = 0.

        self._spec_iterator = 1

    def write(self, outfile_name, overwrite=True):
        """
        Write a PHA Type II and BAK file for the given OGIP plugin. Automatically determines
        if BAK files should be generated.


        :param outfile_name: string (excluding .pha) of the PHA to write
        :param overwrite: (optional) bool to overwrite existing file
        :return:
        """

        # Remove the .pha extension if any
        if os.path.splitext(outfile_name)[-1].lower() == '.pha':

            outfile_name = os.path.splitext(outfile_name)[0]

        self._outfile_basename = outfile_name

        self._outfile_name = {'pha': '%s.pha' % outfile_name, 'bak': '%s_bak.pha' % outfile_name}

        for ogip in self._ogiplike:

            self._append_ogip(ogip)

        self._write_phaII(overwrite)

    def _append_ogip(self, ogip):
        """
        Add an ogip instance's data into the data list

        :param ogip: and OGIPLike instance
        :return: None
        """

        # grab the ogip pha info
        pha_info = ogip.get_pha_files()

        self._out_rsp = []

        first_channel = pha_info['rsp'].first_channel

        for key in ['pha', 'bak']:

            if key == 'pha':

                if pha_info[key].background_file is not None:

                    self._backfile[key].append(pha_info[key].background_file)

                else:

                    self._backfile[key].append('%s_bak.pha{%d}' % (self._outfile_basename, self._spec_iterator))

                    # We want to write the bak file

                    self._write_bak_file = True

            if pha_info[key].ancillary_file is not None:

                self._ancrfile[key].append(pha_info[key].ancillary_file)

            else:

                # There is no ancillary file, so we need to flag it.

                self._ancrfile[key].append('NONE')


            if pha_info['rsp'].rsp_filename is not None:

                self._respfile[key].append(pha_info['rsp'].rsp_filename)

            else:

                # This will be reached in the case that a response was generated from a plugin
                # e.g. if we want to use weighted DRMs from GBM.

                rsp_file_name = "%s.rsp{%d}"%(self._outfile_basename,self._spec_iterator)

                self._respfile[key].append(rsp_file_name)

                if key == 'pha':

                        self._out_rsp.append(pha_info['rsp'])


            self._rate[key].append(pha_info[key].rates.tolist())

            self._backscal[key].append(pha_info[key].scale_factor)

            if not pha_info[key].is_poisson():

                self._is_poisson[key] = pha_info[key].is_poisson()

                self._stat_err[key].append(pha_info[key].rate_errors.tolist())

            # If there is systematic error, we add it
            # otherwise create an array of zeros as XSPEC
            # simply adds systematic in quadrature to statistical
            # error.

            if pha_info[key].sys_errors.tolist() is not None:  # It returns an array which does not work!

                self._sys_err[key].append(pha_info[key].sys_errors.tolist())

            else:

                self._sys_err[key].append(np.zeros_like(pha_info[key].rates, dtype=np.float32).tolist())

            self._exposure[key].append(pha_info[key].exposure)
            self._quality[key].append(ogip.ogip_quality.tolist())
            self._grouping[key].append(ogip.ogip_grouping.tolist())
            self._channel[key].append(np.arange(pha_info[key].n_channels, dtype=np.int32) + first_channel)
            self._instrument[key] = pha_info[key].instrument
            self._mission[key] = pha_info[key].mission

            if ogip.tstart is not None:

                self._tstart[key].append(ogip.tstart)

                if ogip.tstop is not None:

                    self._tstop[key].append(ogip.tstop)

                else:

                    RuntimeError('OGIP TSTART is a number but TSTOP is None. This is a bug.')

            # We will assume that the exposure is the true DT
            # and assign starts and stops accordingly. This means
            # we are most likely are dealing with a simulation.
            else:

                self._tstart[key].append(self._pseudo_time)

                self._pseudo_time += pha_info[key].exposure

                self._tstop[key].append(self._pseudo_time)

        self._spec_iterator += 1



    def _write_phaII(self, overwrite):

        # Fix this later... if needed.
        trigger_time = None

        # Assuming background and pha files have the same
        # number of channels


        assert len(self._rate['pha'][0]) == len(
                self._rate['bak'][0]), "PHA and BAK files do not have the same number of channels. Something is wrong."

        assert self._instrument['pha'] == self._instrument[
            'bak'], "Instrument for PHA and BAK (%s,%s) are not the same. Something is wrong with the files. " % (
            self._instrument['pha'], self._instrument['bak'])

        assert self._mission['pha'] == self._mission[
            'bak'], "Mission for PHA and BAK (%s,%s) are not the same. Something is wrong with the files. " % (
            self._mission['pha'], self._mission['bak'])



        if self._write_bak_file:

            keys = ['pha', 'bak']

        else:

            keys = ['pha']

        for key in keys:

            if trigger_time is not None:

                tstart = self._tstart[key] - trigger_time

            else:

                tstart = self._tstart[key]

            if not self._is_poisson[key]:

                if key == 'pha':

                    fits_file = PHAII(self._instrument[key],
                                      self._mission[key],
                                      tstart,
                                      np.array(self._tstop[key]) - np.array(self._tstart[key]),
                                      self._channel[key],
                                      self._rate[key],
                                      self._quality[key],
                                      self._grouping[key],
                                      self._exposure[key],
                                      self._backscal[key],
                                      self._respfile[key],
                                      self._ancrfile[key],
                                      back_file=self._backfile[key],
                                      sys_err=self._sys_err[key],
                                      stat_err=self._stat_err[key])

                else:

                    fits_file = BAK_PHAII(self._instrument[key],
                                      self._mission[key],
                                      tstart,
                                      np.array(self._tstop[key]) - np.array(self._tstart[key]),
                                      self._channel[key],
                                      self._rate[key],
                                      self._quality[key],
                                      self._grouping[key],
                                      self._exposure[key],
                                      self._backscal[key],
                                      self._respfile[key],
                                      self._ancrfile[key],
                                      self._sys_err[key],
                                      self._stat_err[key])

            else:

                if key == 'pha':

                    fits_file = POISSON_PHAII(self._instrument[key],
                                              self._mission[key],
                                              tstart,
                                              np.array(self._tstop[key]) - np.array(self._tstart[key]),
                                              self._channel[key],
                                              self._rate[key],
                                              self._quality[key],
                                              self._grouping[key],
                                              self._exposure[key],
                                              self._backscal[key],
                                              self._respfile[key],
                                              self._ancrfile[key],
                                              back_file=self._backfile[key])

                else:

                    fits_file = POISSON_BAK_PHAII(self._instrument[key],
                                                  self._mission[key],
                                                  tstart,
                                                  np.array(self._tstop[key]) - np.array(self._tstart[key]),
                                                  self._channel[key],
                                                  self._rate[key],
                                                  self._quality[key],
                                                  self._grouping[key],
                                                  self._exposure[key],
                                                  self._backscal[key],
                                                  self._respfile[key],
                                                  self._ancrfile[key])



            fits_file.writeto(self._outfile_name[key], overwrite=overwrite)


            if self._out_rsp:


                extensions = [EBOUNDS(self._out_rsp[0].ebounds)]

                extensions.extend([SPECRESP_MATRIX(this_rsp.monte_carlo_energies, this_rsp.ebounds, this_rsp.matrix) for this_rsp in self._out_rsp])

                rsp2 = FITSFile(fits_extensions=extensions)

                rsp2.writeto("%s.rsp" % self._outfile_basename,overwrite=True)




####################################################################################
# The following classes are used to create OGIP-compliant PHAII files


def _atleast_2d_with_dtype(value,dtype=None):


    if dtype is not None:
        value = np.array(value,dtype=dtype)

    arr = np.atleast_2d(value)

    return arr

def _atleast_1d_with_dtype(value,dtype=None):


    if dtype is not None:
        value = np.array(value,dtype=dtype)

        if dtype == str:

            # convert None to NONE
            # which is needed for None Type args
            # to string arrays

            idx = np.core.defchararray.lower(value) == 'none'

            value[idx] = 'NONE'

    arr = np.atleast_1d(value)

    return arr


class SPECTRUM(FITSExtension):

    _HEADER_KEYWORDS = (('EXTNAME', 'SPECTRUM', 'Extension name'),
                        ('CONTENT', 'OGIP PHA data', 'File content'),
                        ('HDUCLASS', 'OGIP    ', 'format conforms to OGIP standard'),
                        ('HDUVERS', '1.1.0   ', 'Version of format (OGIP memo CAL/GEN/92-002a)'),
                        ('HDUDOC', 'OGIP memos CAL/GEN/92-002 & 92-002a', 'Documents describing the forma'),
                        ('HDUVERS1', '1.0.0   ', 'Obsolete - included for backwards compatibility'),
                        ('HDUVERS2', '1.1.0   ', 'Obsolete - included for backwards compatibility'),
                        ('HDUCLAS1', 'SPECTRUM', 'Extension contains spectral data  '),
                        ('HDUCLAS2', 'TOTAL ', ''),
                        ('HDUCLAS3', 'RATE ', ''),
                        ('HDUCLAS4', 'TYPE:II ', ''),
                        ('FILTER', '', 'Filter used'),
                        ('CHANTYPE', 'PHA', 'Channel type'),
                        ('POISSERR', False, 'Are the rates Poisson distributed'),
                        ('DETCHANS', None, 'Number of channels'),
                        ('CORRSCAL',1.0,''),
                        ('AREASCAL',1.0,'')



                        )


    def __init__(self, tstart, telapse, channel, rate, quality, grouping, exposure, backscale, respfile,
                 ancrfile, back_file=None, sys_err=None, stat_err=None):

        """
        Represents the SPECTRUM extension of a PHAII file.

        :param tstart: array of interval start times
        :param telapse: array of times elapsed since start
        :param channel: arrary of channel numbers
        :param rate: array of rates
        :param quality: array of OGIP quality values
        :param grouping: array of OGIP grouping values
        :param exposure: array of exposures
        :param backscale: array of backscale values
        :param respfile: array of associated response file names
        :param ancrfile: array of associate ancillary file names
        :param back_file: array of associated background file names
        :param sys_err: array of optional systematic errors
        :param stat_err: array of optional statistical errors (required of non poisson!)
        """

        n_spectra = len(tstart)

        data_list = [('TSTART', tstart),
                      ('TELAPSE', telapse),
                      ('SPEC_NUM',np.arange(1, n_spectra + 1, dtype=np.int32)),
                      ('CHANNEL', channel),
                      ('RATE',rate),
                      ('QUALITY',quality),
                      ('GROUPING',grouping),
                      ('EXPOSURE',exposure),
                      ('BACKSCAL',backscale),
                      ('RESPFILE',respfile),
                      ('ANCRFILE',ancrfile)]


        if back_file is not None:

            data_list.append(('BACKFILE', back_file))


        if stat_err is not None:

            data_list.append(('STAT_ERR', stat_err))

        if sys_err is not None:

            data_list.append(('SYS_ERR', sys_err))


        super(SPECTRUM, self).__init__(tuple(data_list), self._HEADER_KEYWORDS)

class POISSON_SPECTRUM(SPECTRUM):
    def __init__(self, tstart, telapse, channel, rate, quality, grouping, exposure, backscale, respfile,
                 ancrfile, back_file=None):

        """
        Represents the SPECTRUM extension of a PHAII file when the rates are POISSON
        distributed

        :param tstart: array of interval start times
        :param telapse: array of times elapsed since start
        :param channel: arrary of channel numbers
        :param rate: array of rates
        :param quality: array of OGIP quality values
        :param grouping: array of OGIP grouping values
        :param exposure: array of exposures
        :param backscale: array of backscale values
        :param respfile: array of associated response file names
        :param ancrfile: array of associate ancillary file names
        :param back_file: array of associated background file names
        """

        super(POISSON_SPECTRUM, self).__init__(tstart, telapse, channel, rate, quality, grouping, exposure, backscale, respfile,
                 ancrfile, back_file=back_file)

        self.hdu.header.set("POISSERR", True)

class BAK_SPECTRUM(SPECTRUM):

    def __init__(self, tstart, telapse, channel, rate, quality, grouping, exposure, backscale, respfile,
                 ancrfile, sys_err=None, stat_err=None):
        """
          Represents the banckground SPECTRUM extension of a PHAII file with the rates are not
          POISSON distributed

          :param tstart: array of interval start times
          :param telapse: array of times elapsed since start
          :param channel: arrary of channel numbers
          :param rate: array of rates
          :param quality: array of OGIP quality values
          :param grouping: array of OGIP grouping values
          :param exposure: array of exposures
          :param respfile: array of associated response file names
          :param ancrfile: array of associate ancillary file names
          :param sys_err: array of optional systematic errors
          :param stat_err: array of optional statistical errors (required of non poisson!)
          """



        super(BAK_SPECTRUM, self).__init__(tstart, telapse, channel, rate, quality, grouping, exposure, backscale, respfile,
                                           ancrfile, back_file=None, sys_err=sys_err, stat_err=stat_err)

class POISSON_BKG_SPECTRUM(SPECTRUM):

    def __init__(self, tstart, telapse, channel, rate, quality, grouping, exposure, backscale, respfile,
                 ancrfile):
        """
          Represents the background SPECTRUM extension of a PHAII file when the rates
          are POISSON distributed

          :param tstart: array of interval start times
          :param telapse: array of times elapsed since start
          :param channel: arrary of channel numbers
          :param rate: array of rates
          :param quality: array of OGIP quality values
          :param grouping: array of OGIP grouping values
          :param exposure: array of exposures
          :param respfile: array of associated response file names
          :param ancrfile: array of associate ancillary file names
          """

        super(BAK_SPECTRUM, self).__init__(tstart, telapse, channel, rate, quality, grouping, exposure, backscale,
                                           respfile,
                                           ancrfile)

        self.hdu.header.set("POISSERR", True)

class PHAII(FITSFile):


    def __init__(self, instrument_name, telescope_name, tstart, telapse, channel, rate, quality, grouping, exposure, backscale, respfile,
                 ancrfile, back_file=None, sys_err=None, stat_err=None):


        """

        A generic PHAII fits file

        :param instrument_name: name of the instrument
        :param telescope_name: name of the telescope
        :param tstart: array of interval start times
        :param telapse: array of times elapsed since start
        :param channel: arrary of channel numbers
        :param rate: array of rates
        :param quality: array of OGIP quality values
        :param grouping: array of OGIP grouping values
        :param exposure: array of exposures
        :param backscale: array of backscale values
        :param respfile: array of associated response file names
        :param ancrfile: array of associate ancillary file names
        :param back_file: array of associated background file names
        :param sys_err: array of optional systematic errors
        :param stat_err: array of optional statistical errors (required of non poisson!)
        """

        # collect the data so that we can have a general
        # extension builder


        self._tstart = _atleast_1d_with_dtype(tstart , np.float64) * u.s
        self._telapse = _atleast_1d_with_dtype(telapse, np.float64) * u.s
        self._channel = _atleast_2d_with_dtype(channel, np.int16)
        self._rate = _atleast_2d_with_dtype(rate, np.float64) * 1./u.s
        self._exposure = _atleast_1d_with_dtype(exposure, np.float64) * u.s
        self._quality = _atleast_2d_with_dtype(quality, np.int16)
        self._grouping = _atleast_2d_with_dtype(grouping, np.int16)
        self._backscale = _atleast_1d_with_dtype(backscale, np.float64)
        self._respfile = _atleast_1d_with_dtype(respfile,str)
        self._ancrfile = _atleast_1d_with_dtype(ancrfile,str)


        if sys_err is not None:

            self._sys_err = _atleast_2d_with_dtype(sys_err, np.float64)

        else:

            self._sys_err = sys_err

        if stat_err is not None:

            self._stat_err = _atleast_2d_with_dtype(stat_err,np.float64)

        else:

            self._stat_err = stat_err

        if back_file is not None:

            self._back_file = _atleast_1d_with_dtype(back_file,str)
        else:

            self._back_file = np.array(['NONE'] * self._tstart.shape[0])

        # Create the SPECTRUM extension

        spectrum_extension = self._build_spectrum_extension()

        # Set telescope and instrument name

        spectrum_extension.hdu.header.set("TELESCOP", telescope_name)
        spectrum_extension.hdu.header.set("INSTRUME", instrument_name)
        spectrum_extension.hdu.header.set("DETCHANS", len(self._channel[0]))



        super(PHAII, self).__init__(fits_extensions=[spectrum_extension])




    def _build_spectrum_extension(self):


        # build the extension

        spectrum_extension = SPECTRUM(self._tstart,
                                      self._telapse,
                                      self._channel,
                                      self._rate,
                                      self._quality,
                                      self._grouping,
                                      self._exposure,
                                      self._backscale,
                                      self._respfile,
                                      self._ancrfile,
                                      back_file=self._back_file,
                                      sys_err=self._sys_err,
                                      stat_err=self._stat_err)



        return spectrum_extension

    @classmethod
    def from_event_list(cls, event_list, use_poly=False):

        pha_information = event_list.get_pha_information(use_poly)

        if use_poly:

            return BAK_PHAII(instrument_name=pha_information['instrument'],
                             telescope_name=pha_information['telescope'],
                             tstart=pha_information['tstart'],
                             telapse=pha_information['telapse'],
                             channel=pha_information['channel'],
                             rate=pha_information['rate'],
                             stat_err=pha_information['rate error'],
                             quality=pha_information['quality'],
                             grouping=pha_information['grouping'],
                             exposure=pha_information['exposure'],
                             backscale=None,
                             respfile=pha_information['response_file'],
                             ancrfile=None)

        else:

            return POISSON_PHAII(instrument_name=pha_information['instrument'],
                                 telescope_name=pha_information['telescope'],
                                 tstart=pha_information['tstart'],
                                 telapse=pha_information['telapse'],
                                 channel=pha_information['channel'],
                                 rate=pha_information['rate'],
                                 quality=pha_information['quality'],
                                 grouping=pha_information['grouping'],
                                 exposure=pha_information['exposure'],
                                 back_file=pha_information['backfile'],
                                 backscale=None,
                                 respfile=pha_information['response_file'],
                                 ancrfile=None)
    @classmethod
    def from_fits_file(cls,fits_file):

        with fits.open(fits_file) as f:


            spectrum = FITSExtension.from_fits_file_extension(f['SPECTRUM'])



            out = FITSFile(primary_hdu=f['PRIMARY'], fits_extensions=[spectrum])


        return out




    @property
    def instrument(self):
        return



class POISSON_PHAII(PHAII):


    def __init__(self, instrument_name, telescope_name, tstart, telapse, channel, rate, quality, grouping, exposure, backscale, respfile,
                 ancrfile, back_file=None):


        """

        A PHAII file with POISSON distributed rates

        :param instrument_name: name of the instrument
        :param telescope_name: name of the telescope
        :param tstart: array of interval start times
        :param telapse: array of times elapsed since start
        :param channel: arrary of channel numbers
        :param rate: array of rates
        :param quality: array of OGIP quality values
        :param grouping: array of OGIP grouping values
        :param exposure: array of exposures
        :param backscale: array of backscale values
        :param respfile: array of associated response file names
        :param ancrfile: array of associate ancillary file names
        :param back_file: array of associated background file names
        """

        super(POISSON_PHAII, self).__init__( instrument_name, telescope_name, tstart, telapse, channel, rate, quality, grouping, exposure, backscale, respfile,
                 ancrfile, back_file=back_file)

    def _build_spectrum_extension(self):

        # build the extension

        spectrum_extension = POISSON_SPECTRUM(self._tstart,
                                          self._telapse,
                                          self._channel,
                                          self._rate,
                                          self._quality,
                                          self._grouping,
                                          self._exposure,
                                          self._backscale,
                                          self._respfile,
                                          self._ancrfile,
                                          back_file=self._back_file)

        return spectrum_extension


class BAK_PHAII(PHAII):

    def __init__(self, instrument_name, telescope_name, tstart, telapse, channel, rate, quality, grouping, exposure,
                 backscale, respfile,
                 ancrfile, sys_err=None, stat_err=None):
        """

        A background PHAII file


        :param instrument_name: name of the instrument
        :param telescope_name: name of the telescope
        :param tstart: array of interval start times
        :param telapse: array of times elapsed since start
        :param channel: arrary of channel numbers
        :param rate: array of rates
        :param quality: array of OGIP quality values
        :param grouping: array of OGIP grouping values
        :param exposure: array of exposures
        :param backscale: array of backscale values
        :param respfile: array of associated response file names
        :param ancrfile: array of associate ancillary file names
        :param sys_err:
        :param stat_err:
        """

        super(BAK_PHAII, self).__init__(instrument_name, telescope_name, tstart, telapse, channel, rate, quality, grouping, exposure,
                 backscale, respfile,
                 ancrfile, back_file=None, sys_err=sys_err, stat_err=stat_err)

    def _build_spectrum_extension(self):

        # build the extension


        spectrum_extension = BAK_SPECTRUM(self._tstart,
                                          self._telapse,
                                          self._channel,
                                          self._rate,
                                          self._quality,
                                          self._grouping,
                                          self._exposure,
                                          self._backscale,
                                          self._respfile,
                                          self._ancrfile,
                                          sys_err=self._sys_err,
                                          stat_err=self._stat_err)



        return spectrum_extension

class POISSON_BAK_PHAII(BAK_PHAII):
    def __init__(self, instrument_name, telescope_name, tstart, telapse, channel, rate, quality, grouping, exposure,
                 backscale, respfile, ancrfile):
        """
        A background PHAII file with POISSON distributed rates

        :param instrument_name: name of the instrument
        :param telescope_name: name of the telescope
        :param tstart: array of interval start times
        :param telapse: array of times elapsed since start
        :param channel: arrary of channel numbers
        :param rate: array of rates
        :param quality: array of OGIP quality values
        :param grouping: array of OGIP grouping values
        :param exposure: array of exposures
        :param backscale: array of backscale values
        :param respfile: array of associated response file names
        :param ancrfile: array of associate ancillary file names
        """

        super(POISSON_BAK_PHAII, self).__init__(instrument_name, telescope_name, tstart, telapse, channel, rate, quality,
                                        grouping, exposure,
                                        backscale, respfile,
                                        ancrfile)

    def _build_spectrum_extension(self):

        # build the extension

        spectrum_extension = POISSON_BKG_SPECTRUM(self._tstart,
                                                  self._telapse,
                                                  self._channel,
                                                  self._rate,
                                                  self._quality,
                                                  self._grouping,
                                                  self._exposure,
                                                  self._backscale,
                                                  self._respfile,
                                                  self._ancrfile)

        return spectrum_extension

