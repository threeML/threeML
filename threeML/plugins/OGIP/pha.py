import os
import astropy.io.fits as fits
import numpy as np
import warnings
from collections import MutableMapping
from copy import copy
import pkg_resources

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

            if '.root' not in phafile:

                self._init_from_FITS(phafile, spectrum_number, file_type)

            else:

                self._init_from_ROOT()

        else:

            # Assume this is a PHAContainer or some other object with the same interface

            self._init_from_pha_container(phafile, file_type)

    def _init_from_ROOT(self):

        # This will be needed for the VERITAS plugin

        raise NotImplementedError("Not yet implemented")

    def _init_from_pha_container(self, phafile, file_type):

        phafile.setup_pha(self)

        self._file_type = file_type
        self._spectrum_number = None

        if self._rates is None:
            raise RuntimeError("The PHA container has no RATES. It is invalid")

        self._has_rates = True
        self._data_column_name = "RATE"

        self._typeII = False

        assert self._rates.shape[0] == self._gathered_keywords['detchans'], \
            "The lenght of RATES and the number of CHANNELS is not the same"

    def _init_from_FITS(self, phafile, spectrum_number, file_type='observed'):

        assert file_type.lower() in ['observed', 'background'], "Unrecognized filetype keyword value"

        self._file_type = file_type.lower()

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

            self._spectrum_number = spectrum_number

            spectrum = f[HDUidx]
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

                    key_has_been_collected = True

                # Note that we check again because the content of the column can override the content of the header

                if keyname in _might_be_columns[self._file_type] and self._typeII:

                    # Check if there is a column with this name

                    if keyname in data.columns.names:
                        # This will set the exposure, among other things

                        self._gathered_keywords[internal_name] = data[keyname][self._spectrum_number - 1]

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

                        self._gathered_keywords['ancrfile'] = "NONE"

                    else:

                        raise RuntimeError("Keyword %s not found. File %s is not a proper PHA "
                                           "file" % (keyname, phafile))

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


class PHAContainer(MutableMapping):
    _allowed_keys = "rates rate_errors n_channels sys_errors exposure is_poisson background_file scale_factor response_file ancillary_file instrument mission quality".split()

    _gathered_keywords = "n_channels exposure scale_factor is_poisson background_file response_file ancillary_file mission instrument".split()

    def accept(self, key):
        # Only accept items with interger key and string value
        return key in PHAContainer._allowed_keys

    def __init__(self, **kwargs):
        """
        Defines a container for PHA data elements that can only except the correct keywords.

        """

        unset_keys = copy(PHAContainer._allowed_keys)

        kwargs = {k.lower(): v for k, v in kwargs.iteritems()}

        for key, _ in kwargs.items():

            if key not in PHAContainer._allowed_keys:

                kwargs.pop(key)

            else:

                unset_keys.remove(key)

        self.dict = dict(**kwargs)

        for key in unset_keys:

            if key == 'is_poisson':
                self.dict[key] = True
            elif key == 'scale_factor':
                self.dict[key] = 1.
            else:
                self.dict[key] = None

        self.is_container = True

    def __setitem__(self, key, val):
        if key not in PHAContainer._allowed_keys:
            raise KeyError(
                    'Valid keywords: "rates rate_errors n_channels sys_errors exposure is_poisson background_file scale_factor response_file ancillary_file mission instrument" ')
        self.dict[key] = val

    def __getitem__(self, key):

        return self.dict[key]

    def iter_values(self):

        tmp = {}

        for key in PHAContainer._allowed_keys:

            if key not in PHAContainer._gathered_keywords:
                tmp["_" + key] = self.dict[key]

        return tmp.iteritems()

    def iter_keywords(self):

        tmp = {}

        for key in PHAContainer._gathered_keywords:
            tmp[key] = self.dict[key]

        return tmp.iteritems()

    def get_gathered_keywords(self):
        """Return the file dictionary needed by PHA"""

        tmp = {}

        key_lookup = dict(zip(PHAContainer._gathered_keywords,
                              ['detchans', 'exposure', 'backscal', 'poisserr', 'backfile', 'respfile', 'ancrfile',
                               'mission', 'instrument']))

        for key in PHAContainer._gathered_keywords:
            tmp[key_lookup[key]] = self.dict[key]

        return tmp

    def setup_pha(self, pha):

        # Set the values
        if self.dict['rates'] is None:
            RuntimeError('RATES is None. A valid PHA must contain rates!')

        for key, value in self.iter_values():
            setattr(pha, key, np.array(value))

        # self gathered keywords
        setattr(pha, '_gathered_keywords', self.get_gathered_keywords())

    def __delitem__(self, key):
        RuntimeWarning("You cannot delete keys!")

    def __len__(self):
        return sum(1 for _ in self)

    def __iter__(self):
        for key in self.dict:
            yield key

    def __repr__(self):
        return repr(dict(self))

    def __str__(self):
        return str(dict(self))


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

        self._max_length_background_file_name = 0
        self._max_length_resp_file_name = 0
        self._max_length_anc_file_name = 0

        self._pseudo_time = 0.

        self._spec_iterartor = 1

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

            self._append_opig(ogip)

        self._write_phaII(overwrite)

    def _append_opig(self, ogip):
        """
        Add an ogip instance's data into the data list

        :param ogip: and OGIPLike instance
        :return: None
        """

        # grab the ogip pha info
        pha_info = ogip.get_pha_files()

        first_channel = pha_info['rsp'].first_channel

        for key in ['pha', 'bak']:

            if key == 'pha':

                if pha_info[key].background_file is not None:

                    self._backfile[key].append(pha_info[key].background_file)

                    if len(pha_info[key].background_file) > self._max_length_background_file_name:
                        self._max_length_background_file_name = len(pha_info[key].background_file)

                else:

                    self._backfile[key].append('%s_bak.pha{%d}' % (self._outfile_basename, self._spec_iterartor))

                    if len('%s_bak.pha{%d}' % (
                            self._outfile_basename, self._spec_iterartor)) > self._max_length_background_file_name:
                        self._max_length_background_file_name = len(
                                '%s_bak.pha{%d}' % (self._outfile_basename, self._spec_iterartor))

                    # We want to write the bak file

                    self._write_bak_file = True

            if pha_info[key].ancillary_file is not None:

                self._ancrfile[key].append(pha_info[key].ancillary_file)

                if len(pha_info[key].ancillary_file) > self._max_length_anc_file_name:
                    self._max_length_anc_file_name = len(pha_info[key].ancillary_file)



            else:

                # There is no ancillary file, so we need to flag it.

                self._ancrfile[key].append('none')

                if 4 > self._max_length_anc_file_name:
                    self._max_length_anc_file_name = 4

            if pha_info['rsp'].rsp_filename is not None:

                self._respfile[key].append(pha_info['rsp'].rsp_filename)

                if len(pha_info['rsp'].rsp_filename) > self._max_length_resp_file_name:
                    self._max_length_resp_file_name = len(pha_info['rsp'].rsp_filename)

            else:

                # This will be reached in the case that a response was generated from a plugin
                # e.g. if we want to use weighted DRMs from GBM. We do not handle this just yet.

                NotImplementedError("In the future this indicates that we need to generate an RSP file.")

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

            self._spec_iterartor += 1

    def _write_phaII(self, overwrite):

        # Fix this later... if needed.
        trigTime = None

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

        n_channel = len(self._rate['pha'][0])

        vector_format_D = "%sD" % (n_channel)
        vector_format_I = "%sI" % (n_channel)

        # Do we also want to write out a
        # background file?
        if self._write_bak_file:

            keys = ['pha', 'bak']

        else:

            keys = ['pha']

        for key in keys:

            if trigTime is not None:

                # use trigTime as reference for TSTART
                tstart_column = fits.Column(name='TSTART',
                                            format='D',
                                            array=np.array(self._tstart[key]),
                                            unit="s",
                                            bzero=trigTime)
            else:

                tstart_column = fits.Column(name='TSTART',
                                            format='D',
                                            array=np.array(self._tstart[key]),
                                            unit="s")

            t_elapse_column = fits.Column(name='TELAPSE',
                                          format='D',
                                          array=np.array(self._tstop[key]) - np.array(self._tstart[key]),
                                          unit="s")

            spec_num_column = fits.Column(name='SPEC_NUM',
                                          format='I',
                                          array=np.arange(1, self._n_spectra + 1, dtype=np.int32))

            channel_column = fits.Column(name='CHANNEL',
                                         format=vector_format_I,
                                         array=np.array(self._channel[key]))

            rate_column = fits.Column(name='RATE',
                                      format=vector_format_D,
                                      array=np.array(self._rate[key]),
                                      unit="Counts/s")

            if (self._is_poisson[key] == False):
                stat_err_column = fits.Column(name='STAT_ERR',
                                              format=vector_format_D,
                                              array=np.array(self._stat_err[key]))

                sys_err_column = fits.Column(name='SYS_ERR',
                                             format=vector_format_D,
                                             array=np.array(self._sys_err[key]))

            quality_column = fits.Column(name='QUALITY',
                                         format=vector_format_I,
                                         array=np.array(self._quality[key]))

            grouping_column = fits.Column(name='GROUPING',
                                          format=vector_format_I,
                                          array=np.array(self._grouping[key]))

            exposure_column = fits.Column(name='EXPOSURE',
                                          format='D',
                                          array=np.array(self._exposure[key]),
                                          unit="s")

            backscale_column = fits.Column(name='BACKSCAL',
                                           format='D',
                                           array=np.array(self._backscal[key]))

            respfile_column = fits.Column(name='RESPFILE',
                                          format='%iA' % (self._max_length_resp_file_name + 2),
                                          array=np.array(self._respfile[key]))

            ancrfile_column = fits.Column(name='ANCRFILE',
                                          format='%iA' % (self._max_length_anc_file_name + 2),
                                          array=np.array(self._ancrfile[key]))

            # There are the base columns.
            # We will append to them as needed
            # by the type of data.

            use_columns = [tstart_column,
                           t_elapse_column,
                           spec_num_column,
                           channel_column,
                           rate_column,
                           quality_column,
                           grouping_column,
                           exposure_column,
                           backscale_column,
                           respfile_column,
                           ancrfile_column]


            if key == 'pha':

                backfile_column = fits.Column(name='BACKFILE',
                                              format='%iA' % (self._max_length_background_file_name + 2),
                                              array=np.array(self._backfile[key]))

                use_columns.append(backfile_column)

            # Insert the stat and sys columns if not Poisson
            # errors

            if (self._is_poisson[key] == False):

                use_columns.insert(5, stat_err_column)
                use_columns.insert(6, sys_err_column)

            column_defs = fits.ColDefs(use_columns)

            new_table = fits.BinTableHDU.from_columns(column_defs)

            # Add the keywords required by the OGIP standard:
            new_table.header.set('EXTNAME', 'SPECTRUM')

            # TODO: add corrscal once implemented
            new_table.header.set('CORRSCAL', 1.0)
            new_table.header.set('AREASCAL', 1.0)
            # new_table.header.set('BACKSCAL', 1.0)
            new_table.header.set('HDUCLASS', 'OGIP')
            new_table.header.set('HDUCLAS1', 'SPECTRUM')
            # TODO: determine spectrum type in PHA class
            new_table.header.set('HDUCLAS2', 'TOTAL')
            new_table.header.set('HDUCLAS3', 'RATE')
            new_table.header.set('HDUCLAS4', 'TYPE:II')
            new_table.header.set('HDUVERS', '1.2.0')
            new_table.header.set('TELESCOP', self._mission[key])  # Modify this
            new_table.header.set('INSTRUME', self._instrument[key])  # assuming all have the same name

            # TODO: check with GV what this is
            new_table.header.set('FILTER', 'none')

            new_table.header.set('CHANTYPE', 'PHA')
            new_table.header.set('POISSERR', self._is_poisson[key])
            new_table.header.set('DETCHANS', len(self._channel[key][0]))
            new_table.header.set('CREATOR', "3ML v.%s" % (pkg_resources.get_distribution("threeML").version),
                                 "(G.Vianello, giacomov@slac.stanford.edu)")

            # Write to the required filename

            new_table.writeto(self._outfile_name[key], clobber=overwrite)
