import os
import warnings
from builtins import object
from pathlib import Path

import astropy.io.fits as fits
import astropy.units as u
import numpy as np

from threeML.io.file_utils import sanitize_filename
from threeML.io.fits_file import FITSExtension, FITSFile
from threeML.utils.OGIP.response import EBOUNDS, SPECRESP_MATRIX
from threeML.io.logging import setup_logger
log = setup_logger(__name__)


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

        log.debug(f"registered {len(ogiplike)} plugins")
        
        self._n_spectra = len(ogiplike)

        # The following lists corresponds to the different columns in the PHA/CSPEC
        # formats, and they will be filled up by addSpectrum()

        self._tstart = {"pha": [], "bak": []}
        self._tstop = {"pha": [], "bak": []}
        self._channel = {"pha": [], "bak": []}
        self._rate = {"pha": [], "bak": []}
        self._stat_err = {"pha": [], "bak": []}
        self._sys_err = {"pha": [], "bak": []}
        self._backscal = {"pha": [], "bak": []}
        self._quality = {"pha": [], "bak": []}
        self._grouping = {"pha": [], "bak": []}
        self._exposure = {"pha": [], "bak": []}
        self._backfile = {"pha": [], "bak": []}
        self._respfile = {"pha": [], "bak": []}
        self._ancrfile = {"pha": [], "bak": []}
        self._mission = {"pha": [], "bak": []}
        self._instrument = {"pha": [], "bak": []}

        # If the PHAs have existing background files
        # then it is assumed that we will not need to write them
        # out. THe most likely case is that the background file does not
        # exist i.e. these are simulations are from EventList object
        # Just one instance of no background file existing cause the write
        self._write_bak_file = False

        # Assuming all entries will have one answer
        self._is_poisson = {"pha": True, "bak": True}

        self._pseudo_time = 0.0

        self._spec_iterator = 1

    def write(self, outfile_name: str, overwrite:bool=True, force_rsp_write:bool=False) -> None:
        """
        Write a PHA Type II and BAK file for the given OGIP plugin. Automatically determines
        if BAK files should be generated.


        :param outfile_name: string (excluding .pha) of the PHA to write
        :param overwrite: (optional) bool to overwrite existing file
        :param force_rsp_write: force the writing of an RSP
        :return:
        """

        outfile_name: Path = sanitize_filename(outfile_name)
        
        # Remove the .pha extension if any
        if outfile_name.suffix.lower() == ".pha":

            log.debug(f"stripping {outfile_name} of its suffix")
            
            outfile_name = outfile_name.stem

            

        self._outfile_basename = outfile_name

        self._outfile_name = {
            "pha": Path(f"{outfile_name}.pha"),
            "bak": Path(f"{outfile_name}_bak.pha"),
        }

        self._out_rsp = []

        for ogip in self._ogiplike:

            self._append_ogip(ogip, force_rsp_write)

        self._write_phaII(overwrite)

    def _append_ogip(self, ogip, force_rsp_write: bool) -> None:
        """
        Add an ogip instance's data into the data list

        :param ogip: and OGIPLike instance
        :param force_rsp_write: force the writing of an rsp
        :return: None
        """

        # grab the ogip pha info
        pha_info: dict = ogip.get_pha_files()

        first_channel: int = pha_info["rsp"].first_channel

        for key in ["pha", "bak"]:
            if key not in pha_info:
                continue

            if key == "pha" and "bak" in pha_info:

                if pha_info[key].background_file is not None:

                    log.debug(f" keeping original bak file: {pha_info[key].background_file}")

                    self._backfile[key].append(pha_info[key].background_file)

                else:

                    log.debug(f"creating new bak file: {self._outfile_basename}_bak.pha" + "{%d}" % self._spec_iterator)
                    
                    self._backfile[key].append(
                        f"{self._outfile_basename}_bak.pha"
                        + "{%d}" % self._spec_iterator
                    )

                    # We want to write the bak file

                    self._write_bak_file = True

            else:

                log.debug("not creating a bak file")

                self._backfile[key] = None

            if pha_info[key].ancillary_file is not None:

                log.debug("appending the ancillary file")

                self._ancrfile[key].append(pha_info[key].ancillary_file)

            else:

                # There is no ancillary file, so we need to flag it.

                self._ancrfile[key].append("NONE")

            if pha_info["rsp"].rsp_filename is not None and not force_rsp_write:

                log.debug(f"not creating a new response and keeping {pha_info['rsp'].rsp_filename}")
                
                self._respfile[key].append(pha_info["rsp"].rsp_filename)

            else:

                # This will be reached in the case that a response was generated from a plugin
                # e.g. if we want to use weighted DRMs from GBM.

                rsp_file_name = (
                    f"{self._outfile_basename}.rsp" + "{%d}" % self._spec_iterator
                )

                log.debug(f"creating a new response and saving it to {rsp_file_name}")
                
                self._respfile[key].append(rsp_file_name)

                if key == "pha":

                    self._out_rsp.append(pha_info["rsp"])

            self._rate[key].append(pha_info[key].rates.tolist())

            self._backscal[key].append(pha_info[key].scale_factor)

            if not pha_info[key].is_poisson:

                log.debug("this file is not Poisson and we save the errors")
                
                self._is_poisson[key] = pha_info[key].is_poisson

                self._stat_err[key].append(pha_info[key].rate_errors.tolist())

            else:

                log.debug("this file is Poisson and we do not save the errors")

                self._stat_err[key] = None

            # If there is systematic error, we add it
            # otherwise create an array of zeros as XSPEC
            # simply adds systematic in quadrature to statistical
            # error.

            if (
                pha_info[key].sys_errors.tolist() is not None
            ):  # It returns an array which does not work!

                self._sys_err[key].append(pha_info[key].sys_errors.tolist())

            else:

                self._sys_err[key].append(
                    np.zeros_like(pha_info[key].rates, dtype=np.float32).tolist()
                )

            self._exposure[key].append(pha_info[key].exposure)
            self._quality[key].append(ogip.quality.to_ogip().tolist())
            self._grouping[key].append(ogip.grouping.tolist())
            self._channel[key].append(
                np.arange(pha_info[key].n_channels, dtype=np.int32) + first_channel
            )
            self._instrument[key] = pha_info[key].instrument
            self._mission[key] = pha_info[key].mission

            if ogip.tstart is not None:

                self._tstart[key].append(ogip.tstart)

                if ogip.tstop is not None:

                    self._tstop[key].append(ogip.tstop)

                else:

                    log.error("OGIP TSTART is a number but TSTOP is None. This is a bug.")

                    RuntimeError()

            # We will assume that the exposure is the true DT
            # and assign starts and stops accordingly. This means
            # we are most likely are dealing with a simulation.
            else:

                log.debug("setting duration to exposure")

                self._tstart[key].append(self._pseudo_time)

                self._pseudo_time += pha_info[key].exposure

                self._tstop[key].append(self._pseudo_time)

        self._spec_iterator += 1

    def _write_phaII(self, overwrite):

        # Fix this later... if needed.
        trigger_time = None

        if self._backfile["pha"] is not None:
            # Assuming background and pha files have the same
            # number of channels

            assert len(self._rate["pha"][0]) == len(
                self._rate["bak"][0]
            ), "PHA and BAK files do not have the same number of channels. Something is wrong."

            assert self._instrument["pha"] == self._instrument["bak"], (
                "Instrument for PHA and BAK (%s,%s) are not the same. Something is wrong with the files. "
                % (self._instrument["pha"], self._instrument["bak"])
            )

            assert self._mission["pha"] == self._mission["bak"], (
                "Mission for PHA and BAK (%s,%s) are not the same. Something is wrong with the files. "
                % (self._mission["pha"], self._mission["bak"])
            )

        if self._write_bak_file:

            keys = ["pha", "bak"]

        else:

            keys = ["pha"]

        for key in keys:

            if trigger_time is not None:

                tstart = self._tstart[key] - trigger_time

            else:

                tstart = self._tstart[key]

            # build a PHAII instance

            fits_file = PHAII(
                self._instrument[key],
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
                stat_err=self._stat_err[key],
                is_poisson=self._is_poisson[key],
            )

            # write the file

            fits_file.writeto(self._outfile_name[key], overwrite=overwrite)

        if self._out_rsp:

            # add the various responses needed

            extensions = [EBOUNDS(self._out_rsp[0].ebounds)]

            extensions.extend(
                [
                    SPECRESP_MATRIX(
                        this_rsp.monte_carlo_energies, this_rsp.ebounds, this_rsp.matrix
                    )
                    for this_rsp in self._out_rsp
                ]
            )

            for i, ext in enumerate(extensions[1:]):

                # Set telescope and instrument name
                ext.hdu.header.set("TELESCOP", self._mission["pha"])
                ext.hdu.header.set("INSTRUME", self._instrument["pha"])
                ext.hdu.header.set("EXTVER", i + 1)

            rsp2 = FITSFile(fits_extensions=extensions)

            rsp2.writeto("%s.rsp" % self._outfile_basename, overwrite=True)


def _atleast_2d_with_dtype(value, dtype=None):

    if dtype is not None:
        value = np.array(value, dtype=dtype)

    arr = np.atleast_2d(value)

    return arr


def _atleast_1d_with_dtype(value, dtype=None):

    if dtype is not None:
        value = np.array(value, dtype=dtype)

        if dtype == str:

            # convert None to NONE
            # which is needed for None Type args
            # to string arrays

            idx = np.core.defchararray.lower(value) == "none"

            value[idx] = "NONE"

    arr = np.atleast_1d(value)

    return arr


class SPECTRUM(FITSExtension):

    _HEADER_KEYWORDS = (
        ("EXTNAME", "SPECTRUM", "Extension name"),
        ("CONTENT", "OGIP PHA data", "File content"),
        ("HDUCLASS", "OGIP    ", "format conforms to OGIP standard"),
        ("HDUVERS", "1.1.0   ", "Version of format (OGIP memo CAL/GEN/92-002a)"),
        (
            "HDUDOC",
            "OGIP memos CAL/GEN/92-002 & 92-002a",
            "Documents describing the forma",
        ),
        ("HDUVERS1", "1.0.0   ", "Obsolete - included for backwards compatibility"),
        ("HDUVERS2", "1.1.0   ", "Obsolete - included for backwards compatibility"),
        ("HDUCLAS1", "SPECTRUM", "Extension contains spectral data  "),
        ("HDUCLAS2", "TOTAL ", ""),
        ("HDUCLAS3", "RATE ", ""),
        ("HDUCLAS4", "TYPE:II ", ""),
        ("FILTER", "", "Filter used"),
        ("CHANTYPE", "PHA", "Channel type"),
        ("POISSERR", False, "Are the rates Poisson distributed"),
        ("DETCHANS", None, "Number of channels"),
        ("CORRSCAL", 1.0, ""),
        ("AREASCAL", 1.0, ""),
    )

    def __init__(
        self,
        tstart,
        telapse,
        channel,
        rate,
        quality,
        grouping,
        exposure,
        backscale,
        respfile,
        ancrfile,
        back_file=None,
        sys_err=None,
        stat_err=None,
        is_poisson=False,
    ):

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

        data_list = [
            ("TSTART", tstart),
            ("TELAPSE", telapse),
            ("SPEC_NUM", np.arange(1, n_spectra + 1, dtype=np.int16)),
            ("CHANNEL", channel),
            ("RATE", rate),
            ("QUALITY", quality),
            ("BACKSCAL", backscale),
            ("GROUPING", grouping),
            ("EXPOSURE", exposure),
            ("RESPFILE", respfile),
            ("ANCRFILE", ancrfile),
        ]

        if back_file is not None:

            data_list.append(("BACKFILE", back_file))

        if stat_err is not None:

            assert (
                is_poisson == False
            ), "Tying to enter STAT_ERR error but have POISSERR set true"

            data_list.append(("STAT_ERR", stat_err))

        if sys_err is not None:

            data_list.append(("SYS_ERR", sys_err))

        super(SPECTRUM, self).__init__(tuple(data_list), self._HEADER_KEYWORDS)

        self.hdu.header.set("POISSERR", is_poisson)


class PHAII(FITSFile):
    def __init__(
        self,
        instrument_name,
        telescope_name,
        tstart,
        telapse,
        channel,
        rate,
        quality,
        grouping,
        exposure,
        backscale,
        respfile,
        ancrfile,
        back_file=None,
        sys_err=None,
        stat_err=None,
        is_poisson=False,
    ):

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

        self._tstart = _atleast_1d_with_dtype(tstart, np.float32) * u.s
        self._telapse = _atleast_1d_with_dtype(telapse, np.float32) * u.s
        self._channel = _atleast_2d_with_dtype(channel, np.int16)
        self._rate = _atleast_2d_with_dtype(rate, np.float32) * 1.0 / u.s
        self._exposure = _atleast_1d_with_dtype(exposure, np.float32) * u.s
        self._quality = _atleast_2d_with_dtype(quality, np.int16)
        self._grouping = _atleast_2d_with_dtype(grouping, np.int16)
        self._backscale = _atleast_1d_with_dtype(backscale, np.float32)
        self._respfile = _atleast_1d_with_dtype(respfile, str)
        self._ancrfile = _atleast_1d_with_dtype(ancrfile, str)

        if sys_err is not None:

            self._sys_err = _atleast_2d_with_dtype(sys_err, np.float32)

        else:

            self._sys_err = sys_err

        if stat_err is not None:

            self._stat_err = _atleast_2d_with_dtype(stat_err, np.float32)

        else:

            self._stat_err = stat_err

        if back_file is not None:

            self._back_file = _atleast_1d_with_dtype(back_file, str)
        else:

            self._back_file = np.array(["NONE"] * self._tstart.shape[0])

        # Create the SPECTRUM extension

        spectrum_extension = SPECTRUM(
            self._tstart,
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
            stat_err=self._stat_err,
            is_poisson=is_poisson,
        )

        # Set telescope and instrument name

        spectrum_extension.hdu.header.set("TELESCOP", telescope_name)
        spectrum_extension.hdu.header.set("INSTRUME", instrument_name)
        spectrum_extension.hdu.header.set("DETCHANS", len(self._channel[0]))

        super(PHAII, self).__init__(fits_extensions=[spectrum_extension])

    @classmethod
    def from_time_series(cls, time_series, use_poly=False):

        pha_information = time_series.get_information_dict(use_poly)

        is_poisson = True

        if use_poly:

            is_poisson = False

        return PHAII(
            instrument_name=pha_information["instrument"],
            telescope_name=pha_information["telescope"],
            tstart=pha_information["tstart"],
            telapse=pha_information["telapse"],
            channel=pha_information["channel"],
            rate=pha_information["rates"],
            stat_err=pha_information["rate error"],
            quality=pha_information["quality"].to_ogip(),
            grouping=pha_information["grouping"],
            exposure=pha_information["exposure"],
            backscale=1.0,
            respfile=None,  # pha_information['response_file'],
            ancrfile=None,
            is_poisson=is_poisson,
        )

    @classmethod
    def from_fits_file(cls, fits_file):

        with fits.open(fits_file) as f:

            if "SPECTRUM" in f:
                spectrum_extension = f["SPECTRUM"]
            else:
                log.warning("unable to find SPECTRUM extension: not OGIP PHA!")

                spectrum_extension = None

                for extension in f:
                    hduclass = extension.header.get("HDUCLASS")
                    hduclas1 = extension.header.get("HDUCLAS1")

                    if hduclass == "OGIP" and hduclas1 == "SPECTRUM":
                        spectrum_extension = extension
                        log.warning(
                            "File has no SPECTRUM extension, but found a spectrum in extension %s"
                            % (spectrum_extension.header.get("EXTNAME"))
                        )
                        spectrum_extension.header["EXTNAME"] = "SPECTRUM"
                        break

            spectrum = FITSExtension.from_fits_file_extension(spectrum_extension)

            out = FITSFile(primary_hdu=f["PRIMARY"], fits_extensions=[spectrum])

        return out

    @property
    def instrument(self):
        return
