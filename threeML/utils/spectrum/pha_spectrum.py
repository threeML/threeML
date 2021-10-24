from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import astropy.io.fits as fits
import numpy as np
from numpy.ma import count
import six
from past.utils import old_div

from threeML.io.logging import setup_logger
from threeML.utils.OGIP.pha import PHAII
from threeML.utils.OGIP.response import InstrumentResponse, OGIPResponse
from threeML.utils.progress_bar import trange
from threeML.utils.spectrum.binned_spectrum import (
    BinnedSpectrumWithDispersion,
    Quality,
)
from threeML.utils.spectrum.binned_spectrum_set import BinnedSpectrumSet
from threeML.utils.time_interval import TimeIntervalSet

log = setup_logger(__name__)

_required_keywords = {}
_required_keywords["observed"] = (
    "mission:TELESCOP,instrument:INSTRUME,filter:FILTER,"
    + "exposure:EXPOSURE,backfile:BACKFILE,"
    + "respfile:RESPFILE,"
    + "ancrfile:ANCRFILE,hduclass:HDUCLASS,"
    + "hduclas1:HDUCLAS1,poisserr:POISSERR,"
    + "chantype:CHANTYPE,detchans:DETCHANS,"
    "backscal:BACKSCAL"
).split(",")

# python types, not fits
_required_keyword_types = {"POISSERR": bool}

# hduvers:HDUVERS

_required_keywords["background"] = (
    "mission:TELESCOP,instrument:INSTRUME,filter:FILTER,"
    + "exposure:EXPOSURE,"
    + "hduclass:HDUCLASS,"
    + "hduclas1:HDUCLAS1,poisserr:POISSERR,"
    + "chantype:CHANTYPE,detchans:DETCHANS,"
    "backscal:BACKSCAL"
).split(",")

# hduvers:HDUVERS

_might_be_columns = {}
_might_be_columns["observed"] = (
    "EXPOSURE,BACKFILE," + "CORRFILE,CORRSCAL," + "RESPFILE,ANCRFILE,"
    "BACKSCAL"
).split(",")
_might_be_columns["background"] = ("EXPOSURE,BACKSCAL").split(",")


_valid_input_types = (str, Path, PHAII)


@dataclass(frozen=True)
class _PHAInfo:
    """
    A container to hold all the gathered information
    """

    counts: Iterable[float]
    rates: Iterable[float]
    exposure: Iterable[float]
    is_poisson: bool
    rsp: InstrumentResponse
    gathered_keywords: Dict[str, Any]
    quality: Quality
    file_name: str
    tstart: Optional[Union[float, Iterable[float]]]
    tstop: Optional[Union[float, Iterable[float]]]
    rate_errors: Optional[Iterable[float]]
    sys_errors: Optional[Iterable[float]]
    count_errors: Optional[Iterable[float]]


def _read_pha_or_pha2_file(
    pha_file_or_instance: Union[str, Path, PHAII],
    spectrum_number: Optional[int] = None,
    file_type: str = "observed",
    rsp_file: Optional[Union[str, InstrumentResponse]] = None,
    arf_file: Optional[str] = None,
    treat_as_time_series: bool = False,
) -> _PHAInfo:
    """
    A function to extract information from pha and pha2 files. It is kept separate because the same method is
    used for reading time series (MUCH faster than building a lot of individual spectra) and single spectra.


    :param pha_file_or_instance: either a PHA file name or threeML.plugins.OGIP.pha.PHAII instance
    :param spectrum_number: (optional) the spectrum number of the TypeII file to be used
    :param file_type: observed or background
    :param rsp_file: RMF filename or threeML.plugins.OGIP.response.InstrumentResponse instance
    :param arf_file: (optional) and ARF filename
    :param treat_as_time_series:
    :return:
    """

    for t in _valid_input_types:

        if isinstance(pha_file_or_instance, t):
            break

    else:

        log.error(
            f"Must provide a FITS file name or PHAII instance. Got {type(pha_file_or_instance)}"
        )

        raise RuntimeError()

    if not isinstance(pha_file_or_instance, PHAII):

        pha_file_or_instance: Path = Path(pha_file_or_instance)

        ext = pha_file_or_instance.suffix

        if "{" in ext:
            spectrum_number = int(ext.split("{")[-1].replace("}", ""))

            pha_file_or_instance = str(pha_file_or_instance).split("{")[0]

        # Read the data

        file_name: Path = Path(pha_file_or_instance)

        # create a FITS_FILE instance

        pha_file_or_instance = PHAII.from_fits_file(pha_file_or_instance)

    # If this is already a FITS_FILE instance,

    elif isinstance(pha_file_or_instance, PHAII):

        # we simply create a dummy filename

        file_name: Path = Path("pha_instance")

    else:

        log.error("This is a bug. Should never get here!")

        raise RuntimeError()

    if not file_type.lower() in [
        "observed",
        "background",
    ]:

        log.error("Unrecognized filetype keyword value")

        raise RuntimeError()

    file_type = file_type.lower()

    try:

        HDUidx = pha_file_or_instance.index_of("SPECTRUM")

    except KeyError:

        log.error(f"The input file {file_name} is not in PHA format")
        raise RuntimeError()

    # spectrum_number = spectrum_number

    spectrum = pha_file_or_instance[HDUidx]

    data = spectrum.data

    header = spectrum.header

    # We don't support yet the rescaling

    if "CORRFILE" in header:

        if (header.get("CORRFILE").upper().strip() != "NONE") and (
            header.get("CORRFILE").upper().strip() != ""
        ):

            log.error("CORRFILE is not yet supported")

            raise RuntimeError()

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

            log.warning(
                "Could not find QUALITY in columns or header of PHA file. This is not a valid OGIP file. Assuming QUALITY =0 (good)"
            )

    # looking for tstart and tstop

    tstart = None
    tstop = None

    has_tstart = False
    has_tstop = False
    has_telapse = False

    if "TSTART" in header:

        has_tstart_column = False

        has_tstart = True

    else:

        if "TSTART" in data.columns.names:

            has_tstart_column = True

            has_tstart = True

    if "TELAPSE" in header:

        has_telapse_column = False

        has_telapse = True

    else:

        if "TELAPSE" in data.columns.names:
            has_telapse_column = True

            has_telapse = True

    if "TSTOP" in header:

        has_tstop_column = False

        has_tstop = True

    else:

        if "TSTOP" in data.columns.names:
            has_tstop_column = True

            has_tstop = True

    if has_tstop and has_telapse:

        log.warning(
            "Found TSTOP and TELAPSE. This file is invalid. Using TSTOP."
        )

        has_telapse = False

    # Determine if this file contains COUNTS or RATES

    if "COUNTS" in data.columns.names:

        has_rates = False
        data_column_name = "COUNTS"

    elif "RATE" in data.columns.names:

        has_rates = True
        data_column_name = "RATE"

    else:

        log.error(
            "This file does not contain a RATE nor a COUNTS column. "
            "This is not a valid PHA file"
        )
        raise RuntimeError()

    # Determine if this is a PHA I or PHA II

    if len(data.field(data_column_name).shape) == 2:

        is_typeII_file = True

        if spectrum_number == None and not treat_as_time_series:

            log.error(
                "This is a PHA Type II file. You have to provide a spectrum number"
            )

            raise RuntimeError(
                "This is a PHA Type II file. You have to provide a spectrum number"
            )

    else:

        is_typeII_file = False

    # Collect information from mandatory keywords

    keys = _required_keywords[file_type]

    gathered_keywords = {}

    for k in keys:

        internal_name, keyname = k.split(":")

        key_has_been_collected = False

        if keyname in header:
            if (
                keyname in _required_keyword_types
                and type(header.get(keyname))
                is not _required_keyword_types[keyname]
            ):
                log.warning(
                    "unexpected type of %(keyname)s, expected %(expected_type)s\n found %(found_type)s: %(found_value)s"
                    % dict(
                        keyname=keyname,
                        expected_type=_required_keyword_types[keyname],
                        found_type=type(header.get(keyname)),
                        found_value=header.get(keyname),
                    )
                )
            else:

                gathered_keywords[internal_name] = header.get(keyname)

                # Fix "NONE" in None

                if (
                    gathered_keywords[internal_name] == "NONE"
                    or gathered_keywords[internal_name] == "none"
                ):
                    gathered_keywords[internal_name] = None

                key_has_been_collected = True

        # Note that we check again because the content of the column can override the content of the header

        if keyname in _might_be_columns[file_type] and is_typeII_file:

            # Check if there is a column with this name

            if keyname in data.columns.names:
                # This will set the exposure, among other things

                if not treat_as_time_series:

                    # if we just want a single spectrum

                    gathered_keywords[internal_name] = data[keyname][
                        spectrum_number - 1
                    ]

                else:

                    # else get all the columns

                    gathered_keywords[internal_name] = data[keyname]

                # Fix "NONE" in None
                if (
                    gathered_keywords[internal_name] == "NONE"
                    or gathered_keywords[internal_name] == "none"
                ):
                    gathered_keywords[internal_name] = None

                key_has_been_collected = True

        if not key_has_been_collected:

            # The keyword POISSERR is a special case, because even if it is missing,
            # it is assumed to be False if there is a STAT_ERR column in the file

            if keyname == "POISSERR" and "STAT_ERR" in data.columns.names:

                log.warning(
                    "POISSERR is not set. Assuming non-poisson errors as given in the "
                    "STAT_ERR column"
                )

                gathered_keywords["poisserr"] = False

            elif keyname == "ANCRFILE":

                # Some non-compliant files have no ARF because they don't need one. Don't fail, but issue a
                # warning

                log.warning(
                    "ANCRFILE is not set. This is not a compliant OGIP file. Assuming no ARF."
                )

                gathered_keywords["ancrfile"] = None

            elif keyname == "FILTER":

                # Some non-compliant files have no FILTER because they don't need one. Don't fail, but issue a
                # warning

                log.warning(
                    "FILTER is not set. This is not a compliant OGIP file. Assuming no FILTER."
                )

                gathered_keywords["filter"] = None

            else:

                log.error(
                    f"Keyword {keyname} not found. File {file_name} is not a proper PHA "
                    "file"
                )

                raise RuntimeError()

    is_poisson = gathered_keywords["poisserr"]

    exposure = gathered_keywords["exposure"]

    # now we need to get the response file so that we can extract the EBOUNDS

    if file_type == "observed":

        if rsp_file is None:

            # this means it should be specified in the header
            rsp_file = gathered_keywords["respfile"]

            if arf_file is None:
                arf_file = gathered_keywords["ancrfile"]

                # Read in the response

        if (
            isinstance(rsp_file, six.string_types)
            or isinstance(rsp_file, str)
            or isinstance(rsp_file, Path)
        ):

            rsp: InstrumentResponse = OGIPResponse(rsp_file, arf_file=arf_file)

        elif isinstance(rsp_file, InstrumentResponse):

            # assume a fully formed OGIPResponse
            rsp = rsp_file

        else:

            log.error(f"{rsp_file} is not correct type")

            raise RuntimeError()

    if file_type == "background":
        # we need the rsp ebounds from response to build the histogram

        if not isinstance(rsp_file, InstrumentResponse):

            log.error(
                "You must supply and OGIPResponse to extract the energy bounds"
            )

            raise RuntimeError()

        rsp = rsp_file

    # Now get the data (counts or rates) and their errors. If counts, transform them in rates

    if is_typeII_file:

        # PHA II file
        if has_rates:

            log.debug(f"{file_name} has rates and NOT counts")

            if not treat_as_time_series:

                rates = data.field(data_column_name)[spectrum_number - 1, :]

                rate_errors = None

                if not is_poisson:
                    rate_errors = data.field("STAT_ERR")[spectrum_number - 1, :]

            else:

                rates = data.field(data_column_name)

                rate_errors = None

                if not is_poisson:
                    rate_errors = data.field("STAT_ERR")

        else:

            log.debug(f"{file_name} has counts and NOT rates")

            if not treat_as_time_series:

                # extract the counts

                counts = data.field(data_column_name)[
                    spectrum_number - 1, :
                ].astype(np.int64)

                # count the rates

                rates = counts / exposure

                rate_errors = None

                if not is_poisson:
                    rate_errors = old_div(
                        data.field("STAT_ERR")[spectrum_number - 1, :], exposure
                    )

            else:

                counts = data.field(data_column_name).astype(np.int64)

                rates = counts / np.atleast_2d(exposure).T

                rate_errors = None

                if not is_poisson:
                    rate_errors = old_div(
                        data.field("STAT_ERR"), np.atleast_2d(exposure).T
                    )

        if "SYS_ERR" in data.columns.names:

            if not treat_as_time_series:

                sys_errors = data.field("SYS_ERR")[spectrum_number - 1, :]

            else:

                sys_errors = data.field("SYS_ERR")

        else:

            sys_errors = np.zeros(rates.shape)

        if has_quality_column:

            if not treat_as_time_series:

                try:

                    quality = data.field("QUALITY")[spectrum_number - 1, :]

                except (IndexError):

                    # GBM CSPEC files do not follow OGIP conventions and instead
                    # list simply QUALITY=0 for each spectrum
                    # so we have to read them differently

                    quality_element = data.field("QUALITY")[spectrum_number - 1]

                    log.warning(
                        "The QUALITY column has the wrong shape. This PHAII file does not follow OGIP standards"
                    )

                    if quality_element == 0:

                        quality = np.zeros_like(rates, dtype=int)

                    else:

                        quality = np.zeros_like(rates, dtype=int) + 5

            else:

                # we need to be careful again because the QUALITY column is not always the correct shape

                quality_element = data.field("QUALITY")

                if quality_element.shape == rates.shape:

                    # This is the proper way for the quality to be stored

                    quality = quality_element

                else:

                    quality = np.zeros_like(rates, dtype=int)

                    for i, q in enumerate(quality_element):

                        if q != 0:
                            quality[i, :] = 5

        else:

            if is_all_data_good:

                quality = np.zeros_like(rates, dtype=int)

            else:

                quality = np.zeros_like(rates, dtype=int) + 5

        if has_tstart:

            if has_tstart_column:

                if not treat_as_time_series:

                    tstart = data.field("TSTART")[spectrum_number - 1]

                else:

                    tstart = data.field("TSTART")

        if has_tstop:

            if has_tstop_column:

                if not treat_as_time_series:

                    tstop = data.field("TSTOP")[spectrum_number - 1]

                else:

                    tstop = data.field("TSTOP")

        if has_telapse:

            if has_telapse_column:

                if not treat_as_time_series:

                    tstop = tstart + data.field("TELAPSE")[spectrum_number - 1]

                else:

                    tstop = tstart + data.field("TELAPSE")

    elif not is_typeII_file:

        if treat_as_time_series:

            log.error(
                "This is not a PHAII file but you specified to treat it as a time series"
            )
            raise RuntimeError()

        # PHA 1 file
        if has_rates:

            rates = data.field(data_column_name)

            rate_errors = None

            if not is_poisson:
                rate_errors = data.field("STAT_ERR")

        else:

            counts = data.field(data_column_name).astype(np.int64)

            rates = counts / exposure

            rate_errors = None

            if not is_poisson:
                rate_errors = old_div(data.field("STAT_ERR"), exposure)

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

        # read start and stop times if needed

        if has_tstart:

            if has_tstart_column:

                tstart = data.field("TSTART")

            else:

                tstart = header["TSTART"]

        if has_tstop:

            if has_tstop_column:

                tstop = data.field("TSTOP")

            else:

                tstop = header["TSTOP"]

        if has_telapse:

            if has_telapse_column:

                tstop = tstart + data.field("TELAPSE")

            else:

                tstop = tstart + header["TELAPSE"]

        # Now that we have read it, some safety checks

        if rates.shape[0] != gathered_keywords["detchans"]:
            log.error(
                "The data column (RATES or COUNTS) has a different number of entries than the "
                "DETCHANS declared in the header"
            )
            raise RuntimeError()

    quality = Quality.from_ogip(quality)

    if not treat_as_time_series:

        log.debug(f"{file_name} is not a time series")

        if has_rates:

            counts = rates * exposure

        if not is_poisson:

            log.debug(f"{file_name} is not Poisson")

            count_errors = rate_errors * exposure

        else:

            log.debug(f"{file_name} is Poisson")

            count_errors = None

    else:

        log.debug(f"{file_name} is a time series")

        exposure = np.atleast_2d(exposure).T

        if has_rates:

            counts = rates * exposure

        if not is_poisson:

            log.debug(f"{file_name} is not Poisson")

            count_errors = rate_errors * exposure

        else:

            log.debug(f"{file_name} is Poisson")

            count_errors = None

    return _PHAInfo(
        counts=counts,
        count_errors=count_errors,
        rates=rates,
        rate_errors=rate_errors,
        sys_errors=sys_errors,
        exposure=exposure,
        is_poisson=is_poisson,
        rsp=rsp,
        gathered_keywords=gathered_keywords,
        quality=quality,
        file_name=file_name,
        tstart=tstart,
        tstop=tstop,
    )


class PHASpectrum(BinnedSpectrumWithDispersion):
    def __init__(
        self,
        pha_file_or_instance: Union[str, Path, PHAII],
        spectrum_number: Optional[int] = None,
        file_type: str = "observed",
        rsp_file: Optional[Union[str, InstrumentResponse]] = None,
        arf_file: Optional[str] = None,
    ) -> None:
        """
        A spectrum with dispersion build from an OGIP-compliant PHA FITS file. Both Type I & II files can be read. Type II
        spectra are selected either by specifying the spectrum_number or via the {spectrum_number} file name convention used
        in XSPEC. If the file_type is background, a 3ML InstrumentResponse or subclass must be passed so that the energy
        bounds can be obtained.


        :param pha_file_or_instance: either a PHA file name or threeML.plugins.OGIP.pha.PHAII instance
        :param spectrum_number: (optional) the spectrum number of the TypeII file to be used
        :param file_type: observed or background
        :param rsp_file: RMF filename or threeML.plugins.OGIP.response.InstrumentResponse instance
        :param arf_file: (optional) and ARF filename
        """

        # extract the spectrum number if needed

        for t in _valid_input_types:

            if isinstance(pha_file_or_instance, t):
                break

        else:

            log.error(
                f"Must provide a FITS file name or PHAII instance. Got {type(pha_file_or_instance)}"
            )

            raise RuntimeError()

        pha_information: _PHAInfo = _read_pha_or_pha2_file(
            pha_file_or_instance,
            spectrum_number,
            file_type,
            rsp_file,
            arf_file,
            treat_as_time_series=False,
        )

        # default the grouping to all open bins
        # this will only be altered if the spectrum is rebinned

        self._grouping: np.ndarray = np.ones_like(pha_information.counts)

        # this saves the extra properties to the class

        self._gathered_keywords = pha_information.gathered_keywords

        self._file_type: str = file_type

        self._file_name: str = pha_information.file_name

        # pass the needed spectrum values back up
        # remember that Spectrum reads counts, but returns
        # rates!

        super(PHASpectrum, self).__init__(
            counts=pha_information.counts,
            exposure=pha_information.exposure,
            response=pha_information.rsp,
            count_errors=pha_information.count_errors,
            sys_errors=pha_information.sys_errors,
            is_poisson=pha_information.is_poisson,
            quality=pha_information.quality,
            mission=pha_information.gathered_keywords["mission"],
            instrument=pha_information.gathered_keywords["instrument"],
            tstart=pha_information.tstart,
            tstop=pha_information.tstop,
        )

    def _return_file(self, key) -> Union[None, str]:

        if key in self._gathered_keywords:

            return self._gathered_keywords[key]

        else:

            return None

    def set_ogip_grouping(self, grouping) -> None:
        """
        If the counts are rebinned, this updates the grouping
        :param grouping:

        """

        self._grouping = grouping

    def to_binned_spectrum(self) -> BinnedSpectrumWithDispersion:
        """
        Convert directly to as Binned Spectrum
        :returns:

        """
        return BinnedSpectrumWithDispersion(
            counts=self.counts,
            exposure=self.exposure,
            response=self.response,
            count_errors=self.count_errors,
            sys_errors=self.sys_errors,
            quality=self.quality,
            scale_factor=self.scale_factor,
            is_poisson=self.is_poisson,
            mission=self.mission,
            instrument=self.instrument,
            tstart=self.tstart,
            tstop=self.tstart,
        )

    @property
    def filename(self) -> str:

        return self._file_name

    @property
    def background_file(self) -> Union[None, str]:
        """
                Returns the background file definied in the header, or None if there is none defined
        p
                :return: a path to a file, or None
        """

        back_file = self._return_file("backfile")

        if back_file == "":
            back_file = None

        return back_file

    @property
    def scale_factor(self) -> float:
        """
        This is a scale factor (in the BACKSCAL keyword) which must be used to rescale background and source
        regions

        :return:
        """
        return self._gathered_keywords["backscal"]

    @property
    def response_file(self) -> Union[str, None]:
        """
        Returns the response file definied in the header, or None if there is none defined

        :return: a path to a file, or None
        """
        return self._return_file("respfile")

    @property
    def ancillary_file(self) -> Union[str, None]:
        """
        Returns the ancillary file definied in the header, or None if there is none defined

        :return: a path to a file, or None
        """
        return self._return_file("ancrfile")

    @property
    def grouping(self) -> np.ndarray:

        return self._grouping

    def clone(
        self,
        new_counts=None,
        new_count_errors=None,
        new_exposure=None,
        new_scale_factor=None,
    ) -> "PHASpectrum":
        """
        make a new spectrum with new counts and errors and all other
        parameters the same


        :param new_exposure: the new exposure for the clone
        :param new_scale_factor: the new scale factor for the clone

        :param new_counts: new counts for the spectrum
        :param new_count_errors: new errors from the spectrum
        :return: new pha spectrum
        """

        if new_exposure is None:

            new_exposure = self.exposure

        if new_counts is None:
            new_counts = self.counts
            new_count_errors = self.count_errors

        if new_count_errors is None:
            stat_err = None

        else:

            stat_err = old_div(new_count_errors, new_exposure)

        if self._tstart is None:

            tstart = 0

        else:

            tstart = self._tstart

        if self._tstop is None:

            telapse = new_exposure

        else:

            telapse = self._tstop - tstart

        if new_scale_factor is None:

            new_scale_factor = self.scale_factor

        # create a new PHAII instance

        pha = PHAII(
            instrument_name=self.instrument,
            telescope_name=self.mission,
            tstart=tstart,
            telapse=telapse,
            channel=list(range(1, len(self) + 1)),
            rate=old_div(new_counts, self.exposure),
            stat_err=stat_err,
            quality=self.quality.to_ogip(),
            grouping=self.grouping,
            exposure=new_exposure,
            backscale=new_scale_factor,
            respfile=None,
            ancrfile=None,
            is_poisson=self.is_poisson,
        )

        return pha

    @classmethod
    def from_dispersion_spectrum(
        cls, dispersion_spectrum, file_type="observed", response=None
    ):
        # type: (BinnedSpectrumWithDispersion, str) -> PHASpectrum

        if dispersion_spectrum.is_poisson:

            rate_errors = None

        else:

            rate_errors = dispersion_spectrum.rate_errors

        if dispersion_spectrum.tstart is None:

            tstart = 0

        else:

            tstart = dispersion_spectrum.tstart

        if dispersion_spectrum.tstop is None:

            telapse = dispersion_spectrum.exposure

        else:

            telapse = dispersion_spectrum.tstop - tstart

        pha = PHAII(
            instrument_name=dispersion_spectrum.instrument,
            telescope_name=dispersion_spectrum.mission,
            tstart=tstart,  # TODO: add this in so that we have proper time!
            telapse=telapse,
            channel=list(range(1, len(dispersion_spectrum) + 1)),
            rate=dispersion_spectrum.rates,
            stat_err=rate_errors,
            quality=dispersion_spectrum.quality.to_ogip(),
            grouping=np.ones(len(dispersion_spectrum)),
            exposure=dispersion_spectrum.exposure,
            backscale=dispersion_spectrum.scale_factor,
            respfile=None,
            ancrfile=None,
            is_poisson=dispersion_spectrum.is_poisson,
        )

        if file_type == "background":

            if response is None:

                log.error(
                    "passed a background file but no response to extract energy spectra."
                )

                raise AssertionError()

        else:

            response = dispersion_spectrum.response

        return cls(
            pha_file_or_instance=pha,
            spectrum_number=1,
            file_type=file_type,
            rsp_file=response,
        )


class PHASpectrumSet(BinnedSpectrumSet):
    def __init__(
        self,
        pha_file_or_instance: Union[str, Path, PHAII],
        file_type: str = "observed",
        rsp_file: Optional[str] = None,
        arf_file: Optional[str] = None,
    ):
        """
        A spectrum with dispersion build from an OGIP-compliant PHA FITS file. Both Type I & II files can be read. Type II
        spectra are selected either by specifying the spectrum_number or via the {spectrum_number} file name convention used
        in XSPEC. If the file_type is background, a 3ML InstrumentResponse or subclass must be passed so that the energy
        bounds can be obtained.


        :param pha_file_or_instance: either a PHA file name or threeML.plugins.OGIP.pha.PHAII instance
        :param spectrum_number: (optional) the spectrum number of the TypeII file to be used
        :param file_type: observed or background
        :param rsp_file: RMF filename or threeML.plugins.OGIP.response.InstrumentResponse instance
        :param arf_file: (optional) and ARF filename
        """

        # extract the spectrum number if needed

        for t in _valid_input_types:

            if isinstance(pha_file_or_instance, t):
                break

        else:

            log.error(
                f"Must provide a FITS file name or PHAII instance. Got {type(pha_file_or_instance)}"
            )

            raise RuntimeError()

        with fits.open(pha_file_or_instance) as f:

            try:

                HDUidx = f.index_of("SPECTRUM")

            except KeyError:

                raise RuntimeError(
                    "The input file %s is not in PHA format"
                    % (pha_file_or_instance)
                )

            spectrum = f[HDUidx]
            data = spectrum.data

            if "COUNTS" in data.columns.names:

                has_rates = False
                data_column_name = "COUNTS"

            elif "RATE" in data.columns.names:

                has_rates = True
                data_column_name = "RATE"

            else:

                log.error(
                    "This file does not contain a RATE nor a COUNTS column. "
                    "This is not a valid PHA file"
                )

                raise RuntimeError()

                # Determine if this is a PHA I or PHA II
            if len(data.field(data_column_name).shape) == 2:

                num_spectra = data.field(data_column_name).shape[0]

            else:

                log.error("This appears to be a PHA I and not PHA II file")

                raise RuntimeError()

        pha_information: _PHAInfo = _read_pha_or_pha2_file(
            pha_file_or_instance,
            None,
            file_type,
            rsp_file,
            arf_file,
            treat_as_time_series=True,
        )

        # default the grouping to all open bins
        # this will only be altered if the spectrum is rebinned
        self._grouping = np.ones_like(pha_information.counts)

        # this saves the extra properties to the class

        self._gathered_keywords = pha_information.gathered_keywords

        self._file_type = file_type

        # need to see if we have count errors, tstart, tstop
        # if not, we create an list of None

        if pha_information.count_errors is None:

            count_errors = [None] * num_spectra

        else:

            count_errors = pha_information.count_errors

        if pha_information.tstart is None:

            tstart = [None] * num_spectra

        else:

            tstart = pha_information.tstart

        if pha_information.tstop is None:

            tstop = [None] * num_spectra

        else:

            tstop = pha_information.tstop

        # now build the list of binned spectra

        list_of_binned_spectra = []

        for i in trange(num_spectra, desc="Loading PHAII Spectra"):

            list_of_binned_spectra.append(
                BinnedSpectrumWithDispersion(
                    counts=pha_information.counts[i],
                    exposure=pha_information.exposure[i, 0],
                    response=pha_information.rsp,
                    count_errors=count_errors[i],
                    sys_errors=pha_information.sys_errors[i],
                    is_poisson=pha_information.is_poisson,
                    quality=pha_information.quality.get_slice(i),
                    mission=pha_information.gathered_keywords["mission"],
                    instrument=pha_information.gathered_keywords["instrument"],
                    tstart=tstart[i],
                    tstop=tstop[i],
                )
            )

        # now get the time intervals

        _allowed_time_keys = (("TIME", "ENDTIME"), ("TSTART", "TSTOP"))

        for keys in _allowed_time_keys:

            try:

                start_times = data.field(keys[0])
                stop_times = data.field(keys[1])
                break

            except (KeyError):

                pass

        else:

            log.error(
                f"Could not find times in {pha_file_or_instance}. Tried: {_allowed_time_keys}"
            )

            raise RuntimeError()

        time_intervals = TimeIntervalSet.from_starts_and_stops(
            start_times, stop_times
        )

        reference_time = 0

        # see if there is a reference time in the file

        if "TRIGTIME" in spectrum.header:
            reference_time = spectrum.header["TRIGTIME"]

        for t_number in range(spectrum.header["TFIELDS"]):

            if "TZERO%d" % t_number in spectrum.header:
                reference_time = spectrum.header["TZERO%d" % t_number]

        super(PHASpectrumSet, self).__init__(
            list_of_binned_spectra,
            reference_time=reference_time,
            time_intervals=time_intervals,
        )

    def _return_file(self, key):

        if key in self._gathered_keywords:

            return self._gathered_keywords[key]

        else:

            return None

    def set_ogip_grouping(self, grouping):
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

        return self._return_file("backfile")

    @property
    def scale_factor(self):
        """
        This is a scale factor (in the BACKSCAL keyword) which must be used to rescale background and source
        regions

        :return:
        """
        return self._gathered_keywords["backscal"]

    @property
    def response_file(self):
        """
        Returns the response file definied in the header, or None if there is none defined

        :return: a path to a file, or None
        """
        return self._return_file("respfile")

    @property
    def ancillary_file(self):
        """
        Returns the ancillary file definied in the header, or None if there is none defined

        :return: a path to a file, or None
        """
        return self._return_file("ancrfile")

    @property
    def grouping(self):

        return self._grouping

    def clone(
        self,
        new_counts=None,
        new_count_errors=None,
    ):
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

            stat_err = old_div(new_count_errors, self.exposure)

        # create a new PHAII instance

        pha = PHAII(
            instrument_name=self.instrument,
            telescope_name=self.mission,
            tstart=0,
            telapse=self.exposure,
            channel=list(range(1, len(self) + 1)),
            rate=old_div(new_counts, self.exposure),
            stat_err=stat_err,
            quality=self.quality.to_ogip(),
            grouping=self.grouping,
            exposure=self.exposure,
            backscale=self.scale_factor,
            respfile=None,
            ancrfile=None,
            is_poisson=self.is_poisson,
        )

        return pha

    @classmethod
    def from_dispersion_spectrum(
        cls, dispersion_spectrum, file_type="observed"
    ):
        # type: (BinnedSpectrumWithDispersion, str) -> PHASpectrum

        if dispersion_spectrum.is_poisson:

            rate_errors = None

        else:

            rate_errors = dispersion_spectrum.rate_errors

        pha = PHAII(
            instrument_name=dispersion_spectrum.instrument,
            telescope_name=dispersion_spectrum.mission,
            tstart=dispersion_spectrum.tstart,
            telapse=dispersion_spectrum.tstop - dispersion_spectrum.tstart,
            channel=list(range(1, len(dispersion_spectrum) + 1)),
            rate=dispersion_spectrum.rates,
            stat_err=rate_errors,
            quality=dispersion_spectrum.quality.to_ogip(),
            grouping=np.ones(len(dispersion_spectrum)),
            exposure=dispersion_spectrum.exposure,
            backscale=dispersion_spectrum.scale_factor,
            respfile=None,
            ancrfile=None,
            is_poisson=dispersion_spectrum.is_poisson,
        )

        return cls(
            pha_file_or_instance=pha,
            spectrum_number=1,
            file_type=file_type,
            rsp_file=dispersion_spectrum.response,
        )
