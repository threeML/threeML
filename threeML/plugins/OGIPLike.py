from pathlib import Path
from typing import Optional, Union

import pandas as pd
from astromodels.utils.valid_variable import is_valid_variable_name

from threeML.io.logging import setup_logger
from threeML.plugins.DispersionSpectrumLike import DispersionSpectrumLike
from threeML.plugins.SpectrumLike import SpectrumLike
from threeML.plugins.XYLike import XYLike
from threeML.utils.OGIP.pha import PHAII, PHAWrite
from threeML.utils.spectrum.pha_spectrum import PHASpectrum

__instrument_name = "All OGIP-compliant instruments"

log = setup_logger(__name__)

_valid_obs_types = (str, Path, PHASpectrum, PHAII)
_valid_bkg_types = (str, Path, PHASpectrum, PHAII, SpectrumLike, XYLike)


class OGIPLike(DispersionSpectrumLike):
    def __init__(
        self,
        name: str,
        observation: Union[str, Path, PHASpectrum, PHAII],
        background: Optional[Union[str, Path, PHASpectrum, PHAII, SpectrumLike,
                                   XYLike]] = None,
        response: Optional[str] = None,
        arf_file: Optional[str] = None,
        spectrum_number: Optional[int] = None,
        verbose: bool = True,
    ):

        assert is_valid_variable_name(name), (
            "Name %s is not a valid name for a plugin. You must use a name which is "
            "a valid python identifier: no spaces, no operators (+,-,/,*), "
            "it cannot start with a number, no special characters" % name)

        # Read the pha file (or the PHAContainer instance)

        for t in _valid_obs_types:
            if isinstance(observation, t):
                break
        else:

            log.error(
                f"observation must be a FITS file name or PHASpectrum, not {type(observation)}"
            )
            raise RuntimeError()

        for t in _valid_bkg_types:
            if isinstance(background, t) or (background is None):
                break

        else:

            log.error(
                f"background must be a FITS file name, PHASpectrum, a Plugin or None, not {type(background)}"
            )

            raise RuntimeError()

        if not isinstance(observation, PHASpectrum):

            pha = PHASpectrum(
                observation,
                spectrum_number=spectrum_number,
                file_type="observed",
                rsp_file=response,
                arf_file=arf_file,
            )

        else:

            pha = observation

        # Get the required background file, response and (if present) arf_file either from the
        # calling sequence or the file.
        # NOTE: if one of the file is specified in the calling sequence, it will be used whether or not there is an
        # equivalent specification in the header. This allows the user to override the content of the header of the
        # PHA file, if needed

        if background is None:

            log.debug(f"{name} has no bkg set")

            background = pha.background_file

            # assert background is not None, "No background file provided, and the PHA file does not specify one."

        # Get a PHA instance with the background, we pass the response to get the energy bounds in the
        # histogram constructor. It is not saved to the background class

        if background is None:

            # in the case there is no background file

            bak = None

        elif isinstance(background, SpectrumLike) or isinstance(
                background, XYLike):

            # this will be a background

            bak = background

        elif not isinstance(background, PHASpectrum):

            bak = PHASpectrum(
                background,
                spectrum_number=spectrum_number,
                file_type="background",
                rsp_file=pha.response,
            )

        else:

            bak = background

        # we do not need to pass the response as it is contained in the observation (pha) spectrum
        # already.

        super(OGIPLike, self).__init__(name=name,
                                       observation=pha,
                                       background=bak,
                                       verbose=verbose)

    def get_simulated_dataset(self, new_name: str = None, **kwargs):
        # type: (str, dict) -> OGIPLike
        """
        Returns another OGIPLike instance where data have been obtained by randomizing the current expectation from the
        model, as well as from the background (depending on the respective noise models)

        :param new_name: name of the simulated plugin
        :param kwargs: keywords to pass back up to parents
        :return: a DispersionSpectrumLike simulated instance
         """

        # pass the response thru to the constructor
        return super(OGIPLike, self).get_simulated_dataset(new_name=new_name,
                                                           spectrum_number=1,
                                                           response=self._rsp,
                                                           **kwargs)

    @property
    def grouping(self):

        return self._observed_spectrum.grouping

    def write_pha(self,
                  file_name: str,
                  overwrite: bool = False,
                  force_rsp_write: bool = False) -> None:
        """
        Create a pha file of the current pha selections


        :param file_name: output file name (excluding extension)
        :param overwrite: overwrite the files
        :param force_rsp_write: for an rsp to be saved

        :return: None
        """

        pha_writer = PHAWrite(self)

        pha_writer.write(file_name,
                         overwrite=overwrite,
                         force_rsp_write=force_rsp_write)

    def _output(self):
        # type: () -> pd.Series

        superout = super(OGIPLike, self)._output()

        if self._background_spectrum is not None:
            bak_file = self._background_spectrum.filename
        else:
            bak_file = None

        this_out = {
            "pha file": self._observed_spectrum.filename,
            "bak file": bak_file
        }

        this_df = pd.Series(this_out)

        return this_df.append(superout)

    @classmethod
    def from_general_dispersion_spectrum(cls, dispersion_like):
        # type: (DispersionSpectrumLike) -> OGIPLike
        """
        Build on OGIPLike from a dispersion like.
        This makes it easy to write a dispersion like to a
        pha file

        :param dispersion_like:
        :return:
        """

        pha_files = dispersion_like.get_pha_files()
        observed = pha_files["pha"]
        background = pha_files["bak"]

        observed_pha = PHASpectrum.from_dispersion_spectrum(
            observed, file_type="observed")

        if background is None:
            background_pha = None
        else:
            background_pha = PHASpectrum.from_dispersion_spectrum(
                background, file_type="background")

        return cls(
            dispersion_like.name,
            observation=observed_pha,
            background=background_pha,
            verbose=False,
        )
