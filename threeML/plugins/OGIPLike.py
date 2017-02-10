import collections
import pandas as pd

from threeML.plugins.DispersionSpectrumLike import DispersionSpectrumLike
from threeML.plugins.spectrum.pha_spectrum import PHASpectrum
from threeML.io.rich_display import display
from astromodels.utils.valid_variable import is_valid_variable_name

from threeML.plugins.OGIP.pha import PHAWrite

__instrument_name = "All OGIP-compliant instruments"

class OGIPLike(DispersionSpectrumLike):

    def __init__(self, name, observation, background=None, response=None, arf_file=None, spectrum_number=None, verbose=True):

        assert is_valid_variable_name(name), "Name %s is not a valid name for a plugin. You must use a name which is " \
                                             "a valid python identifier: no spaces, no operators (+,-,/,*), " \
                                             "it cannot start with a number, no special characters" % name

        # Read the pha file (or the PHAContainer instance)

        pha = PHASpectrum(observation,spectrum_number=spectrum_number,file_type='observed',rsp_file=response,arf_file=arf_file)



        # Get the required background file, response and (if present) arf_file either from the
        # calling sequence or the file.
        # NOTE: if one of the file is specified in the calling sequence, it will be used whether or not there is an
        # equivalent specification in the header. This allows the user to override the content of the header of the
        # PHA file, if needed

        if background is None:

            background = pha.background_file

            assert background is not None, "No background file provided, and the PHA file does not specify one."

        # Get a PHA instance with the background, we pass the response to get the energy bounds in the
        # histogram constructor. It is not saved to the background class

        bak = PHASpectrum(background, spectrum_number=spectrum_number,file_type='background',rsp_file=pha.response)

        # we do not need to pass the response as it is contained in the observation (pha) spectrum
        # already.

        super(OGIPLike, self).__init__(name=name,
                                       observed_spectrum=pha,
                                       background_spectrum=bak,
                                       verbose=verbose)

    def get_simulated_dataset(self, new_name=None,**kwargs):
        """
        Returns another OGIPLike instance where data have been obtained by randomizing the current expectation from the
        model, as well as from the background (depending on the respective noise models)

        :return: a DispersionSpectrumLike simulated instance
         """

        # pass the response thru to the constructor
        return super(OGIPLike, self).get_simulated_dataset(new_name=new_name,
                                                           spectrum_number=1,
                                                           **kwargs)

    @property
    def grouping(self):

        return self._observed_spectrum.grouping

    def write_pha(self, file_name, overwrite=False):
        """
        Create a pha file of the current pha selections


        :param file_name: output file name (excluding extension)
        :return: None
        """

        pha_writer = PHAWrite(self)

        pha_writer.write(file_name, overwrite=overwrite)

    def display(self):
        """
        Displays the current content of the OGIP object
        :return:
        """

        display(self._output().to_frame())

    def __repr__(self):




            return self._output().to_string()

    def _output(self):

        obs = collections.OrderedDict()

        obs['n. channels'] = self._observed_spectrum.n_channels

        obs['total rate'] = self._observed_spectrum.total_rate

        if not self._observed_spectrum.is_poisson:
            obs['total rate error'] = self._observed_spectrum.total_rate_error

        obs['total bkg. rate'] = self._background_spectrum.total_rate

        if not self._background_spectrum.is_poisson:
            obs['total bkg. rate error'] = self._background_spectrum.total_rate_error

        obs['exposure'] = self.exposure
        obs['bkg. exposure'] = self.background_exposure
        obs['significance'] = self.significance
        obs['is poisson'] = self._observed_spectrum.is_poisson
        obs['bkg. is poisson'] = self._background_spectrum.is_poisson
        obs['response'] = self._observed_spectrum.response_file

        return pd.Series(data=obs, index=obs.keys())



