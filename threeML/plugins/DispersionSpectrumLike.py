import copy
from typing import Optional

import numpy as np
import pandas as pd

from threeML.io.logging import setup_logger
from threeML.plugins.SpectrumLike import SpectrumLike
from threeML.utils.OGIP.response import InstrumentResponse
from threeML.utils.spectrum.binned_spectrum import (
    BinnedSpectrumWithDispersion, ChannelSet)

log = setup_logger(__name__)

__instrument_name = "General binned spectral data with energy dispersion"


class DispersionSpectrumLike(SpectrumLike):
    def __init__(
        self,
        name: str,
        observation,
        background=None,
        background_exposure=None,
        verbose=True,
        tstart=None,
        tstop=None,
    ):
        """
        A plugin for generic spectral data with energy dispersion, accepts an observed binned spectrum,
        and a background binned spectrum or plugin with the background data.

        In the case of a binned background spectrum, the background model is profiled
        out and the appropriate profile-likelihood is used to fit the total spectrum. In this
        case, caution must be used when there are zero background counts in bins as the
        profiled background parameters (one per channel) will then have zero information from which to
        constrain the background. It is recommended to bin the spectrum such that there is one background count
        per channel.

        If either an SpectrumLike or XYLike instance is provided as background, it is assumed that this is the
        background data and the likelihood model from this plugin is used to simultaneously fit the background
        and source.

        :param name: the plugin name
        :param observation: the observed spectrum
        :param background: the background spectrum or a plugin from which the background will be modeled
        :param background_exposure: (optional) adjust the background exposure of the modeled background data comes from and
        XYLike plugin
        :param verbose: turn on/off verbose logging
                """
        assert isinstance(
            observation, BinnedSpectrumWithDispersion
        ), "observed spectrum is not an instance of BinnedSpectrumWithDispersion"

        assert (
            observation.response is not None
        ), "the observed spectrum does not have a response"

        # assign the response to the plugins

        self._rsp = observation.response  # type: InstrumentResponse

        super(DispersionSpectrumLike, self).__init__(
            name=name,
            observation=observation,
            background=background,
            background_exposure=background_exposure,
            verbose=verbose,
            tstart=tstart,
            tstop=tstop,

        )

        self._predefined_energies = self._rsp.monte_carlo_energies

    def set_model(self, likelihoodModel):
        """
        Set the model to be used in the joint minimization.
        """

        log.debug(f"model set for {self._name}")

        # Store likelihood model

        self._like_model = likelihoodModel

        # We assume there are no extended sources, since we cannot handle them here

        assert self._like_model.get_number_of_extended_sources() == 0, (
            "OGIP-like plugins do not support " "extended sources"
        )

        # Get the differential flux function, and the integral function, with no dispersion,
        # we simply integrate the model over the bins

        differential_flux, integral = self._get_diff_flux_and_integral(self._like_model,
                                                                       integrate_method=self._model_integrate_method)

        log.debug(f"{self._name} passing intfral flux function to RSP")

        self._rsp.set_function(integral)
        self._integral_flux = integral

    def _evaluate_model(self, precalc_fluxes: Optional[np.array] = None):
        """
        evaluates the full model over all channels
        :return:
        """

        return self._rsp.convolve(precalc_fluxes=precalc_fluxes)

    def set_model_integrate_method(self,
                                   method: str):
        """
        Change the integrate method for the model integration
        :param method: (str) which method should be used (simpson or trapz)
        """
        assert method in [
            "simpson", "trapz"], "Only simpson and trapz are valid intergate methods."
        self._model_integrate_method = method
        log.info(f"{self._name} changing model integration method to {method}")

        # if like_model already set, upadte the integral function
        if self._like_model is not None:
            differential_flux, integral = self._get_diff_flux_and_integral(self._like_model,
                                                                           integrate_method=method)
            self._rsp.set_function(integral)
            self._integral_flux = integral

    def get_simulated_dataset(self, new_name=None, **kwargs):
        """
        Returns another DispersionSpectrumLike instance where data have been obtained by randomizing the current expectation from the
        model, as well as from the background (depending on the respective noise models)

        :return: a DispersionSpectrumLike simulated instance
         """

        # pass the response thru to the constructor
        return super(DispersionSpectrumLike, self).get_simulated_dataset(
            new_name=new_name, **kwargs
        )

    def get_pha_files(self):
        info = {}

        # we want to pass copies so that
        # the user doesn't grab the instance
        # and try to modify things. protection
        info["pha"] = copy.copy(self._observed_spectrum)

        if self._background_spectrum is not None:
            info["bak"] = copy.copy(self._background_spectrum)

        info["rsp"] = copy.copy(self._rsp)

        return info

    def display_rsp(self):
        """
        Display the currently loaded full response matrix, i.e., RMF and ARF convolved
        :return:
        """

        self._rsp.plot_matrix()

    @property
    def response(self) -> InstrumentResponse:
        return self._rsp

    def _output(self):
        # type: () -> pd.Series

        super_out = super(DispersionSpectrumLike,
                          self)._output()  # type: pd.Series

        the_df = pd.Series({"response": self._rsp.rsp_filename})

        return super_out.append(the_df)

    def write_pha(self, filename: str, overwrite: bool = False, force_rsp_write: bool = False) -> None:
        """
        Writes the observation, background and (optional) rsp to PHAII fits files

        :param filename: base file name to write out
        :param overwrite: if you would like to force overwriting of the files
        :param force_rsp_write: force the writing of an rsp even if not required

        """

        # we need to pass up the variables to an OGIPLike
        # so that we have the proper variable name

        # a local import here because OGIPLike is dependent on this

        from threeML.plugins.OGIPLike import OGIPLike

        ogiplike = OGIPLike.from_general_dispersion_spectrum(self)
        ogiplike.write_pha(
            file_name=filename, overwrite=overwrite, force_rsp_write=force_rsp_write
        )

    @staticmethod
    def _build_fake_observation(
        fake_data, channel_set, source_errors, source_sys_errors, is_poisson, **kwargs
    ):
        """
        This is the fake observation builder for SpectrumLike which builds data
        for a binned spectrum without dispersion. It must be overridden in child classes.

        :param fake_data: series of values... they are ignored later
        :param channel_set: a channel set
        :param source_errors:
        :param source_sys_errors:
        :param is_poisson:
        :return:
        """

        assert (
            "response" in kwargs
        ), "A response was not provided. Cannor build synthetic observation"

        response = kwargs.pop("response")

        observation = BinnedSpectrumWithDispersion(
            fake_data,
            exposure=1.0,
            response=response,
            count_errors=source_errors,
            sys_errors=source_sys_errors,
            quality=None,
            scale_factor=1.0,
            is_poisson=is_poisson,
            mission="fake_mission",
            instrument="fake_instrument",
            tstart=0.0,
            tstop=1.0,
        )

        return observation

    @classmethod
    def from_function(
        cls,
        name: str,
        source_function,
        response,
        source_errors=None,
        source_sys_errors=None,
        background_function=None,
        background_errors=None,
        background_sys_errors=None,
    ):
        # type: () -> DispersionSpectrumLike
        """

        Construct a simulated spectrum from a given source function and (optional) background function. If source and/or background errors are not supplied, the likelihood is assumed to be Poisson.

        :param name: simulated data set name
        :param source_function: astromodels function
        :param response: 3ML Instrument response
        :param source_errors: (optional) gaussian source errors
        :param source_sys_errors: (optional) systematic source errors
        :param background_function: (optional) astromodels background function
        :param background_errors: (optional) gaussian background errors
        :param background_sys_errors: (optional) background systematic errors
        :return: simulated DispersionSpectrumLike plugin
        """

        channel_set = ChannelSet.from_instrument_response(response)

        energy_min, energy_max = channel_set.bin_stack.T

        # pass the variables to the super class

        return super(DispersionSpectrumLike, cls).from_function(
            name,
            source_function,
            energy_min,
            energy_max,
            source_errors,
            source_sys_errors,
            background_function,
            background_errors,
            background_sys_errors,
            response=response,
        )
