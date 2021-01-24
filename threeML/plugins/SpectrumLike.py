from __future__ import division, print_function

import collections
import copy
import types
from builtins import range, str, zip
from collections.abc import Iterable
from contextlib import contextmanager
from typing import Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import pandas as pd
from astromodels import Model, PointSource, clone_model
from astromodels.core.parameter import Parameter
from astromodels.functions.priors import Uniform_prior
from astromodels.utils.valid_variable import is_valid_variable_name
from past.utils import old_div

from threeML.config.config import threeML_config
from threeML.exceptions.custom_exceptions import (NegativeBackground,
                                                  custom_warnings)
from threeML.io.logging import setup_logger
from threeML.io.plotting.data_residual_plot import ResidualPlot
from threeML.io.plotting.light_curve_plots import (channel_plot,
                                                   disjoint_patch_plot)
from threeML.io.rich_display import display
from threeML.plugin_prototype import PluginPrototype
from threeML.plugins.XYLike import XYLike
from threeML.utils.binner import Rebinner
from threeML.utils.spectrum.binned_spectrum import BinnedSpectrum, ChannelSet
from threeML.utils.spectrum.pha_spectrum import PHASpectrum
from threeML.utils.spectrum.spectrum_likelihood import statistic_lookup
from threeML.utils.statistics.stats_tools import Significance
from threeML.utils.string_utils import dash_separated_string_to_tuple

log = setup_logger(__name__)

NO_REBIN = 1e-99

__instrument_name = "General binned spectral data"

# This defines the known noise models for source and/or background spectra
_known_noise_models = ["poisson", "gaussian", "ideal", "modeled"]


class SpectrumLike(PluginPrototype):
    def __init__(
        self,
        name: str,
        observation: BinnedSpectrum,
        background=None,
        verbose: bool = True,
        background_exposure=None,
        tstart: Optional[Union[float, int]] = None,
        tstop: Optional[Union[float, int]] = None,
    ):
        # type: (str, BinnedSpectrum, BinnedSpectrum, bool) -> None
        """
        A plugin for generic spectral data, accepts an observed binned spectrum,
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

        # Just a toggle for verbosity
        self._verbose = bool(verbose)
        self._name = name

        assert is_valid_variable_name(name), (
            "Name %s is not a valid name for a plugin. You must use a name which is "
            "a valid python identifier: no spaces, no operators (+,-,/,*), "
            "it cannot start with a number, no special characters" % name
        )

        assert isinstance(
            observation, BinnedSpectrum
        ), "The observed spectrum is not an instance of BinnedSpectrum"

        # Precomputed observed (for speed)

        self._observed_spectrum = observation  # type: BinnedSpectrum

        self._has_contiguous_energies = observation.is_contiguous()

        self._predefined_energies = observation.edges

        self._observed_counts = self._observed_spectrum.counts  # type: np.ndarray

        # initialize the background

        background_parameters = self._background_setup(background, observation)

        # unpack the parameters

        (
            self._background_spectrum,
            self._background_plugin,
            self._background_counts,
            self._scaled_background_counts,
        ) = background_parameters

        # Init everything else to None
        self._like_model = None
        self._rebinner = None
        self._source_name = None

        # probe the noise models and then setup the appropriate count errors

        (
            self._observation_noise_model,
            self._background_noise_model,
        ) = self._probe_noise_models()

        (
            self._observed_count_errors,
            self._back_count_errors,
        ) = self._count_errors_initialization()

        # Init the integral methods for background and model integration to default
        self._model_integrate_method = "simpson"
        self._background_integrate_method = "simpson"

        # Initialize a mask that selects all the data.
        # We will initially use the quality mask for the PHA file
        # and set any quality greater than 0 to False. We want to save
        # the native quality so that we can warn the user if they decide to
        # select channels that were flagged as bad.

        self._mask = np.asarray(
            np.ones(self._observed_spectrum.n_channels), np.bool)

        # Now create the nuisance parameter for the effective area correction, which is fixed
        # by default. This factor multiplies the model so that it can account for calibration uncertainties on the
        # global effective area. By default it is limited to stay within 20%

        self._nuisance_parameter = Parameter(
            "cons_%s" % name,
            1.0,
            min_value=0.8,
            max_value=1.2,
            delta=0.05,
            free=False,
            desc="Effective area correction for %s" % name,
        )

        nuisance_parameters = collections.OrderedDict()
        nuisance_parameters[self._nuisance_parameter.name] = self._nuisance_parameter

        # if we have a background model we are going
        # to link all those parameters to new nuisance parameters

        if self._background_plugin is not None:

            log.debug(f"{name} is using a modeled background")

            self._background_noise_model = "modeled"

            for par_name, parameter in list(
                self._background_plugin.likelihood_model.parameters.items()
            ):

                # create a new parameters that is like the one from the background model

                local_name = "bkg_%s_%s" % (par_name, name)
                local_name = local_name.replace(".", "_")

                nuisance_parameters[local_name] = parameter

                # now get the background likelihood model

                differential_flux, integral = self._get_diff_flux_and_integral(
                    self._background_plugin.likelihood_model,
                    integrate_method=self._background_integrate_method
                )

                self._background_integral_flux = integral

        super(SpectrumLike, self).__init__(name, nuisance_parameters)

        if isinstance(self._background_plugin, XYLike):

            if background_exposure is None:
                log.warning(
                    "An XYLike plugin is modeling the background but background_exposure is not set. "
                    "It is assumed the observation and background have the same exposure"
                )

                self._explict_background_exposure = self.exposure

            else:

                self._explicit_background_exposure = background_exposure

        # The following vectors are the ones that will be really used for the computation. At the beginning they just
        # point to the original ones, but if a rebinner is used and/or a mask is created through set_active_measurements,
        # they will contain the rebinned and/or masked versions

        self._current_observed_counts = self._observed_counts
        self._current_observed_count_errors = self._observed_count_errors
        self._current_background_counts = self._background_counts
        self._current_scaled_background_counts = self._scaled_background_counts
        self._current_back_count_errors = self._back_count_errors

        # This will be used to keep track of how many syntethic datasets have been generated
        self._n_synthetic_datasets = 0

        if tstart is not None:

            self._tstart = tstart

        else:

            self._tstart = observation.tstart

        if tstop is not None:

            self._tstop = tstop

        else:

            self._tstop = observation.tstop

        # This is so far not a simulated data set
        self._simulation_storage = None

        # set the mask to the native quality
        self._mask = self._observed_spectrum.quality.good
        # Apply the mask
        self._apply_mask_to_original_vectors()

        # this will be immeadiately changed if inherited

        # calculate all scalings between area and exposure

        self._precalculations()

        # now create a likelihood object for the call
        # we pass the current object over as well
        # the likelihood object is opaque to the class and
        # keeps a pointer of the plugin inside so that the current
        # counts, bkg, etc. are always up to date
        # This way, when evaluating the likelihood,
        # no checks are involved because the appropriate
        # noise models are pre-selected

        self._likelihood_evaluator = statistic_lookup[self.observation_noise_model][
            self.background_noise_model
        ](self)

    def _count_errors_initialization(self):
        """
        compute the  count errors for the observed and background spectra


        :return:  (observed_count_errors, background_count errors)
        """

        # if there is not a background the dictionary
        # will crash, so we need to do a small check

        tmp_bkg_count_errors = None

        if self._background_spectrum is not None:

            tmp_bkg_count_errors = self._background_spectrum.count_errors

        count_errors_lookup = {
            "poisson": {
                "poisson": (None, None),
                "gaussian": (None, tmp_bkg_count_errors),
                None: (None, None),
            },
            # gaussian source
            "gaussian": {
                "gaussian": (
                    self._observed_spectrum.count_errors,
                    tmp_bkg_count_errors,
                ),
                None: (self._observed_spectrum.count_errors, None),
            },
        }

        try:

            error_tuple = count_errors_lookup[self._observation_noise_model][
                self._background_noise_model
            ]  # type: tuple

        except (KeyError):

            RuntimeError(
                "The noise combination of source: %s, background: %s  is not supported"
                % (self._observation_noise_model, self._background_noise_model)
            )

        for errors, counts, name in zip(
            error_tuple,
            [self._observed_counts, self._background_counts],
            ["observed", "background"],
        ):

            # if the errors are not None then we want to make sure they make sense
            if errors is not None:

                zero_idx = errors == 0  # type: np.ndarray

                # check that zero error => zero counts
                assert np.all(errors[zero_idx] == counts[zero_idx]), (
                    "Error in %s spectrum: if the error on the background is zero, "
                    "also the expected %s must be zero" % name
                )

        observed_count_errors, background_count_errors = error_tuple

        return observed_count_errors, background_count_errors

    def _probe_noise_models(self):
        """

        probe the noise models




        :return: (observation_noise_model, background_noise_model)
        """

        observation_noise_model, background_noise_model = None, None

        # Now auto-probe the statistic to use
        if self._background_spectrum is not None:

            if self._observed_spectrum.is_poisson:

                self._observed_count_errors = None

                self._observed_counts = self._observed_counts.astype(np.int64)

                if self._background_spectrum.is_poisson:

                    observation_noise_model = "poisson"
                    background_noise_model = "poisson"

                    self._background_counts = self._background_counts.astype(
                        np.int64)

                    assert np.all(
                        self._observed_counts >= 0
                    ), "Error in PHA: negative counts!"

                    if not np.all(self._background_counts >= 0):
                        raise NegativeBackground(
                            "Error in background spectrum: negative counts!"
                        )

                else:

                    observation_noise_model = "poisson"
                    background_noise_model = "gaussian"

                    if not np.all(self._background_counts >= 0):
                        raise NegativeBackground(
                            "Error in background spectrum: negative background!"
                        )

            else:

                if self._background_spectrum.is_poisson:

                    raise NotImplementedError(
                        "We currently do not support Gaussian observation and Poisson background"
                    )

                else:

                    observation_noise_model = "gaussian"
                    background_noise_model = "gaussian"

                    if not np.all(self._background_counts >= 0):
                        raise NegativeBackground(
                            "Error in background spectrum: negative background!"
                        )

        else:

            # this is the case for no background

            self._background_counts = None
            self._back_count_errors = None
            self._scaled_background_counts = None

            if self._observed_spectrum.is_poisson:

                self._observed_count_errors = None
                self._observed_counts = self._observed_counts.astype(np.int64)

                assert np.all(
                    self._observed_counts >= 0
                ), "Error in PHA: negative counts!"

                assert np.all(
                    self._observed_counts >= 0
                ), "Error in PHA: negative counts!"

                observation_noise_model = "poisson"
                background_noise_model = None

            else:

                observation_noise_model = "gaussian"
                background_noise_model = None

        # Print the auto-probed noise models

        if self._background_plugin is not None:
            log.info(
                "Background modeled from plugin: %s" % self._background_plugin.name
            )

            bkg_noise = self._background_plugin.observation_noise_model

        else:

            bkg_noise = background_noise_model

        log.info("Auto-probed noise models:")
        log.info("- observation: %s" % observation_noise_model)
        log.info("- background: %s" % bkg_noise)

        return observation_noise_model, background_noise_model

    def _background_setup(self,
                          background,
                          observation: BinnedSpectrum):
        """

        :param background: background arguments (spectrum or plugin)
        :param observation: observed spectrum
        :return: (background_spectrum, background_plugin, background_counts, scaled_background_counts)
        """

        # this is only called during once during construction

        # setup up defaults as none

        background_plugin = None
        background_spectrum = None
        background_counts = None
        scaled_background_counts = None

        if background is not None:

            # If this is a plugin created from a background
            # we extract the observed spectrum (it should not have a background...
            #  it is a background)

            # we are explicitly violating duck-typing

            if isinstance(background, SpectrumLike) or isinstance(background, XYLike):

                background_plugin = background

            else:

                # if the background is not a plugin then we need to make sure it is a spectrum
                # and that the spectrum is the same size as the observation

                assert isinstance(
                    background, BinnedSpectrum
                ), "The background spectrum is not an instance of BinnedSpectrum"

                assert observation.n_channels == background.n_channels, (
                    "Data file and background file have different " "number of channels"
                )

                background_spectrum = background  # type: BinnedSpectrum

                background_counts = background_spectrum.counts  # type: np.ndarray

                # this assumes the observed spectrum is already set!

                scaled_background_counts = self._get_expected_background_counts_scaled(
                    background_spectrum
                )  # type: np.ndarray

        return (
            background_spectrum,
            background_plugin,
            background_counts,
            scaled_background_counts,
        )

    def _precalculations(self):
        """
        pre calculate values for speed.

        originally, the plugins were calculating these values on the fly, which was very slow

        :return:
        """

        log.debug("Starting precalculations")

        # area scale factor between background and source
        # and exposure ratio between background and source

        if (self._background_spectrum is None) and (self._background_plugin is None):

            # there is no background so the area scaling is unity

            log.debug("no background set in precalculations")

            self._area_ratio = 1.0
            self._exposure_ratio = 1.0
            self._background_exposure = 1.0
            self._background_scale_factor = None

        else:

            log.debug("background set in precalculations")

            if self._background_plugin is not None:

                log.debug("detected background plugin")

                if isinstance(self._background_plugin, SpectrumLike):

                    # use the background plugin's observed spectrum  and exposure to scale the area and time

                    self._background_scale_factor = (
                        self._background_plugin.observed_spectrum.scale_factor
                    )
                    self._background_exposure = (
                        self._background_plugin.observed_spectrum.exposure
                    )

                else:

                    # in this case, the XYLike data could come from anything, so area scaling is set to unity
                    # TODO: could this be wrong?

                    self._background_scale_factor = self._observed_spectrum.scale_factor

                    # if the background exposure is set in the constructor, then this will scale it, otherwise
                    # this will be unity

                    self._exposure_ratio = (
                        self._background_exposure
                    ) = self._explict_background_exposure

            else:
                # this is the normal case with no background model, get the scale factor directly

                log.debug("this is a normal background observation")

                self._background_scale_factor = self._background_spectrum.scale_factor
                self._background_exposure = self._background_spectrum.exposure

            self._area_ratio = old_div(
                self._observed_spectrum.scale_factor, self._background_scale_factor
            )

            self._exposure_ratio = old_div(
                self._observed_spectrum.exposure, self._background_exposure
            )

        self._total_scale_factor = self._area_ratio * self._exposure_ratio

        log.debug("completed precalculations")
        # deal with background exposure and scale factor
        # we run through this separately to

    @property
    def exposure(self) -> float:
        """
        Exposure of the source spectrum
        """

        return self._observed_spectrum.exposure

    @property
    def area_ratio(self) -> float:
        """
        :return: ratio between source and background area
        """

        assert (self._background_plugin is not None) or (
            self._background_spectrum
        ) is not None, "No background exists!"

        return self._area_ratio

    @property
    def exposure_ratio(self) -> float:
        """

        :return:  ratio between source and background exposure
        """
        assert (self._background_plugin is not None) or (
            self._background_spectrum
        ) is not None, "No background exists!"

        return self._exposure_ratio

    @property
    def scale_factor(self) -> float:
        """
        Ratio between the source and the background exposure and area

        :return:
        """

        # if (self._background_spectrum is None) and (self._background_plugin is None):
        #     return 1
        #
        # return self._observed_spectrum.exposure / self.background_exposure * self._observed_spectrum.scale_factor / self.background_scale_factor
        assert (self._background_plugin is not None) or (
            self._background_spectrum
        ) is not None, "No background exists!"

        return self._total_scale_factor

    @property
    def background_exposure(self) -> float:
        """
        Exposure of the background spectrum, if present
        """

        return self._background_exposure

    @property
    def background_scale_factor(self) -> float:
        """
        The background scale factor

        :return:
        """

        return self._background_scale_factor

    @property
    def background_spectrum(self) -> BinnedSpectrum:

        assert (
            self._background_spectrum is not None
        ), "This SpectrumLike instance has no background"

        return self._background_spectrum

    @property
    def background_plugin(self):
        # type: () -> SpectrumLike

        return self._background_plugin

    @property
    def observed_spectrum(self) -> BinnedSpectrum:

        return self._observed_spectrum

    @classmethod
    def _get_synthetic_plugin(cls,
                              observation: BinnedSpectrum,
                              background,
                              source_function, are_contiguous=False):

        speclike_gen = cls("generator", observation, background,
                           verbose=False)

        pts = PointSource("fake", 0.0, 0.0, source_function)

        model = Model(pts)

        speclike_gen.set_model(model)

        return speclike_gen

    @staticmethod
    def _build_fake_observation(
        fake_data, channel_set, source_errors, source_sys_errors, is_poisson, **kwargs
    ) -> BinnedSpectrum:
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

        observation = BinnedSpectrum(
            fake_data,
            exposure=1.0,
            ebounds=channel_set.edges,
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
    def from_background(cls,
                        name: str,
                        spectrum_like,
                        verbose: bool = True):
        """
        Extract a SpectrumLike plugin from the background of another SpectrumLike (or subclass) instance


        :param name: name of the extracted_plugin
        :param spectrum_like: plugin with background to extract
        :param verbose: if the plugin should be verbose
        :return: SpectrumLike instance from the background
        """

        log.debug("creating new spectrumlike from background")

        background_only_spectrum = copy.deepcopy(
            spectrum_like.background_spectrum)

        background_spectrum_like = SpectrumLike(
            name, observation=background_only_spectrum, background=None, verbose=verbose
        )

        return background_spectrum_like

    @classmethod
    def from_function(
        cls,
        name: str,
        source_function,
        energy_min,
        energy_max,
        source_errors=None,
        source_sys_errors=None,
        background_function=None,
        background_errors=None,
        background_sys_errors=None,
        **kwargs,
    ):
        """

        Construct a simulated spectrum from a given source function and (optional) background function. If source and/or background errors are not supplied, the likelihood is assumed to be Poisson.

        :param name: simulkated data set name
        :param source_function: astromodels function
        :param energy_min: array of low energy bin edges
        :param energy_max: array of high energy bin edges
        :param source_errors: (optional) gaussian source errors
        :param source_sys_errors: (optional) systematic source errors
        :param background_function: (optional) astromodels background function
        :param background_errors: (optional) gaussian background errors
        :param background_sys_errors: (optional) background systematic errors
        :return: simulated SpectrumLike plugin
        """

        log.debug("creating new spectrumlike from function")

        channel_set = ChannelSet.from_starts_and_stops(energy_min, energy_max)

        # this is just for construction

        fake_data = np.ones(len(energy_min))

        if source_errors is None:

            is_poisson = True

        else:

            assert len(source_errors) == len(
                energy_min
            ), "source error array is not the same dimension as the energy array"

            is_poisson = False

        if source_sys_errors is not None:
            assert len(source_sys_errors) == len(
                energy_min
            ), "background  systematic error array is not the same dimension as the energy array"

        # call the class dependent observation builder

        observation = cls._build_fake_observation(
            fake_data,
            channel_set,
            source_errors,
            source_sys_errors,
            is_poisson,
            **kwargs,
        )

        if background_function is not None:

            fake_background = np.ones(len(energy_min))

            if background_errors is None:

                is_poisson = True

            else:

                assert len(background_errors) == len(
                    energy_min
                ), "background error array is not the same dimension as the energy array"

                is_poisson = False

            if background_sys_errors is not None:
                assert len(background_sys_errors) == len(
                    energy_min
                ), "background  systematic error array is not the same dimension as the energy array"

            tmp_background = BinnedSpectrum(
                fake_background,
                exposure=1.0,
                ebounds=channel_set.edges,
                count_errors=background_errors,
                sys_errors=background_sys_errors,
                quality=None,
                scale_factor=1.0,
                is_poisson=is_poisson,
                mission="fake_mission",
                instrument="fake_instrument",
                tstart=0.0,
                tstop=1.0,
            )

            # now we have to generate the background counts
            # we treat the background as a simple observation with no
            # other background

            background_gen = SpectrumLike(
                "generator", tmp_background, None, verbose=False
            )

            pts_background = PointSource(
                "fake_background", 0.0, 0.0, background_function
            )

            background_model = Model(pts_background)

            background_gen.set_model(background_model)

            sim_background = background_gen.get_simulated_dataset("fake")

            background = sim_background._observed_spectrum

        else:

            background = None

        generator = cls._get_synthetic_plugin(
            observation, background, source_function,

        )  # type: SpectrumLike

        return generator.get_simulated_dataset(name)

    def assign_to_source(self,
                         source_name: str) -> None:
        """
        Assign these data to the given source (instead of to the sum of all sources, which is the default)

        :param source_name: name of the source (must be contained in the likelihood model)
        :return: none
        """

        if self._like_model is not None:
            assert source_name in self._like_model.sources, (
                "Source %s is not contained in " "the likelihood model" % source_name
            )

        self._source_name = source_name

    @property
    def likelihood_model(self) -> Model:

        assert self._like_model is not None, (
            "plugin %s does not have a likelihood model" % self._name
        )

        return self._like_model

    def get_pha_files(self) -> dict:

        info = {}

        # we want to pass copies so that
        # the user doesn't grab the instance
        # and try to modify things. protection
        info["pha"] = copy.copy(self._observed_spectrum)

        if self._background_spectrum is not None:
            info["bak"] = copy.copy(self._background_spectrum)

        return info

    def set_active_measurements(self, *args, **kwargs) -> None:
        """
        Set the measurements to be used during the analysis. Use as many ranges as you need, and you can specify
        either energies or channels to be used.

        NOTE to Xspec users: while XSpec uses integers and floats to distinguish between energies and channels
        specifications, 3ML does not, as it would be error-prone when writing scripts. Read the following documentation
        to know how to achieve the same functionality.

        * Energy selections:

        They are specified as 'emin-emax'. Energies are in keV. Example:

        set_active_measurements('10-12.5','56.0-100.0')

        which will set the energy range 10-12.5 keV and 56-100 keV to be
        used in the analysis. Note that there is no difference in saying 10 or 10.0.

        * Channel selections:

        They are specified as 'c[channel min]-c[channel max]'. Example:

        set_active_measurements('c10-c12','c56-c100')

        This will set channels 10-12 and 56-100 as active channels to be used in the analysis

        * Mixed channel and energy selections:

        You can also specify mixed energy/channel selections, for example to go from 0.2 keV to channel 20 and from
        channel 50 to 10 keV:

        set_active_measurements('0.2-c10','c50-10')

        * Use all measurements (i.e., reset to initial state):

        Use 'all' to select all measurements, as in:

        set_active_measurements('all')

        Use 'reset' to return to native PHA quality from file, as in:

        set_active_measurements('reset')


        * Exclude measurements:

        Excluding measurements work as selecting measurements, but with the "exclude" keyword set to the energies and/or
        channels to be excluded. To exclude between channel 10 and 20 keV and 50 keV to channel 120 do:

        set_active_measurements(exclude=["c10-20", "50-c120"])

        * Select and exclude:

        Call this method more than once if you need to select and exclude. For example, to select between 0.2 keV and
        channel 10, but exclude channel 30-50 and energy , do:

        set_active_measurements("0.2-c10",exclude=["c30-c50"])

        * Using native PHA quality:

        To simply add or exclude channels from the native PHA, one can use the use_quailty
        option:

        set_active_measurements(
            "0.2-c10",exclude=["c30-c50"], use_quality=True)

        This translates to including the channels from 0.2 keV - channel 10, exluding channels
        30-50 and any channels flagged BAD in the PHA file will also be excluded.



        :param args:
        :param exclude: (list) exclude the provided channel/energy ranges
        :param use_quality: (bool) use the native quality on the PHA file (default=False)
        :return:
        """

        # To implement this we will use an array of boolean index,
        # which will filter
        # out the non-used channels during the logLike

        # Now build the new mask: values for which the mask is 0 will be masked

        # We will build the high res mask even if we are
        # already rebinned so that it can be saved

        assert self._rebinner is None, (
            "You cannot select active measurements if you have a rebinning active. "
            "Remove it first with remove_rebinning"
        )

        if "use_quality" in kwargs:

            use_quality = kwargs.pop("use_quality")

        else:

            use_quality = False

        if use_quality:

            # Start with quality mask. This means that channels
            # marked good by quality will be used unless exluded in the arguments
            # and channels marked bad by quality will be excluded unless included
            # by the arguments

            self._mask = self._observed_spectrum.qaulity.good

        else:

            # otherwise, we will start out with all channels deselected
            # and turn the on/off by the arguments

            self._mask = np.zeros(
                self._observed_spectrum.n_channels, dtype=bool)

        if "all" in args:

            # Just make sure than no further selections have been made.

            assert (
                len(args) == 1
            ), "If you specify 'all', you cannot specify more than one energy range."

            # Convert the mask to all True (we use all channels)
            self._mask[:] = True

        elif "reset" in args:

            # Convert the to native PHA masking specified in quality

            assert (
                len(args) == 1
            ), "If you specify 'reset', you cannot specify more than one energy range."

            self._mask = self._observed_spectrum.quality.good

        else:

            for arg in args:

                selections = dash_separated_string_to_tuple(arg)

                # We need to find out if it is a channel or and energy being requested

                idx = np.empty(2, dtype=int)
                for i, s in enumerate(selections):

                    if s[0].lower() == "c":

                        assert (
                            int(s[1:]) <= self._observed_spectrum.n_channels
                        ), "%s is larger than the number of channels: %d" % (
                            s,
                            self._observed_spectrum.n_channels,
                        )
                        idx[i] = int(s[1:])

                    else:

                        idx[i] = self._observed_spectrum.containing_bin(
                            float(s))

                assert idx[0] < idx[1], (
                    "The channel and energy selection (%s) are out of order and translates to %s-%s"
                    % (selections, idx[0], idx[1])
                )

                # we do the opposite of the exclude command!
                self._mask[idx[0]: idx[1] + 1] = True

                log.info(
                    "Range %s translates to channels %s-%s" % (
                        arg, idx[0], idx[1])
                )

        # If you are just excluding channels
        if len(args) == 0:
            self._mask[:] = True

        if "exclude" in kwargs:

            exclude = list(kwargs.pop("exclude"))

            for arg in exclude:

                selections = dash_separated_string_to_tuple(arg)

                # We need to find out if it is a channel or and energy being requested

                idx = np.empty(2, dtype=int)
                for i, s in enumerate(selections):

                    if s[0].lower() == "c":

                        assert (
                            int(s[1:]) <= self._observed_spectrum.n_channels
                        ), "%s is larger than the number of channels: %d" % (
                            s,
                            self._observed_spectrum.n_channels,
                        )
                        idx[i] = int(s[1:])

                    else:

                        idx[i] = self._observed_spectrum.containing_bin(
                            float(s))

                assert idx[0] < idx[1], (
                    "The channel and energy selection (%s) are out of order and translate to %s-%s"
                    % (selections, idx[0], idx[1])
                )

                # we do the opposite of the exclude command!
                self._mask[idx[0]: idx[1] + 1] = False

                log.info(
                    "Range %s translates to excluding channels %s-%s"
                    % (arg, idx[0], idx[1])
                )

            log.info(
                "Now using %s channels out of %s"
                % (np.sum(self._mask), self._observed_spectrum.n_channels)
            )

        # Apply the mask
        self._apply_mask_to_original_vectors()

        # if the user did not specify use_quality, they may have selected channels which
        # are marked BAD (5) in the native PHA file. We want to warn them in this case only (or maybe in all cases?)

        if not use_quality:

            number_of_native_good_channels = sum(
                self._observed_spectrum.quality.good)
            number_of_user_good_channels = sum(self._mask)

            if number_of_user_good_channels > number_of_native_good_channels:

                # we have more good channels than specified in the PHA file
                # so we need to figure out which channels these are where excluded

                deselected_channels = []
                for i in range(self._observed_spectrum.n_channels):

                    if self._observed_spectrum.quality.bad[i] and self._mask[i]:
                        deselected_channels.append(i)

                log.warning(
                    "You have opted to use channels which are flagged BAD in the PHA file."
                )

                log.warning(
                    "These channels are: %s"
                    % (", ".join([str(ch) for ch in deselected_channels]))
                )

    def _apply_mask_to_original_vectors(self):

        # Apply the mask

        self._current_observed_counts = self._observed_counts[self._mask]

        if self._observed_count_errors is not None:
            self._current_observed_count_errors = self._observed_count_errors[
                self._mask
            ]

        if self._background_spectrum is not None:

            self._current_background_counts = self._background_counts[self._mask]
            self._current_scaled_background_counts = self._scaled_background_counts[
                self._mask
            ]

            if self._back_count_errors is not None:
                self._current_back_count_errors = self._back_count_errors[self._mask]

    @contextmanager
    def _without_mask_nor_rebinner(self):

        # Store mask and rebinner for later use

        mask = self._mask
        rebinner = self._rebinner

        # Clean mask and rebinning

        self.remove_rebinning()
        self.set_active_measurements("all")

        # Execute whathever

        yield

        # Restore mask and rebinner (if any)

        self._mask = mask

        if rebinner is not None:

            # There was a rebinner, use it. Note that the rebinner applies the mask by itself

            self._apply_rebinner(rebinner)

        else:

            # There was no rebinner, so we need to explicitly apply the mask

            self._apply_mask_to_original_vectors()

    def get_simulated_dataset(self, new_name=None, **kwargs):
        """
        Returns another Binned instance where data have been obtained by randomizing the current expectation from the
        model, as well as from the background (depending on the respective noise models)

        :return: an BinnedSpectrum or child instance
        """

        assert (
            self._like_model is not None
        ), "You need to set up a model before randomizing"

        # Keep track of how many syntethic datasets we have generated

        self._n_synthetic_datasets += 1

        # Generate a name for the new dataset if needed
        if new_name is None:
            new_name = "%s_sim_%i" % (self.name, self._n_synthetic_datasets)

        # Generate randomized data depending on the different noise models

        # We remove the mask temporarily because we need the various elements for all channels. We will restore it
        # at the end

        original_mask = np.array(self._mask, copy=True)
        original_rebinner = self._rebinner

        with self._without_mask_nor_rebinner():

            # Get the source model for all channels (that's why we don't use the .folded_model property)

            source_model_counts = self._evaluate_model() * self.exposure

            # The likelihood evaluator keeps track of the proper likelihood needed to randomize
            # quantities. It properly returns None if needed. This avoids multiple checks and dupilcate
            # code for the MANY cases we can have. As new cases are added, this code will adapt.

            randomized_source_counts = (
                self._likelihood_evaluator.get_randomized_source_counts(
                    source_model_counts
                )
            )
            randomized_source_count_err = (
                self._likelihood_evaluator.get_randomized_source_errors()
            )
            randomized_background_counts = (
                self._likelihood_evaluator.get_randomized_background_counts()
            )
            randomized_background_count_err = (
                self._likelihood_evaluator.get_randomized_background_errors()
            )

            # create new source and background spectra
            # the children of BinnedSpectra must properly override the new_spectrum
            # member so as to build the appropriate spectrum type. All parameters of the current
            # spectrum remain the same except for the rate and rate errors

            # the profile likelihood automatically adjust the background spectrum to the
            # same exposure and scale as the observation
            # therefore, we must  set the background simulation to have the exposure and scale
            # of the observation

            new_observation = self._observed_spectrum.clone(
                new_counts=randomized_source_counts,
                new_count_errors=randomized_source_count_err,
                new_scale_factor=1.0,
            )

            if self._background_spectrum is not None:

                new_background = self._background_spectrum.clone(
                    new_counts=randomized_background_counts,
                    new_count_errors=randomized_background_count_err,
                    new_exposure=self._observed_spectrum.exposure,  # because it was adjusted
                    new_scale_factor=1.0,  # because it was adjusted
                )

            elif self._background_plugin is not None:

                new_background = self._likelihood_evaluator.synthetic_background_plugin

            else:

                new_background = None

            # Now create another instance of BinnedSpectrum with the randomized data we just generated
            # notice that the _new member is a classmethod
            # (we use verbose=False to avoid many messages when doing many simulations)
            new_spectrum_plugin = self._new_plugin(
                name=new_name,
                observation=new_observation,
                background=new_background,
                verbose=False,
                **kwargs,
            )

            # Apply the same selections as the current data set
            if original_rebinner is not None:

                # Apply rebinning, which also applies the mask
                new_spectrum_plugin._apply_rebinner(original_rebinner)

            else:

                # Only apply the mask
                new_spectrum_plugin._mask = original_mask
                new_spectrum_plugin._apply_mask_to_original_vectors()

            # We want to store the simulated parameters so that the user
            # can recall them later

            new_spectrum_plugin._simulation_storage = clone_model(
                self._like_model)

            # TODO: nuisance parameters

            return new_spectrum_plugin

    @classmethod
    def _new_plugin(cls, *args, **kwargs):
        """
        allows for constructing a new plugin of the appropriate
        type in conjunction with the Spectrum.clone method.
        It is used for example in get_simulated_dataset

        new_background = self._background_spectrum.clone(new_counts=randomized_background_counts,
                                                  new_count_errors=randomized_background_count_err)


        new_spectrum_plugin = self._new_plugin(name=new_name,
                                               observation=new_observation,
                                               background=new_background,
                                               verbose=self._verbose,
                                               **kwargs)


        :param args:
        :param kwargs:
        :return:
        """

        return cls(*args, **kwargs)

    @property
    def simulated_parameters(self):
        """
        Return the simulated dataset parameters
        :return: a likelihood model copy
        """

        assert self._simulation_storage is not None, "This is not a simulated data set"

        return self._simulation_storage

    def rebin_on_background(self, min_number_of_counts):
        """
        Rebin the spectrum guaranteeing the provided minimum number of counts in each background bin. This is usually
        required for spectra with very few background counts to make the Poisson profile likelihood meaningful.
        Of course this is not relevant if you treat the background as ideal, nor if the background spectrum has
        Gaussian errors.

        The observed spectrum will be rebinned in the same fashion as the background spectrum.

        To neutralize this completely, use "remove_rebinning"

        :param min_number_of_counts: the minimum number of counts in each bin
        :return: none
        """

        # NOTE: the rebinner takes care of the mask already

        assert (
            self._background_spectrum is not None
        ), "This data has no background, cannot rebin on background!"

        rebinner: Rebinner = Rebinner(self._background_counts,
                                      min_number_of_counts, self._mask)

        if rebinner.n_bins < len(self._mask):

            # only for the PHASpectrum subclass do we need to update the the grouping
            if isinstance(self._observed_spectrum, PHASpectrum):
                self._observed_spectrum.set_ogip_grouping(rebinner.grouping)
                self._background_spectrum.set_ogip_grouping(rebinner.grouping)

            self._apply_rebinner(rebinner)

        else:

            log.info("rebinning had no effect")

    def rebin_on_source(self, min_number_of_counts):
        """
        Rebin the spectrum guaranteeing the provided minimum number of counts in each source bin.

        To neutralize this completely, use "remove_rebinning"

        :param min_number_of_counts: the minimum number of counts in each bin
        :return: none
        """

        # NOTE: the rebinner takes care of the mask already

        rebinner = Rebinner(self._observed_counts,
                            min_number_of_counts, self._mask)

        if rebinner.n_bins < len(self._mask):

            # only for the PHASpectrum subclass do we need to update the the grouping
            if isinstance(self._observed_spectrum, PHASpectrum):
                self._observed_spectrum.set_ogip_grouping(rebinner.grouping)

                if self._background_spectrum is not None:
                    self._background_spectrum.set_ogip_grouping(
                        rebinner.grouping)

            self._apply_rebinner(rebinner)

        else:

            log.info("rebinning had no effect")

    def _apply_rebinner(self, rebinner):

        self._rebinner = rebinner

        # Apply the rebinning to everything.
        # NOTE: the output of the .rebin method are the vectors with the mask *already applied*

        (self._current_observed_counts,) = self._rebinner.rebin(
            self._observed_counts)

        if self._observed_count_errors is not None:
            (self._current_observed_count_errors,) = self._rebinner.rebin_errors(
                self._observed_count_errors
            )

        if self._background_spectrum is not None:

            (
                self._current_background_counts,
                self._current_scaled_background_counts,
            ) = self._rebinner.rebin(
                self._background_counts, self._scaled_background_counts
            )

            if self._back_count_errors is not None:
                # NOTE: the output of the .rebin method are the vectors with the mask *already applied*

                (self._current_back_count_errors,) = self._rebinner.rebin_errors(
                    self._back_count_errors
                )

        log.info("Now using %s bins" % self._rebinner.n_bins)

    def remove_rebinning(self):
        """
        Remove the rebinning scheme set with rebin_on_background.

        :return:
        """

        # Restore original vectors with mask applied
        self._apply_mask_to_original_vectors()

        self._rebinner = None

    def _get_expected_background_counts_scaled(self, background_spectrum):
        """
        Get the background counts expected in the source interval and in the source region, based on the observed
        background.

        :return:
        """

        # NOTE : this is called only once during construction!

        # The scale factor is the ratio between the collection area of the source spectrum and the
        # background spectrum. It is used for example for the typical aperture-photometry method used in
        # X-ray astronomy, where the background region has a different size with respect to the source region

        scale_factor = old_div(
            self._observed_spectrum.scale_factor, background_spectrum.scale_factor
        )

        # The expected number of counts is the rate in the background file multiplied by its exposure, renormalized
        # by the scale factor.
        # (see http://heasarc.gsfc.nasa.gov/docs/asca/abc_backscal.html)

        bkg_counts = (
            background_spectrum.rates * self._observed_spectrum.exposure * scale_factor
        )

        return bkg_counts

    @property
    def current_observed_counts(self):
        return self._current_observed_counts

    @property
    def current_background_counts(self):
        return self._current_background_counts

    @property
    def current_scaled_background_counts(self):
        return self._current_scaled_background_counts

    @property
    def current_background_count_errors(self):
        return self._current_back_count_errors

    @property
    def current_observed_count_errors(self):
        return self._current_observed_count_errors

    def _set_background_noise_model(self, new_model):

        # Do not make differences between upper and lower cases
        if new_model is not None:
            new_model = new_model.lower()

            assert (
                new_model in _known_noise_models
            ), "Noise model %s not recognized. " "Allowed models are: %s" % (
                new_model,
                ", ".join(_known_noise_models),
            )

        self._background_noise_model = new_model

        # reset the likelihood

        self._likelihood_evaluator = statistic_lookup[self._observation_noise_model][
            new_model
        ](self)

        log.warning(
            "You are setting the background noise model to something that is not specified in the spectrum.\
         Verify that this makes statistical sense."
        )

    def _get_background_noise_model(self):

        return self._background_noise_model

    background_noise_model = property(
        _get_background_noise_model,
        _set_background_noise_model,
        doc="Sets/gets the noise model for the background spectrum",
    )

    def _set_observation_noise_model(self, new_model):

        # Do not make differences between upper and lower cases
        new_model = new_model.lower()

        assert (
            new_model in _known_noise_models
        ), "Noise model %s not recognized. " "Allowed models are: %s" % (
            new_model,
            ", ".join(_known_noise_models),
        )

        self._observation_noise_model = new_model

        # reset the likelihood

        self._likelihood_evaluator = statistic_lookup[new_model][
            self._background_noise_model
        ](self)

        log.warning(
            "You are setting the observation noise model to something that is not specified in the spectrum.\
                 Verify that this makes statistical sense."
        )

    def _get_observation_noise_model(self):

        return self._observation_noise_model

    observation_noise_model = property(
        _get_observation_noise_model,
        _set_observation_noise_model,
        doc="Sets/gets the noise model for the background spectrum",
    )

    def get_log_like(self, precalc_fluxes: Optional[np.array] = None):
        """
        Calls the likelihood from the pre-setup likelihood evaluator that "knows" of the currently set
        noise models

        :return:
        """

        loglike, _ = self._likelihood_evaluator.get_current_value(
            precalc_fluxes=precalc_fluxes)

        return loglike

    def inner_fit(self):

        return self.get_log_like()

    def set_model(self, likelihoodModel):
        """
        Set the model to be used in the joint minimization.
        """

        # Store likelihood model

        self._like_model = likelihoodModel

        # We assume there are no extended sources, since we cannot handle them here

        assert self._like_model.get_number_of_extended_sources() == 0, (
            "SpectrumLike plugins do not support " "extended sources"
        )

        # check if we set a source name that the source is in the model

        if self._source_name is not None:
            assert self._source_name in self._like_model.sources, (
                "Source %s is not contained in "
                "the likelihood model" % self._source_name
            )

        # Get the differential flux function, and the integral function, with no dispersion,
        # we simply integrate the model over the bins

        differential_flux, integral = self._get_diff_flux_and_integral(self._like_model,
                                                                       integrate_method=self._model_integrate_method)

        self._integral_flux = integral

    def _evaluate_model(self, precalc_fluxes: Optional[np.array] = None):
        """
        Since there is no dispersion, we simply evaluate the model by integrating over the energy bins.
        This can be overloaded to convolve the model with a response, for example


        :return:
        """
        if precalc_fluxes is not None:
            return precalc_fluxes

        elif self._has_contiguous_energies:

            if self._predefined_energies is None:

                return self._integral_flux(self._observed_spectrum.edges)

            else:

                return self._integral_flux()

        else:
            return np.array(
                [
                    self._integral_flux(emin, emax)
                    for emin, emax in self._observed_spectrum.bin_stack
                ]
            )

    def get_model(self,
                  precalc_fluxes: Optional[np.array] = None):
        """
        The model integrated over the energy bins. Note that it only returns the  model for the
        currently active channels/measurements

        :return: array of folded model
        """

        if self._rebinner is not None:

            (model,) = self._rebinner.rebin(
                self._evaluate_model(
                    precalc_fluxes=precalc_fluxes) * self._observed_spectrum.exposure
            )

        else:

            model = (
                self._evaluate_model(precalc_fluxes=precalc_fluxes)[
                    self._mask] * self._observed_spectrum.exposure
            )

        return self._nuisance_parameter.value * model

    def _evaluate_background_model(self):
        """
        Since there is no dispersion, we simply evaluate the model by integrating over the energy bins.
        This can be overloaded to convolve the model with a response, for example


        :return:
        """

        if self._has_contiguous_energies:

            if self._predefined_energies is None:

                return self._background_integral_flux(self._observed_spectrum.edges)

            else:

                return self._background_integral_flux()

        else:
            return np.array(
                [
                    self._background_integral_flux(emin, emax)
                    for emin, emax in self._observed_spectrum.bin_stack
                ])

    def get_background_model(self,
                             without_mask: bool = False):
        """
        The background model integrated over the energy bins. Note that it only returns the  model for the
        currently active channels/measurements

        :return: array of folded model
        """

        if not without_mask:
            if self._rebinner is not None:

                (model,) = self._rebinner.rebin(
                    self._evaluate_background_model() * self._background_exposure
                )

            else:

                model = (
                    self._evaluate_background_model()[self._mask]
                    * self._background_exposure
                )

        else:

            model = self._evaluate_background_model() * self._background_exposure

        # TODO: should I use the constant here?

        # return self._nuisance_parameter.value * model

        return model

    def _get_diff_flux_and_integral(self,
                                    likelihood_model: Model,
                                    integrate_method: str = "simpson") -> Tuple[types.FunctionType,
                                                                                types.FunctionType]:

        assert integrate_method in [
            "simpson", "trapz"], "Only simpson and trapz are valid integral_methods."

        if self._source_name is None:

            n_point_sources = likelihood_model.get_number_of_point_sources()

            # Make a function which will stack all point sources (OGIP do not support spatial dimension)

            def differential_flux(energies):
                fluxes = likelihood_model.get_point_source_fluxes(
                    0, energies, tag=self._tag
                )

                # If we have only one point source, this will never be executed
                for i in range(1, n_point_sources):
                    fluxes += likelihood_model.get_point_source_fluxes(
                        i, energies, tag=self._tag
                    )

                return fluxes

        else:

            # This SpectrumLike dataset refers to a specific source

            # Note that we checked that self._source_name is in the model when the model was set

            try:

                def differential_flux(energies):

                    return likelihood_model.sources[self._source_name](
                        energies, tag=self._tag
                    )

            except KeyError:

                raise KeyError(
                    "This XYLike plugin has been assigned to source %s, "
                    "which does not exist in the current model" % self._source_name
                )

        # The following integrates the diffFlux function using Simpson's rule
        # This assume that the intervals e1,e2 are all small, which is guaranteed
        # for any reasonable response matrix, given that e1 and e2 are Monte-Carlo
        # energies. It also assumes that the function is smooth in the interval
        # e1 - e2 and twice-differentiable, again reasonable on small intervals for
        # decent models. It might fail for models with too sharp features, smaller
        # than the size of the monte carlo interval.

        if integrate_method == "simpson":

            # New way with simpson rule.
            # Make sure to not calculate the model twice for the same energies

            if self._has_contiguous_energies:

                if self._predefined_energies is None:

                    def integral(e_edges):

                        # Make sure we do not calculate the flux two times at the same energy
                        # e_edges = np.append(e1, e2[-1])
                        e_m = (e_edges[1:] + e_edges[:-1])/2.

                        diff_fluxes_edges = differential_flux(e_edges)
                        diff_fluxes_mid = differential_flux(e_m)

                        return _simps(e_edges[:-1], e_edges[1:], diff_fluxes_edges, diff_fluxes_mid)

                else:

                    e_edges = np.array(self._predefined_energies)
                    ee1 = e_edges[:-1]
                    ee2 = e_edges[1:]

                    e_m = (ee1+ee2)/2.

                    def integral():

                        diff_fluxes_edges = differential_flux(e_edges)
                        diff_fluxes_mid = differential_flux(e_m)

                        return _simps(ee1, ee2, diff_fluxes_edges, diff_fluxes_mid)

            else:

                def integral(e1, e2):
                    # single energy values given
                    return (
                        (e2 - e1)
                        / 6.0
                        * (
                            differential_flux(e1)
                            + 4*differential_flux((e2 + e1) / 2.0)
                            + differential_flux(e2)
                        )
                    )

        elif integrate_method == "trapz":

            def integral(e1, e2):
                # Trapz rule
                if isinstance(e1, Iterable):
                    # Energy given as list or array

                    # Make sure we do not calculate the flux two times at the same energy
                    e_edges = np.append(e1, e2[-1])
                    diff_fluxes_edges = differential_flux(e_edges)

                    return _trapz(np.array([diff_fluxes_edges[:-1], diff_fluxes_edges[1:]]).T, np.array([e1, e2]).T)
                else:
                    # single energy values given
                    return _trapz(np.array([differential_flux(e1), differential_flux(e2)]), np.array([e1, e2]))

        return differential_flux, integral

    def use_effective_area_correction(self,
                                      min_value: Union[int, float] = 0.8,
                                      max_value: Union[int, float] = 1.2):
        """
        Activate the use of the effective area correction, which is a multiplicative factor in front of the model which
        might be used to mitigate the effect of intercalibration mismatch between different instruments.

        NOTE: do not use this is you are using only one detector, as the multiplicative constant will be completely
        degenerate with the normalization of the model.

        NOTE2: always keep at least one multiplicative constant fixed to one (its default value), when using this
        with other OGIPLike-type detectors

        :param min_value: minimum allowed value (default: 0.8, corresponding to a 20% - effect)
        :param max_value: maximum allowed value (default: 1.2, corresponding to a 20% + effect
        :return:
        """
        log.info(
            f"{self._name} is using effective area correction (between {min_value} and {max_value})")
        self._nuisance_parameter.free = True
        self._nuisance_parameter.bounds = (min_value, max_value)

        # Use a uniform prior by default

        self._nuisance_parameter.set_uninformative_prior(Uniform_prior)

    def fix_effective_area_correction(self,
                                      value: Union[int, float] = 1):
        """
        Fix the multiplicative factor (see use_effective_area_correction) to the provided value (default: 1)

        :param value: new value (default: 1, i.e., no correction)
        :return:
        """

        self._nuisance_parameter.value = value
        self._nuisance_parameter.fix = True

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
            self._integral_flux = integral

    def set_background_integrate_method(self,
                                        method: str):
        """
        Change the integrate method for the background integration
        :param method: (str) which method should be used (simpson or trapz)
        """
        assert method in [
            "simpson", "trapz"], "Only simpson and trapz are valid intergate methods."
        self._background_integrate_method = method
        log.info(
            f"{self._name} changing background integration method to {method}")

        # if background_plugin is set, update the integral function
        if self._background_plugin is not None:
            differential_flux, integral = self._get_diff_flux_and_integral(self._background_plugin.likelihood_model,
                                                                           integrate_method=method)
            self._background_integral_flux = integral

    @property
    def mask(self):
        """
        The channel mask
        :return:
        """

        return self._mask

    @property
    def tstart(self):
        return self._tstart

    @property
    def tstop(self):
        return self._tstop

    @property
    def expected_model_rate(self):

        return self._evaluate_model() * self._nuisance_parameter.value

    @property
    def observed_counts(self):
        """

        :return: the observed counts
        """

        return self._observed_counts

    @property
    def observed_count_errors(self):
        """

        :return: the observed counts errors
        """

        cnt_err = None

        if self._observation_noise_model == "poisson":

            cnt_err = np.sqrt(self._observed_counts)

        else:

            if self._background_noise_model is None:
                cnt_err = self._observed_count_errors

                # calculate all the correct quantites

        return cnt_err

    @property
    def background_counts(self):
        """

        :return: the observed background counts
        """

        background_counts = None

        if self._observation_noise_model == "poisson":

            if self._background_noise_model == "poisson":

                background_counts = self._background_counts

                # Gehrels weighting, a little bit better approximation when statistic is low
                # (and inconsequential when statistic is high)

            elif self._background_noise_model == "ideal":

                background_counts = self._scaled_background_counts

            elif self._background_noise_model == "gaussian":

                background_counts = self._background_counts

            elif self._background_noise_model is None:

                background_counts = None

            elif self._background_noise_model == "modeled":

                # get the background counts from the background
                # plugin.. NOT SCALED

                background_counts = self.get_background_model(
                    without_mask=True)

            else:

                raise RuntimeError("This is a bug")

        else:

            if self._background_noise_model is None:
                # Observed counts
                background_counts = None

                # calculate all the correct quantites

        return background_counts

    @property
    def background_count_errors(self):
        """

        :return: the observed background count errors
        """

        background_errors = None

        if self._observation_noise_model == "poisson":

            if self._background_noise_model == "poisson":

                # Gehrels weighting, a little bit better approximation when statistic is low
                # (and inconsequential when statistic is high)

                background_errors = 1 + np.sqrt(self._background_counts + 0.75)

            elif self._background_noise_model == "ideal":

                background_errors = np.zeros_like(
                    self._scaled_background_counts)

            elif self._background_noise_model == "gaussian":

                background_errors = self._back_count_errors

            elif self._background_noise_model is None:

                return None

            elif self._background_noise_model == "modeled":

                # get the background count error from the background
                # plugin.. NOT SCALED

                background_errors = np.sqrt(
                    self.get_background_model(without_mask=True)
                )

            else:

                raise RuntimeError("This is a bug")

        else:

            if self._background_noise_model is None:
                background_errors = None

        return background_errors

    @property
    def source_rate(self):
        """
        The source rate of the model. If there is background or a background background plugin present,
        the source is background subtracted, but only for visual purposes. If no background is present,
        then, this is just the observed rate.

        :return: the source rate
        """

        if (self._background_noise_model is not None) or (
            self._background_plugin is not None
        ):

            # since we compare to the model rate... background subtract but with proper propagation
            src_rate = (
                old_div(self.observed_counts, self._observed_spectrum.exposure)
                - (old_div(self.background_counts, self._background_exposure))
                * self._area_ratio
            )

        else:

            # since we compare to the model rate... background subtract but with proper propagation
            src_rate = old_div(self.observed_counts,
                               self._observed_spectrum.exposure)

        return src_rate

    @property
    def source_rate_error(self):
        """
        The source rate error of the model. If there is background or a background background plugin present,
        the source is background subtracted, but only for visual purposes. If no background is present,
        then, this is just the observed rate.

        :return: the source rate error
        """

        if (self._background_noise_model is not None) or (
            self._background_plugin is not None
        ):

            src_rate_err = np.sqrt(
                (old_div(self.observed_count_errors, self._observed_spectrum.exposure))
                ** 2
                + (
                    (old_div(self.background_count_errors, self._background_exposure))
                    * self._area_ratio
                )
                ** 2
            )

        else:

            src_rate_err = old_div(
                self.observed_count_errors, self._observed_spectrum.exposure
            )

        return src_rate_err

    @property
    def quality(self):

        return self._observed_spectrum.quality

    @property
    def energy_boundaries(self, mask=True):
        """
        Energy boundaries of channels currently in use (rebinned, if a rebinner is active)

        :return: (energy_min, energy_max)
        """

        energies = np.array(self._observed_spectrum.edges)

        energy_min, energy_max = energies[:-1], energies[1:]

        if self._rebinner is not None:
            # Get the rebinned chans. NOTE: these are already masked

            energy_min, energy_max = self._rebinner.get_new_start_and_stop(
                energy_min, energy_max
            )

        else:

            # Apply the mask
            energy_min = energy_min[self._mask]
            energy_max = energy_max[self._mask]

        return energy_min, energy_max

    @property
    def n_data_points(self):

        if self._rebinner is not None:

            return self._rebinner.n_bins

        else:

            return int(self._mask.sum())

    @property
    def significance(self):
        """
        :return: the significance of the data over background
        """

        sig_obj = Significance(
            Non=self._observed_spectrum.total_count,
            Noff=self._background_spectrum.total_count
            if self._background_spectrum is not None
            else None,
            alpha=self._total_scale_factor,
        )

        if self._background_spectrum is not None:
            if (
                self._observed_spectrum.is_poisson
                and self._background_spectrum.is_poisson
            ):

                # use simple li & ma
                significance = sig_obj.li_and_ma()

            elif (
                self._observed_spectrum.is_poisson
                and not self._background_spectrum.is_poisson
            ):

                significance = sig_obj.li_and_ma_equivalent_for_gaussian_background(
                    self._background_spectrum.total_count_error
                )

            else:

                raise NotImplementedError(
                    "We haven't put in other significances yet")
        else:
            log.warning(
                "Significance with no background is not yet computed accurately"
            )
            significance = [np.NaN]

        return significance[0]

    @property
    def significance_per_channel(self):
        """
        :return: the significance of the data over background per channel
        """

        with np.errstate(divide="ignore", invalid="ignore"):

            sig_obj = Significance(
                Non=self._current_observed_counts,
                Noff=self._current_background_counts,
                alpha=self._total_scale_factor,
            )

            if (
                self._observed_spectrum.is_poisson
                and self._background_spectrum.is_poisson
            ):

                # use simple li & ma
                significance = sig_obj.li_and_ma()

            elif (
                self._observed_spectrum.is_poisson
                and not self._background_spectrum.is_poisson
            ):

                significance = sig_obj.li_and_ma_equivalent_for_gaussian_background(
                    self._current_back_count_errors
                )

            else:

                raise NotImplementedError(
                    "We haven't put in other significances yet")

            return significance

    def write_pha(self):

        raise NotImplementedError("this is in progress")

        # we just need to make a diagonal response and then follow the example in dispersion like

    def view_count_spectrum(
        self,
        plot_errors: bool = True,
        show_bad_channels: bool = True,
        show_warn_channels: bool = False,
        significance_level: bool = None,
        scale_background: bool = True,
    ) -> matplotlib.figure.Figure:
        """
        View the count and background spectrum. Useful to check energy selections.
        :param plot_errors: plot errors on the counts
        :param show_bad_channels: (bool) show which channels are bad in the native PHA quality
        :return:
        """

        if sum(self._mask) == 0:
            raise RuntimeError(
                "There are no active channels selected to plot!")

        # adding the rebinner: j. michael.

        # In the future read colors from config file

        # First plot the counts

        # find out the type of observation

        modeled_label = " "

        if self._observation_noise_model == "poisson":

            # Observed counts
            observed_counts = copy.copy(self._current_observed_counts)

            cnt_err = np.sqrt(observed_counts)

            if self._background_noise_model == "poisson":

                background_counts = copy.copy(self._current_background_counts)

                background_errors = np.sqrt(background_counts)

            elif self._background_noise_model == "ideal":

                background_counts = copy.copy(
                    self._current_scaled_background_counts)

                background_errors = np.zeros_like(background_counts)

            elif self._background_noise_model == "gaussian":

                background_counts = copy.copy(self._current_background_counts)

                background_errors = copy.copy(self._current_back_count_errors)

            elif self._background_noise_model is None:

                if self._background_plugin is None:

                    observed_counts = copy.copy(self._current_observed_counts)
                    background_counts = np.zeros(observed_counts.shape)
                    background_errors = np.zeros(observed_counts.shape)

                else:

                    raise RuntimeError("This is a bug")

                    # we will show the modeled counts

                    background_counts = self.get_background_model()
                    background_errors = np.sqrt(background_counts)

                    modeled_label = "Modeled "

            elif self._background_noise_model == "modeled":

                background_counts = self.get_background_model()
                background_errors = np.sqrt(background_counts)

                modeled_label = "Modeled "

            else:

                raise RuntimeError("This is a bug")

                # convert to rates, ugly, yes

            #            background_counts /= self._background_exposure
            #            background_errors /= self._background_exposure

            background_rate = background_counts / self._background_exposure
            background_rate_errors = background_errors / self._background_exposure

        # Gaussian observation
        else:

            if self._background_noise_model is None:
                observed_counts = copy.copy(self._current_observed_counts)

                background_counts = np.zeros(
                    observed_counts.shape, dtype=np.int64)
                background_errors = np.zeros(
                    observed_counts.shape, dtype=np.int64)

                background_rate = np.zeros(observed_counts.shape)

                background_rate_errors = np.zeros(observed_counts.shape)

                cnt_err = copy.copy(self._current_observed_count_errors)

        # convert to rates, ugly, yes

        observed_rates = observed_counts / self._observed_spectrum.exposure
        rate_err = cnt_err / self._observed_spectrum.exposure
        # observed_counts /= self._observed_spectrum.exposure
        #        cnt_err /= self._observed_spectrum.exposure

        if scale_background:

            background_rate *= self._area_ratio
            background_rate_errors *= self._area_ratio

            background_label = "Scaled %sBackground" % modeled_label

        else:

            background_label = "%sBackground" % modeled_label

        # Make the plots
        fig, ax = plt.subplots()

        # Get the energy boundaries

        energy_min, energy_max = self.energy_boundaries

        energy_width = energy_max - energy_min

        # plot counts and background for the currently selected data

        channel_plot(
            ax,
            energy_min,
            energy_max,
            observed_rates,
            color=threeML_config["ogip"]["counts color"],
            lw=1.5,
            alpha=1,
            label="Total",
        )

        if not np.all(background_rate == 0):

            channel_plot(
                ax,
                energy_min,
                energy_max,
                background_rate,
                color=threeML_config["ogip"]["background color"],
                alpha=0.8,
                label=background_label,
            )

        mean_chan = np.mean([energy_min, energy_max], axis=0)

        # if asked, plot the errors

        if plot_errors:
            ax.errorbar(
                mean_chan,
                old_div(observed_rates, energy_width),
                yerr=old_div(rate_err, energy_width),
                fmt="",
                # markersize=3,
                linestyle="",
                elinewidth=0.7,
                alpha=0.9,
                capsize=0,
                # label=data._name,
                color=threeML_config["ogip"]["counts color"],
            )

            if self._background_noise_model is not None:
                ax.errorbar(
                    mean_chan,
                    old_div(background_rate, energy_width),
                    yerr=old_div(background_rate_errors, energy_width),
                    fmt="",
                    # markersize=3,
                    linestyle="",
                    elinewidth=0.7,
                    alpha=0.9,
                    capsize=0,
                    # label=data._name,
                    color=threeML_config["ogip"]["background color"],
                )

        # Now plot and fade the non-used channels
        non_used_mask = ~self._mask

        if True:  # non_used_mask.sum() > 0:

            # Get un-rebinned versions of all arrays, so we can directly apply the mask
            energy_min_unrebinned, energy_max_unrebinned = (
                np.array(self._observed_spectrum.starts),
                np.array(self._observed_spectrum.stops),
            )
            energy_width_unrebinned = energy_max_unrebinned - energy_min_unrebinned
            observed_rate_unrebinned = old_div(
                self._observed_counts, self.exposure)
            observed_rate_unrebinned_err = old_div(
                np.sqrt(self._observed_counts), self.exposure
            )

            if non_used_mask.sum() > 0:

                channel_plot(
                    ax,
                    energy_min_unrebinned[non_used_mask],
                    energy_max_unrebinned[non_used_mask],
                    observed_rate_unrebinned[non_used_mask],
                    color=threeML_config["ogip"]["counts color"],
                    lw=1.5,
                    alpha=1,
                )

            if self._background_noise_model is not None:

                if self._background_noise_model == "modeled":

                    background_rate_unrebinned = (
                        self.get_background_model(without_mask=True)
                        / self._background_exposure
                    )
                    background_rate_unrebinned_err = (
                        np.sqrt(self.get_background_model(without_mask=True))
                        / self._background_exposure
                    )

                else:

                    background_rate_unrebinned = old_div(
                        self._background_counts, self.background_exposure
                    )
                    background_rate_unrebinned_err = old_div(
                        np.sqrt(self._background_counts), self.background_exposure
                    )

                if non_used_mask.sum() > 0:

                    channel_plot(
                        ax,
                        energy_min_unrebinned[non_used_mask],
                        energy_max_unrebinned[non_used_mask],
                        background_rate_unrebinned[non_used_mask],
                        color=threeML_config["ogip"]["background color"],
                        alpha=0.8,
                    )
            else:

                background_rate_unrebinned = np.zeros_like(
                    observed_rate_unrebinned)
                background_rate_unrebinned_err = np.zeros_like(
                    observed_rate_unrebinned_err
                )

            if plot_errors:
                mean_chan_unrebinned = np.mean(
                    [energy_min_unrebinned, energy_max_unrebinned], axis=0
                )

                ax.errorbar(
                    mean_chan_unrebinned[non_used_mask],
                    old_div(
                        observed_rate_unrebinned[non_used_mask],
                        energy_width_unrebinned[non_used_mask],
                    ),
                    yerr=old_div(
                        observed_rate_unrebinned_err[non_used_mask],
                        energy_width_unrebinned[non_used_mask],
                    ),
                    fmt="",
                    # markersize=3,
                    linestyle="",
                    elinewidth=0.7,
                    alpha=0.9,
                    capsize=0,
                    # label=data._name,
                    color=threeML_config["ogip"]["counts color"],
                )

                if self._background_noise_model is not None:
                    ax.errorbar(
                        mean_chan_unrebinned[non_used_mask],
                        old_div(
                            background_rate_unrebinned[non_used_mask],
                            energy_width_unrebinned[non_used_mask],
                        ),
                        yerr=old_div(
                            background_rate_unrebinned_err[non_used_mask],
                            energy_width_unrebinned[non_used_mask],
                        ),
                        fmt="",
                        # markersize=3,
                        linestyle="",
                        elinewidth=0.7,
                        alpha=0.9,
                        capsize=0,
                        # label=data._name,
                        color=threeML_config["ogip"]["background color"],
                    )

            # make some nice top and bottom plot ranges

            tmp_bkg = background_rate_unrebinned / energy_width_unrebinned
            tmp_bkg = tmp_bkg[np.isfinite(tmp_bkg)]

            tmp_obs = observed_rate_unrebinned / energy_width_unrebinned
            tmp_obs = tmp_obs[np.isfinite(tmp_obs)]

            top = max([max(tmp_bkg), max(tmp_obs)]) * 1.5

            bottom = (
                min(
                    [
                        min(tmp_bkg),
                        min(tmp_obs),
                    ]
                )
                * 0.8
            )

            # plot the deselected regions

            disjoint_patch_plot(
                ax,
                energy_min_unrebinned,
                energy_max_unrebinned,
                top,
                bottom,
                ~self._mask,
                color="k",
                alpha=0.4,
            )

            # plot the bad quality channels if requested

            if show_bad_channels:

                if sum(self._observed_spectrum.quality.bad) > 0:
                    log.info("bad channels shown in red hatching\n")

                disjoint_patch_plot(
                    ax,
                    energy_min_unrebinned,
                    energy_max_unrebinned,
                    top,
                    bottom,
                    self._observed_spectrum.quality.bad,
                    color="none",
                    edgecolor="#FE3131",
                    hatch="/",
                    alpha=0.95,
                )

            if show_warn_channels:

                if sum(self._observed_spectrum.quality.warn) > 0:
                    log.info("warned channels shown in purple hatching\n")

                disjoint_patch_plot(
                    ax,
                    energy_min_unrebinned,
                    energy_max_unrebinned,
                    top,
                    bottom,
                    self._observed_spectrum.quality.bad,
                    color="none",
                    edgecolor="#C79BFE",
                    hatch="/",
                    alpha=0.95,
                )

            if significance_level is not None:

                log.info(
                    "channels below the significance threshold shown in red\n")

                with np.errstate(invalid="ignore"):
                    significance_mask = (
                        self.significance_per_channel < significance_level
                    )

                disjoint_patch_plot(
                    ax,
                    energy_min_unrebinned,
                    energy_max_unrebinned,
                    top,
                    bottom,
                    significance_mask,
                    color="red",
                    alpha=0.3,
                )

        ax.set_xlabel("Energy (keV)")
        ax.set_ylabel("Rate (counts s$^{-1}$ keV$^{-1}$)")
        ax.set_xlim(
            left=self._observed_spectrum.absolute_start,
            right=self._observed_spectrum.absolute_stop,
        )
        ax.legend()

        return fig

    def __repr__(self):

        return self._output().to_string()

    def _output(self):
        # type: () -> pd.Series

        obs = collections.OrderedDict()

        obs["n. channels"] = self._observed_spectrum.n_channels

        obs["total rate"] = self._observed_spectrum.total_rate

        if not self._observed_spectrum.is_poisson:
            obs["total rate error"] = self._observed_spectrum.total_rate_error

        if self._background_spectrum is not None:
            obs["total bkg. rate"] = self._background_spectrum.total_rate
            if not self._background_spectrum.is_poisson:
                obs[
                    "total bkg. rate error"
                ] = self._background_spectrum.total_rate_error
            obs["bkg. exposure"] = self.background_exposure
            obs["bkg. is poisson"] = self._background_spectrum.is_poisson

        obs["exposure"] = self.exposure
        obs["is poisson"] = self._observed_spectrum.is_poisson

        if self._background_plugin is not None:
            obs["background"] = "modeled from plugin %s" % self._background_plugin.name
            obs["significance"] = self.significance
            obs["src/bkg area ratio"] = self._area_ratio
            obs["src/bkg exposure ratio"] = self._exposure_ratio
            obs["src/bkg scale factor"] = self._total_scale_factor

        elif self._background_spectrum is not None:

            obs["background"] = "profiled"
            obs["significance"] = self.significance
            obs["src/bkg area ratio"] = self._area_ratio
            obs["src/bkg exposure ratio"] = self._exposure_ratio
            obs["src/bkg scale factor"] = self._total_scale_factor

        # obs['response'] = self._observed_spectrum.response_file

        return pd.Series(data=obs, index=list(obs.keys()))

    def get_number_of_data_points(self):
        """
        returns the number of active data bins
        :return:
        """

        # the sum of the mask should be the number of data bins in use

        return self._mask.sum()

    def display(self):

        display(self._output().to_frame())

    def __repr__(self):

        return self._output().to_string()

    def _construct_counts_arrays(self,
                                 min_rate: Union[int, float],
                                 ratio_residuals: bool = False,
                                 total_counts: bool = False) -> dict:
        """

        Build new arrays before or after a fit of rebinned data/model
        values. We keep this seperated from the plotting code because
        it is cleaner and allows us to extract these quantites independently

        :param min_rate:
        :param ratio_residuals:
        :param total_counts: Should this construct the total counts as "data". If not, the "data counts" are
        observed-background and the model counts are only source counts. Otherwise "data counts" are observed
        and model counts are source+background
        :return:
        """

        # energy_min, energy_max = self._rsp.ebounds[:-1], self._rsp.ebounds[1:]

        energy_min = np.array(self._observed_spectrum.edges[:-1])
        energy_max = np.array(self._observed_spectrum.edges[1:])

        chan_width = energy_max - energy_min

        # Source model
        expected_model_rate = self.expected_model_rate

        # Observed rate
        observed_rate = old_div(self.observed_counts,
                                self._observed_spectrum.exposure)
        observed_rate_err = old_div(
            self.observed_count_errors, self._observed_spectrum.exposure)

        # Background rate
        if (self._background_noise_model is not None) or (self._background_plugin is not None):
            background_rate = old_div(self.background_counts,
                                      self._background_exposure) *\
                self._area_ratio
            background_rate_err = old_div(self.background_count_errors,
                                          self._background_exposure) *\
                self._area_ratio
        else:
            background_rate = np.zeros(len(observed_rate))
            background_rate_err = np.zeros(len(observed_rate))

        # Create a rebinner if either a min_rate has been given, or if the current data set has no rebinned on its own
        # rebin on expected model rate
        if (min_rate is not NO_REBIN) or (self._rebinner is None):

            this_rebinner = Rebinner(expected_model_rate, min_rate, self._mask)

        else:

            # Use the rebinner already in the data
            this_rebinner = self._rebinner

        # get the rebinned counts
        new_observed_rate, new_model_rate, new_background_rate = \
            this_rebinner.rebin(
                observed_rate, expected_model_rate, background_rate)
        (new_observed_rate_err,) = this_rebinner.rebin_errors(observed_rate_err)
        (new_background_rate_err,) = this_rebinner.rebin_errors(background_rate_err)

        # adjust channels
        new_energy_min, new_energy_max = this_rebinner.get_new_start_and_stop(
            energy_min, energy_max
        )
        new_chan_width = new_energy_max - new_energy_min

        # mean_energy = np.mean([new_energy_min, new_energy_max], axis=0)

        # For each bin find the weighted average of the channel center
        mean_energy = []
        delta_energy = [[], []]
        mean_energy_unrebinned = (energy_max + energy_min) / 2.0

        for e_min, e_max in zip(new_energy_min, new_energy_max):

            # Find all channels in this rebinned bin
            idx = (mean_energy_unrebinned >= e_min) & (
                mean_energy_unrebinned <= e_max)

            # Find the rates for these channels
            r = observed_rate[idx]

            if r.max() == 0:

                # All empty, cannot weight
                this_mean_energy = (e_min + e_max) / 2.0

            else:

                # negative src rates cause the energy mean to
                # go outside of the bounds. So we fix negative rates to
                # zero when computing the mean

                idx_negative = r < 0.0

                r[idx_negative] = 0.0

                # Do the weighted average of the mean energies
                weights = old_div(r, np.sum(r))

                this_mean_energy = np.average(
                    mean_energy_unrebinned[idx], weights=weights
                )

            # Compute "errors" for X (which aren't really errors, just to mark the size of the bin)

            delta_energy[0].append(this_mean_energy - e_min)
            delta_energy[1].append(e_max - this_mean_energy)
            mean_energy.append(this_mean_energy)

        # Residuals

        # we need to get the rebinned counts
        (rebinned_observed_counts,) = this_rebinner.rebin(self.observed_counts)

        (rebinned_observed_count_errors,) = this_rebinner.rebin_errors(
            self.observed_count_errors
        )

        # the rebinned counts expected from the model
        rebinned_model_counts = new_model_rate * self._observed_spectrum.exposure

        # and also the rebinned background

        if self._background_noise_model is not None:

            if False:  # self._background_noise_model == "modeled":

                (rebinned_background_counts,) = this_rebinner.rebin(
                    self.get_background_model()
                )
                (rebinned_background_errors,) = this_rebinner.rebin_errors(
                    np.sqrt(self.get_background_model())
                )

            else:

                (rebinned_background_counts,) = this_rebinner.rebin(
                    self.background_counts
                )
                (rebinned_background_errors,) = this_rebinner.rebin_errors(
                    self.background_count_errors
                )

        else:

            rebinned_background_counts = np.zeros_like(
                rebinned_observed_counts)

        significance_calc = Significance(
            rebinned_observed_counts,
            rebinned_background_counts
            + old_div(rebinned_model_counts, self._total_scale_factor),
            min([self._total_scale_factor, 1.0]),
        )

        # Divide the various cases

        # TODO check this: shoudn't it be obseved-background/model (for the old way) and
        # observed/(model+background) (for the new way). Errors also wrong observed+background error
        if ratio_residuals:
            residuals = old_div(
                (rebinned_observed_counts - rebinned_model_counts),
                rebinned_model_counts,
            )
            residual_errors = old_div(
                rebinned_observed_count_errors, rebinned_model_counts
            )

        else:
            residual_errors = None
            if self._observation_noise_model == "poisson":

                if self._background_noise_model == "poisson":

                    # We use the Li-Ma formula to get the significance (sigma)

                    residuals = significance_calc.li_and_ma()

                elif self._background_noise_model == "ideal":

                    residuals = significance_calc.known_background()

                elif self._background_noise_model == "gaussian":

                    residuals = (
                        significance_calc.li_and_ma_equivalent_for_gaussian_background(
                            rebinned_background_errors
                        )
                    )

                elif self._background_noise_model is None:

                    residuals = significance_calc.known_background()

                elif self._background_noise_model == "modeled":

                    residuals = significance_calc.known_background()

                else:

                    raise RuntimeError("This is a bug")

            else:

                if self._background_noise_model is None:

                    residuals = old_div(
                        (rebinned_observed_counts - rebinned_model_counts),
                        rebinned_observed_count_errors,
                    )

                else:

                    raise NotImplementedError("Not yet implemented")

        # construct a dict with all the new quantities
        # so that we can extract them for plotting

        rebinned_quantities = dict(
            # Rebined
            # observed
            new_observed_rate=new_observed_rate,
            new_observed_rate_err=new_observed_rate_err,
            # background
            new_background_rate=new_background_rate,
            new_background_rate_err=new_background_rate_err,
            # model
            new_model_rate=new_model_rate,
            # New echans
            new_chan_width=new_chan_width,
            new_energy_min=new_energy_min,
            new_energy_max=new_energy_max,
            mean_energy=mean_energy,
            # Residuals
            residuals=residuals,
            residual_errors=residual_errors,
            delta_energy=delta_energy,
            # Unbinned model rate
            expected_model_rate=expected_model_rate,
            # Unbinned echans
            energy_min=energy_min,
            energy_max=energy_max,
            chan_width=chan_width,
        )

        return rebinned_quantities

    def display_model(
        self,
        data_color: str = "k",
        model_color: str = "r",
        background_color: str = "b",
        step: bool = True,
        show_data: bool = True,
        show_residuals: bool = True,
        ratio_residuals: bool = False,
        show_legend: bool = True,
        min_rate: Union[int, float] = 1e-99,
        model_label: Optional[str] = None,
        model_kwargs: Optional[dict] = None,
        data_kwargs: Optional[dict] = None,
        background_label: Optional[str] = None,
        background_kwargs: Optional[dict] = None,
        source_only: bool = True,
        show_background: bool = False,
        **kwargs
    ) -> ResidualPlot:
        """
        Plot the current model with or without the data and the residuals. Multiple models can be plotted by supplying
        a previous axis to 'model_subplot'.

        Example usage:

        fig = data.display_model()

        fig2 = data2.display_model(model_subplot=fig.axes)


        :param data_color: the color of the data
        :param model_color: the color of the model
        :param step: (bool) create a step count histogram or interpolate the model
        :param show_data: (bool) show_the data with the model
        :param show_residuals: (bool) shoe the residuals
        :param ratio_residuals: (bool) use model ratio instead of residuals
        :param show_legend: (bool) show legend
        :param min_rate: the minimum rate per bin
        :param model_label: (optional) the label to use for the model default is plugin name
        :param model_subplot: (optional) axis or list of axes to plot to
        :param model_kwargs: plotting kwargs affecting the plotting of the model
        :param data_kwargs:  plotting kwargs affecting the plotting of the data and residuls
        :return:
        """

        # set up the default plotting

        _default_model_kwargs = dict(color=model_color, alpha=1)

        _default_background_kwargs = dict(
            color=background_color, alpha=1, linestyle="--")

        _default_data_kwargs = dict(
            color=data_color,
            alpha=1,
            fmt=threeML_config["residual plot"]["error marker"],
            markersize=threeML_config["residual plot"]["error marker size"],
            linestyle="",
            elinewidth=threeML_config["residual plot"]["error line width"],
            capsize=0,
        )

        if model_kwargs is not None:

            assert type(model_kwargs) == dict, "model_kwargs must be a dict"

            for k, v in list(model_kwargs.items()):

                if k in _default_model_kwargs:

                    _default_model_kwargs[k] = v

                else:

                    _default_model_kwargs[k] = v

        if data_kwargs is not None:

            assert type(data_kwargs) == dict, "data_kwargs must be a dict"

            for k, v in list(data_kwargs.items()):

                if k in _default_data_kwargs:

                    _default_data_kwargs[k] = v

                else:

                    _default_data_kwargs[k] = v

        if background_kwargs is not None:

            assert type(
                background_kwargs) == dict, "background_kwargs must be a dict"

            for k, v in list(background_kwargs.items()):

                if k in _default_background_kwargs:

                    _default_background_kwargs[k] = v

                else:

                    _default_background_kwargs[k] = v

        if model_label is None:
            model_label = "%s Model" % self._name

        residual_plot = ResidualPlot(
            show_residuals=show_residuals, ratio_residuals=ratio_residuals, **kwargs
        )

        # compute the values for the plotting

        rebinned_quantities = self._construct_counts_arrays(
            min_rate, ratio_residuals)

        if source_only:
            y_label = "Net rate\n(counts s$^{-1}$ keV$^{-1}$)"
            weighted_data = old_div(
                rebinned_quantities["new_observed_rate"] -
                rebinned_quantities["new_background_rate"], rebinned_quantities["new_chan_width"]
            )
            weighted_error = old_div(
                np.sqrt(rebinned_quantities["new_observed_rate_err"]**2 +
                        rebinned_quantities["new_background_rate_err"]**2),
                rebinned_quantities["new_chan_width"]
            )
        else:
            y_label = "Observed rate\n(counts s$^{-1}$ keV$^{-1}$)"
            weighted_data = old_div(
                rebinned_quantities["new_observed_rate"], rebinned_quantities["new_chan_width"]
            )
            weighted_error = old_div(
                rebinned_quantities["new_observed_rate_err"], rebinned_quantities["new_chan_width"]
            )
        # weighted_data = old_div(
        #    rebinned_quantities["new_rate"], rebinned_quantities["new_chan_width"]
        # )
        # weighted_error = old_div(
        #    rebinned_quantities["new_err"], rebinned_quantities["new_chan_width"]
        # )

        residual_plot.add_data(
            rebinned_quantities["mean_energy"],
            weighted_data,
            rebinned_quantities["residuals"],
            residual_yerr=rebinned_quantities["residual_errors"],
            yerr=weighted_error,
            xerr=rebinned_quantities["delta_energy"],
            label=self._name,
            show_data=show_data,
            **_default_data_kwargs,
        )

        if show_background:
            residual_plot.add_model_step(
                rebinned_quantities["new_energy_min"],
                rebinned_quantities["new_energy_max"],
                rebinned_quantities["new_chan_width"],
                rebinned_quantities["new_background_rate"],
                label=background_label,
                **_default_background_kwargs
            )

        # a step historgram
        if step:
            if source_only:
                # only source
                eff_model = rebinned_quantities["new_model_rate"]
            else:
                eff_model = rebinned_quantities["new_model_rate"] + \
                    rebinned_quantities["new_background_rate"]
            residual_plot.add_model_step(
                rebinned_quantities["new_energy_min"],
                rebinned_quantities["new_energy_max"],
                rebinned_quantities["new_chan_width"],
                eff_model,
                label=model_label,
                **_default_model_kwargs,
            )

            # residual_plot.add_model_step(
            #    rebinned_quantities["new_energy_min"],
            #    rebinned_quantities["new_energy_max"],
            #    rebinned_quantities["new_chan_width"],
            #    rebinned_quantities["new_model_rate"],
            #    label=model_label,
            #    **_default_model_kwargs
            # )

        else:

            # We always plot the model un-rebinned here

            # Mask the array so we don't plot the model where data have been excluded
            # y = expected_model_rate / chan_width
            y = np.ma.masked_where(
                ~self._mask,
                old_div(
                    rebinned_quantities["expected_model_rate"],
                    rebinned_quantities["chan_width"],
                ),
            )

            x = np.mean(
                [rebinned_quantities["energy_min"],
                    rebinned_quantities["energy_max"]],
                axis=0,
            )

            residual_plot.add_model(
                x, y, label=model_label, **_default_model_kwargs)

        return residual_plot.finalize(
            xlabel="Energy\n(keV)",
            ylabel=y_label,
            xscale="log",
            yscale="log",
            show_legend=show_legend,
        )


@nb.njit(fastmath=True, cache=True)
def _trapz(x, y):
    return np.trapz(x, y)


@nb.njit(fastmath=True, cache=True)
def _simps(e1, e2, diff_fluxes_edges, diff_fluxes_mid):
    return (e2 - e1) / 6.0 * (
        diff_fluxes_edges[:-1]
        + 4*diff_fluxes_mid
        + diff_fluxes_edges[1:]
    )
