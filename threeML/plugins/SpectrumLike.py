import collections
import copy
from contextlib import contextmanager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from astromodels.core.parameter import Parameter
from astromodels.functions.functions import Uniform_prior
from astromodels.utils.valid_variable import is_valid_variable_name
from astromodels import clone_model

from threeML.io.rich_display import display
from threeML.io.plotting.light_curve_plots import channel_plot, disjoint_patch_plot
from threeML.exceptions.custom_exceptions import custom_warnings, NegativeBackground
from threeML.plugin_prototype import PluginPrototype, set_external_property
from threeML.plugins.OGIP.likelihood_functions import poisson_log_likelihood_ideal_bkg
from threeML.plugins.OGIP.likelihood_functions import poisson_observed_gaussian_background
from threeML.plugins.OGIP.likelihood_functions import poisson_observed_poisson_background
from threeML.plugins.OGIP.likelihood_functions import chi2
from threeML.utils.binner import Rebinner
from threeML.utils.stats_tools import Significance
from threeML.plugins.spectrum.binned_spectrum import BinnedSpectrum
from threeML.plugins.spectrum.pha_spectrum import PHASpectrum

from threeML.config.config import threeML_config

__instrument_name = "General binned spectral data"

# This defines the known noise models for source and/or background spectra
_known_noise_models = ['poisson', 'gaussian', 'ideal']


class SpectrumLike(PluginPrototype):
    def __init__(self, name, observation, background, verbose=True):
        # type: (str, BinnedSpectrum, BinnedSpectrum, bool) -> None
        """
        A plugin for generic spectral data, accepts an observed binned spectrum,
        and a background binned spectrum

        :param name: the plugin name
        :param observation: the observed spectrum
        :param background: the background spectrum
        :param verbose: turn on/off verbose logging
        """

        # Just a toggle for verbosity
        self._verbose = bool(verbose)

        assert is_valid_variable_name(name), "Name %s is not a valid name for a plugin. You must use a name which is " \
                                             "a valid python identifier: no spaces, no operators (+,-,/,*), " \
                                             "it cannot start with a number, no special characters" % name

        assert isinstance(observation,
                          BinnedSpectrum), "The observed spectrum is not an instance of BinnedSpectrum"

        if background is not None:
            assert isinstance(background,
                              BinnedSpectrum), "The background spectrum is not an instance of BinnedSpectrum"

            assert observation.n_channels == background.n_channels, "Data file and background file have different " \
                                                                    "number of channels"

        # Precomputed observed and background counts (for speed)

        self._observed_spectrum = observation  # type: BinnedSpectrum

        self._observed_counts = self._observed_spectrum.counts  # type: np.ndarray

        if background is None:

            self._background_spectrum = background  # type: BinnedSpectrum

        else:

            self._background_spectrum = background  # type: BinnedSpectrum

            self._background_counts = self._background_spectrum.counts  # type: np.ndarray

            self._scaled_background_counts = self._get_expected_background_counts_scaled()  # type: np.ndarray

        # Init everything else to None
        self._like_model = None
        self._rebinner = None

        # Now auto-probe the statistic to use
        if self._background_spectrum is not None:

            if self._observed_spectrum.is_poisson:

                self._observed_count_errors = None

                if self._background_spectrum.is_poisson:

                    self.observation_noise_model = 'poisson'
                    self.background_noise_model = 'poisson'

                    self._back_count_errors = None

                    assert np.all(self._observed_counts >= 0), "Error in PHA: negative counts!"

                    if not np.all(self._background_counts >= 0): raise NegativeBackground(
                        "Error in background spectrum: negative counts!")

                else:

                    self.observation_noise_model = 'poisson'
                    self.background_noise_model = 'gaussian'

                    self._back_count_errors = self._background_spectrum.count_errors  # type: np.ndarray

                    idx = (self._back_count_errors == 0)  # type: np.ndarray

                    assert np.all(self._back_count_errors[idx] == self._background_counts[idx]), \
                        "Error in background spectrum: if the error on the background is zero, " \
                        "also the expected background must be zero"

                    if not np.all(self._background_counts >= 0): raise NegativeBackground(
                        "Error in background spectrum: negative background!")

            else:

                if self._background_spectrum.is_poisson:

                    raise NotImplementedError("We currently do not support Gaussian observation and Poisson background")


                else:

                    self.observation_noise_model = 'gaussian'
                    self.background_noise_model = 'gaussian'

                    self._back_count_errors = self._background_spectrum.count_errors  # type: np.ndarray

                    self._observed_count_errors = self._observed_spectrum.count_errors  # type: np.ndarray

                    idx = (self._back_count_errors == 0)  # type: np.ndarray

                    assert np.all(self._back_count_errors[idx] == self._background_counts[idx]), \
                        "Error in background spectrum: if the error on the background is zero, " \
                        "also the expected background must be zero"

                    if not np.all(self._background_counts >= 0): raise NegativeBackground(
                        "Error in background spectrum: negative background!")

                    idx = (self._observed_count_errors == 0)  # type: np.ndarray

                    assert np.all(self._observed_count_errors[idx] == self._observed_counts[idx]), \
                        "Error in ovserved spectrum: if the error on the observation is zero, " \
                        "also the expected observation must be zero"


        else:

            # this is the case for no background

            self._background_counts = None
            self._back_count_errors = None
            self._scaled_background_counts = None

            if self._observed_spectrum.is_poisson:

                self._observed_count_errors = None

                assert np.all(self._observed_counts >= 0), "Error in PHA: negative counts!"

                self.observation_noise_model = 'poisson'
                self.background_noise_model = None

            else:

                self.observation_noise_model = 'gaussian'
                self.background_noise_model = None

                self._observed_count_errors = self._observed_spectrum.count_errors  # type: np.ndarray

                idx = (self._observed_count_errors == 0)  # type: np.ndarray

                assert np.all(self._observed_count_errors[idx] == self._observed_counts[idx]), \
                    "Error in ovserved spectrum: if the error on the observation is zero, " \
                    "also the expected observation must be zero"




        # Initialize a mask that selects all the data.
        # We will initially use the quality mask for the PHA file
        # and set any quality greater than 0 to False. We want to save
        # the native quality so that we can warn the user if they decide to
        # select channels that were flagged as bad.


        self._mask = np.asarray(np.ones(self._observed_spectrum.n_channels), np.bool)

        # Print the autoprobed noise models
        if self._verbose:
            print("Auto-probed noise models:")
            print("- observation: %s" % self.observation_noise_model)
            print("- background: %s" % self.background_noise_model)

        # Now create the nuisance parameter for the effective area correction, which is fixed
        # by default. This factor multiplies the model so that it can account for calibration uncertainties on the
        # global effective area. By default it is limited to stay within 20%

        self._nuisance_parameter = Parameter("cons_%s" % name, 1.0, min_value=0.8, max_value=1.2, delta=0.05,
                                             free=False, desc="Effective area correction for %s" % name)

        nuisance_parameters = collections.OrderedDict()
        nuisance_parameters[self._nuisance_parameter.name] = self._nuisance_parameter

        super(SpectrumLike, self).__init__(name, nuisance_parameters)

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

        self._tstart = None
        self._tstop = None

        # This is so far not a simulated data set
        self._simulation_storage = None

        # set the mask to the native quality
        self._mask = self._observed_spectrum.quality.good
        # Apply the mask
        self._apply_mask_to_original_vectors()

    def get_pha_files(self):

        info = {}

        # we want to pass copies so that
        # the user doesn't grab the instance
        # and try to modify things. protection
        info['pha'] = copy.copy(self._observed_spectrum)
        info['bak'] = copy.copy(self._background_spectrum)

        return info

    def set_active_measurements(self, *args, **kwargs):
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

        set_active_measurements("0.2-c10",exclude=["c30-c50"], use_quality=True)

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

        assert self._rebinner is None, "You cannot select active measurements if you have a rebinning active. " \
                                       "Remove it first with remove_rebinning"

        if 'use_quality' in kwargs:

            use_quality = kwargs.pop('use_quality')

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

            self._mask = np.zeros(self._observed_spectrum.n_channels, dtype=bool)

        if 'all' in args:

            # Just make sure than no further selections have been made.

            assert len(args) == 1, "If you specify 'all', you cannot specify more than one energy range."

            # Convert the mask to all True (we use all channels)
            self._mask[:] = True



        elif 'reset' in args:

            # Convert the to native PHA masking specified in quality

            assert len(args) == 1, "If you specify 'reset', you cannot specify more than one energy range."

            self._mask = self._observed_spectrum.quality.good

        else:

            for arg in args:

                selections = arg.replace(" ", "").split("-")

                # We need to find out if it is a channel or and energy being requested

                idx = np.empty(2, dtype=int)
                for i, s in enumerate(selections):

                    if s[0].lower() == 'c':

                        assert int(s[
                                   1:]) <= self._observed_spectrum.n_channels, "%s is larger than the number of channels: %d" % (
                            s, self._observed_spectrum.n_channels)
                        idx[i] = int(s[1:])

                    else:

                        idx[i] = self._observed_spectrum.containing_bin(float(s))

                assert idx[0] < idx[
                    1], "The channel and energy selection (%s) are out of order and translates to %s-%s" % (
                    selections, idx[0], idx[1])

                # we do the opposite of the exclude command!
                self._mask[idx[0]:idx[1] + 1] = True

                if self._verbose:
                    print("Range %s translates to channels %s-%s" % (arg, idx[0], idx[1]))

        # If you are just excluding channels
        if len(args) == 0:
            self._mask[:] = True

        if 'exclude' in kwargs:

            exclude = list(kwargs.pop('exclude'))

            for arg in exclude:

                selections = arg.replace(" ", "").split("-")

                # We need to find out if it is a channel or and energy being requested

                idx = np.empty(2, dtype=int)
                for i, s in enumerate(selections):

                    if s[0].lower() == 'c':

                        assert int(s[
                                   1:]) <= self._observed_spectrum.n_channels, "%s is larger than the number of channels: %d" % (
                            s, self._observed_spectrum.n_channels)
                        idx[i] = int(s[1:])

                    else:

                        idx[i] = self._observed_spectrum.containing_bin(float(s))

                assert idx[0] < idx[
                    1], "The channel and energy selection (%s) are out of order and translate to %s-%s" % (
                    selections, idx[0], idx[1])

                # we do the opposite of the exclude command!
                self._mask[idx[0]:idx[1] + 1] = False

                if self._verbose:
                    print("Range %s translates to excluding channels %s-%s" % (arg, idx[0], idx[1]))

        if self._verbose:
            print("Now using %s channels out of %s" % (np.sum(self._mask), self._observed_spectrum.n_channels))

        # Apply the mask
        self._apply_mask_to_original_vectors()

        # if the user did not specify use_quality, they may have selected channels which
        # are marked BAD (5) in the native PHA file. We want to warn them in this case only (or maybe in all cases?)

        if not use_quality:

            number_of_native_good_channels = sum(self._observed_spectrum.quality.good)
            number_of_user_good_channels = sum(self._mask)

            if number_of_user_good_channels > number_of_native_good_channels:

                # we have more good channels than specified in the PHA file
                # so we need to figure out which channels these are where excluded

                deselected_channels = []
                for i in xrange(self._observed_spectrum.n_channels):

                    if self._observed_spectrum.quality.bad[i] and self._mask[i]:
                        deselected_channels.append(i)

                custom_warnings.warn("You have opted to use channels which are flagged BAD in the PHA file.")

                if self._verbose:
                    custom_warnings.warn(
                        "These channels are: %s" % (', '.join([str(ch) for ch in deselected_channels])))

    def _apply_mask_to_original_vectors(self):

        # Apply the mask

        self._current_observed_counts = self._observed_counts[self._mask]

        if self._observed_count_errors is not None:
            self._current_observed_count_errors = self._observed_count_errors[self._mask]

        if self._background_spectrum is not None:

            self._current_background_counts = self._background_counts[self._mask]
            self._current_scaled_background_counts = self._scaled_background_counts[self._mask]

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

        assert self._like_model is not None, "You need to set up a model before randomizing"

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

            # NOTE: we use the unmasked versions because we need to generate ALL data, so that the user can change
            # selections afterwards

            if self._observation_noise_model == 'poisson':

                # We need to generate Poisson variates from the model to get the signal, and from the background
                # to get the new background

                # Now depending on the background noise model, generate randomized values for the background

                if self._background_noise_model == 'poisson':

                    # Since we use a profile likelihood, the background model is conditional on the source model, so let's
                    # get it from the likelihood function
                    _, background_model_counts = self._loglike_poisson_obs_poisson_bkg()

                    # Now randomize the expectations

                    # Randomize expectations for the source

                    randomized_source_counts = np.random.poisson(source_model_counts + background_model_counts)
                    # randomized_source_rate = randomized_source_counts / self.exposure

                    # Randomize expectations for the background

                    randomized_background_counts = np.random.poisson(background_model_counts)
                    # randomized_background_rate = randomized_background_counts / self.background_exposure

                    randomized_background_count_err = None

                    randomized_source_count_err = None

                elif self._background_noise_model == 'ideal':

                    # Randomize expectations for the source

                    randomized_source_counts = np.random.poisson(source_model_counts + self._background_counts)
                    # randomized_source_rate = randomized_source_counts / self.exposure

                    # No randomization for the background in this case

                    randomized_background_counts = self._background_counts

                    randomized_background_count_err = None

                    randomized_source_count_err = None

                elif self._background_noise_model == 'gaussian':

                    # Since we use a profile likelihood, the background model is conditional on the source model, so let's
                    # get it from the likelihood function
                    _, background_model_counts = self._loglike_poisson_obs_gaussian_bkg()

                    # Randomize expectations for the source

                    randomized_source_counts = np.random.poisson(source_model_counts + background_model_counts)
                    # randomized_source_rate = randomized_source_counts / self.exposure

                    # Now randomize the expectations.

                    # We cannot generate variates with zero sigma. They variates from those channel will always be zero
                    # This is a limitation of this whole idea. However, remember that by construction an error of zero
                    # it is only allowed when the background counts are zero as well.
                    idx = (self._back_count_errors > 0)

                    randomized_background_counts = np.zeros_like(background_model_counts)

                    randomized_background_counts[idx] = np.random.normal(loc=background_model_counts[idx],
                                                                         scale=self._back_count_errors[idx])

                    # Issue a warning if the generated background is less than zero, and fix it by placing it at zero

                    idx = (randomized_background_counts < 0)  # type: np.ndarray

                    negative_background_n = np.sum(idx)

                    if negative_background_n > 0:
                        custom_warnings.warn("Generated background has negative counts "
                                             "in %i channels. Fixing them to zero" % (negative_background_n))

                        randomized_background_counts[idx] = 0

                    # Now compute rates and errors

                    # randomized_background_rate = randomized_background_counts / self.background_exposure

                    randomized_background_count_err = copy.copy(self._back_count_errors)

                    randomized_source_count_err = None

                else:

                    raise RuntimeError(
                        "This is a bug. The combination of source (%s) and background (%s) noise models does not exist" % (
                            self._observation_noise_model,
                            self._background_noise_model))

            else:

                # We need to generate Gaussian variates from the model to get the signal, and from the background
                # to get the new background

                # Now depending on the background noise model, generate randomized values for the background

                if self._background_noise_model == 'poisson':

                    raise RuntimeError(
                        "This is a bug. The combination of source (%s) and background (%s) noise models does not exist" % (
                            self._observation_noise_model,
                            self._background_noise_model))

                elif self._background_noise_model == 'ideal':

                    # Randomize expectations for the source

                    idx = (self._observed_count_errors > 0)

                    randomized_source_counts = np.zeros_like(source_model_counts)

                    randomized_source_counts[idx] = np.random.normal(
                        loc=source_model_counts[idx] + self._background_counts[idx],
                        scale=self._observed_count_errors)

                    # Issue a warning if the generated background is less than zero, and fix it by placing it at zero

                    idx = (randomized_source_counts < 0)  # type: np.ndarray

                    negative_source_n = np.sum(idx)

                    if negative_source_n > 0:
                        custom_warnings.warn("Generated source has negative counts "
                                             "in %i channels. Fixing them to zero" % (negative_source_n))

                        randomized_source_counts[idx] = 0

                    randomized_source_count_err = copy.copy(self._observed_count_errors)

                    # No randomization for the background in this case

                    randomized_background_counts = self._background_counts

                    randomized_background_count_err = None

                elif self._background_noise_model == 'gaussian':

                    # Since we use a profile likelihood, the background model is conditional on the source model, so let's
                    # get it from the likelihood function
                    _, background_model_counts = self._loglike_poisson_obs_gaussian_bkg()

                    # Randomize expectations for the source

                    idx = (self._observed_count_errors > 0)

                    randomized_source_counts = np.zeros_like(source_model_counts)

                    randomized_source_counts[idx] = np.random.normal(
                        loc=source_model_counts[idx] + background_model_counts[idx],
                        scale=self._observed_count_errors)

                    # Issue a warning if the generated background is less than zero, and fix it by placing it at zero

                    idx = (randomized_source_counts < 0)  # type: np.ndarray

                    negative_source_n = np.sum(idx)

                    if negative_source_n > 0:
                        custom_warnings.warn("Generated source has negative counts "
                                             "in %i channels. Fixing them to zero" % (negative_source_n))

                        randomized_source_counts[idx] = 0

                    randomized_source_count_err = copy.copy(self._observed_count_errors)

                    # Now randomize the expectations.

                    # We cannot generate variates with zero sigma. They variates from those channel will always be zero
                    # This is a limitation of this whole idea. However, remember that by construction an error of zero
                    # it is only allowed when the background counts are zero as well.
                    idx = (self._back_count_errors > 0)

                    randomized_background_counts = np.zeros_like(background_model_counts)

                    randomized_background_counts[idx] = np.random.normal(loc=background_model_counts[idx],
                                                                         scale=self._back_count_errors[idx])

                    # Issue a warning if the generated background is less than zero, and fix it by placing it at zero

                    idx = (randomized_background_counts < 0)  # type: np.ndarray

                    negative_background_n = np.sum(idx)

                    if negative_background_n > 0:
                        custom_warnings.warn("Generated background has negative counts "
                                             "in %i channels. Fixing them to zero" % (negative_background_n))

                        randomized_background_counts[idx] = 0

                    # Now compute rates and errors

                    # randomized_background_rate = randomized_background_counts / self.background_exposure

                    randomized_background_count_err = copy.copy(self._back_count_errors)

                elif self.background_noise_model is None:

                    idx = (self._observed_count_errors > 0)

                    randomized_source_counts = np.zeros_like(source_model_counts)

                    randomized_source_counts[idx] = np.random.normal(loc=source_model_counts[idx],
                                                                     scale=self._observed_count_errors)

                    # Issue a warning if the generated background is less than zero, and fix it by placing it at zero

                    idx = (randomized_source_counts < 0)  # type: np.ndarray

                    negative_source_n = np.sum(idx)

                    if negative_source_n > 0:
                        custom_warnings.warn("Generated source has negative counts "
                                             "in %i channels. Fixing them to zero" % (negative_source_n))

                        randomized_source_counts[idx] = 0

                    randomized_source_count_err = copy.copy(self._observed_count_errors)

                    randomized_background_counts = None

                else:

                    raise RuntimeError(
                        "This is a bug. The combination of source (%s) and background (%s) noise models does not exist" % (
                            self._observation_noise_model,
                            self._background_noise_model))

            # create new source and background spectra
            # the children of BinnedSpectra must properly override the new_spectrum
            # member so as to build the appropriate spectrum type. All parameters of the current
            # spectrum remain the same except for the rate and rate errors

            new_observation = self._observed_spectrum.clone(new_counts=randomized_source_counts,
                                                            new_count_errors=randomized_source_count_err)

            if self._background_spectrum is not None:

                new_background = self._background_spectrum.clone(new_counts=randomized_background_counts,
                                                                 new_count_errors=randomized_background_count_err)

            else:

                new_background = None

            # Now create another instance of BinnedSpectrum with the randomized data we just generated
            # notice that the _new member is a classmethod

            new_spectrum_plugin = self._new_plugin(name=new_name,
                                                   observation=new_observation,
                                                   background=new_background,
                                                   verbose=self._verbose,
                                                   **kwargs)

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

            new_spectrum_plugin._simulation_storage = clone_model(self._like_model)

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

        assert self._background_spectrum is not None, "This data has no background, cannot rebin on background!"

        rebinner = Rebinner(self._background_counts, min_number_of_counts, self._mask)

        # only for the PHASpectrum subclass do we need to update the the grouping
        if isinstance(self._observed_spectrum, PHASpectrum):
            self._observed_spectrum.set_ogip_grouping(rebinner.grouping)
            self._background_spectrum.set_ogip_grouping(rebinner.grouping)

        self._apply_rebinner(rebinner)

    def rebin_on_source(self, min_number_of_counts):
        """
        Rebin the spectrum guaranteeing the provided minimum number of counts in each source bin.

        To neutralize this completely, use "remove_rebinning"

        :param min_number_of_counts: the minimum number of counts in each bin
        :return: none
        """

        # NOTE: the rebinner takes care of the mask already



        rebinner = Rebinner(self._observed_counts, min_number_of_counts, self._mask)

        # only for the PHASpectrum subclass do we need to update the the grouping
        if isinstance(self._observed_spectrum, PHASpectrum):
            self._observed_spectrum.set_ogip_grouping(rebinner.grouping)
            self._background_spectrum.set_ogip_grouping(rebinner.grouping)

        self._apply_rebinner(rebinner)

    def _apply_rebinner(self, rebinner):

        self._rebinner = rebinner

        # Apply the rebinning to everything.
        # NOTE: the output of the .rebin method are the vectors with the mask *already applied*



        self._current_observed_counts = self._rebinner.rebin(self._observed_counts)

        if self._observed_count_errors is not None:
            self._current_observed_count_errors = self._rebinner.rebin_errors(self._observed_count_errors)

        if self._background_spectrum is not None:

            (self._current_background_counts,
             self._current_scaled_background_counts) = self._rebinner.rebin(self._background_counts,
                                                                            self._scaled_background_counts)

            if self._back_count_errors is not None:
                # NOTE: the output of the .rebin method are the vectors with the mask *already applied*

                self._current_back_count_errors, = self._rebinner.rebin_errors(self._back_count_errors)

        if self._verbose:
            print("Now using %s bins" % self._rebinner.n_bins)

    def remove_rebinning(self):
        """
        Remove the rebinning scheme set with rebin_on_background.

        :return:
        """

        # Restore original vectors with mask applied
        self._apply_mask_to_original_vectors()

        self._rebinner = None

    def _get_expected_background_counts_scaled(self):
        """
        Get the background counts expected in the source interval and in the source region, based on the observed
        background.

        :return:
        """

        # NOTE : this is called only once during construction!

        # The scale factor is the ratio between the collection area of the source spectrum and the
        # background spectrum. It is used for example for the typical aperture-photometry method used in
        # X-ray astronomy, where the background region has a different size with respect to the source region

        scale_factor = self._observed_spectrum.scale_factor / self._background_spectrum.scale_factor

        # The expected number of counts is the rate in the background file multiplied by its exposure, renormalized
        # by the scale factor.
        # (see http://heasarc.gsfc.nasa.gov/docs/asca/abc_backscal.html)

        bkg_counts = self._background_spectrum.rates * self._observed_spectrum.exposure * scale_factor

        return bkg_counts

    def _loglike_gaussian_obs_no_bkg(self):

        model_counts = self.get_model()

        chi2_ = chi2(self._current_observed_counts,
                     self._current_observed_count_errors,
                     model_counts)

        assert np.all(np.isfinite(chi2_))

        return np.sum(chi2_) * (-1)

    def _loglike_gaussian_obs_gaussian_bkg(self):

        raise NotImplementedError("We need to add chi2")
        model_counts = self.get_model()

        # loglike, bkg_model = chi2()

    def _loglike_poisson_obs_poisson_bkg(self):

        # Scale factor between source and background spectrum

        model_counts = self.get_model()

        loglike, bkg_model = poisson_observed_poisson_background(self._current_observed_counts,
                                                                 self._current_background_counts,
                                                                 self.scale_factor,
                                                                 model_counts)

        return np.sum(loglike), bkg_model

    def _loglike_poisson_obs_gaussian_bkg(self):

        expected_model_counts = self.get_model()

        loglike, bkg_model = poisson_observed_gaussian_background(self._current_observed_counts,
                                                                  self._current_background_counts,
                                                                  self._current_back_count_errors,
                                                                  expected_model_counts)

        return np.sum(loglike), bkg_model

    def _loglike_poisson_obs_ideal_bkg(self):

        # In this likelihood the background becomes part of the model, which means that
        # the uncertainty in the background is completely neglected

        model_counts = self.get_model()

        loglike, _ = poisson_log_likelihood_ideal_bkg(self._current_observed_counts,
                                                      self._current_scaled_background_counts,
                                                      model_counts)

        return np.sum(loglike), None

    def _set_background_noise_model(self, new_model):

        # Do not make differences between upper and lower cases
        if new_model is not None:
            new_model = new_model.lower()

            assert new_model in _known_noise_models, "Noise model %s not recognized. " \
                                                     "Allowed models are: %s" % (
                                                     new_model, ", ".join(_known_noise_models))

        self._background_noise_model = new_model

    def _get_background_noise_model(self):

        return self._background_noise_model

    background_noise_model = property(_get_background_noise_model, _set_background_noise_model,
                                      doc="Sets/gets the noise model for the background spectrum")

    def _set_observation_noise_model(self, new_model):

        # Do not make differences between upper and lower cases
        new_model = new_model.lower()

        assert new_model in _known_noise_models, "Noise model %s not recognized. " \
                                                 "Allowed models are: %s" % (new_model, ", ".join(_known_noise_models))

        self._observation_noise_model = new_model

    def _get_observation_noise_model(self):

        return self._observation_noise_model

    observation_noise_model = property(_get_observation_noise_model, _set_observation_noise_model,
                                       doc="Sets/gets the noise model for the background spectrum")

    @set_external_property
    def get_log_like(self):

        if self._observation_noise_model == 'poisson':

            if self._background_noise_model == 'poisson':

                loglike, _ = self._loglike_poisson_obs_poisson_bkg()

            elif self._background_noise_model == 'ideal':

                loglike, _ = self._loglike_poisson_obs_ideal_bkg()

            elif self._background_noise_model == 'gaussian':

                loglike, _ = self._loglike_poisson_obs_gaussian_bkg()

            else:

                raise RuntimeError("This is a bug")

        else:

            if self._background_noise_model is None:
                loglike = self._loglike_gaussian_obs_no_bkg()

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

        assert self._like_model.get_number_of_extended_sources() == 0, "OGIP-like plugins do not support " \
                                                                       "extended sources"

        # Get the differential flux function, and the integral function, with no dispersion,
        # we simply integrate the model over the bins

        differential_flux, integral = self._get_diff_flux_and_integral()

        self._integral_flux = integral

    def _evaluate_model(self):
        """
        Since there is no dispersion, we simply evaluate the model by integrating over the energy bins.
        This can be overloaded to convolve the model with a response, for example


        :return:
        """

        return np.array([self._integral_flux(emin, emax) for emin, emax in self._observed_spectrum.bin_stack])

    def get_model(self):
        """
        The model integrated over the energy bins. Note that it only returns the  model for the
        currently active channels/measurements

        :return: array of folded model
        """

        if self._rebinner is not None:

            model, = self._rebinner.rebin(self._evaluate_model() * self._observed_spectrum.exposure)

        else:

            model = self._evaluate_model()[self._mask] * self._observed_spectrum.exposure

        return self._nuisance_parameter.value * model

    def _get_diff_flux_and_integral(self):

        n_point_sources = self._like_model.get_number_of_point_sources()

        # Make a function which will stack all point sources (OGIP do not support spatial dimension)

        def differential_flux(energies):
            fluxes = self._like_model.get_point_source_fluxes(0, energies)

            # If we have only one point source, this will never be executed
            for i in range(1, n_point_sources):
                fluxes += self._like_model.get_point_source_fluxes(i, energies)

            return fluxes

        # The following integrates the diffFlux function using Simpson's rule
        # This assume that the intervals e1,e2 are all small, which is guaranteed
        # for any reasonable response matrix, given that e1 and e2 are Monte-Carlo
        # energies. It also assumes that the function is smooth in the interval
        # e1 - e2 and twice-differentiable, again reasonable on small intervals for
        # decent models. It might fail for models with too sharp features, smaller
        # than the size of the monte carlo interval.

        def integral(e1, e2):
            # Simpson's rule

            return (e2 - e1) / 6.0 * (differential_flux(e1)
                                      + 4 * differential_flux((e1 + e2) / 2.0)
                                      + differential_flux(e2))

        return differential_flux, integral

    def use_effective_area_correction(self, min_value=0.8, max_value=1.2):
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

        self._nuisance_parameter.free = True
        self._nuisance_parameter.bounds = (min_value, max_value)

        # Use a uniform prior by default

        self._nuisance_parameter.set_uninformative_prior(Uniform_prior)

    def fix_effective_area_correction(self, value=1):
        """
        Fix the multiplicative factor (see use_effective_area_correction) to the provided value (default: 1)

        :param value: new value (default: 1, i.e., no correction)
        :return:
        """

        self._nuisance_parameter.value = value
        self._nuisance_parameter.fix = True

    @property
    def scale_factor(self):
        """
        Ratio between the source and the background exposure and area

        :return:
        """

        return self._observed_spectrum.exposure / self._background_spectrum.exposure * self._observed_spectrum.scale_factor / self._background_spectrum.scale_factor

    @property
    def tstart(self):
        return self._tstart

    @property
    def tstop(self):
        return self._tstop

    @property
    def background_exposure(self):
        """
        Exposure of the background spectrum
        """
        return self._background_spectrum.exposure

    @property
    def exposure(self):
        """
        Exposure of the source spectrum
        """

        return self._observed_spectrum.exposure

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

            energy_min, energy_max = self._rebinner.get_new_start_and_stop(energy_min, energy_max)

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

        sig_obj = Significance(Non=self._observed_spectrum.total_count,
                               Noff=self._background_spectrum.total_count,
                               alpha=self.exposure / self.background_exposure)

        if self._observed_spectrum.is_poisson and self._background_spectrum.is_poisson:

            # use simple li & ma
            significance = sig_obj.li_and_ma()

        elif self._observed_spectrum.is_poisson and not self._background_spectrum.is_poisson:

            significance = sig_obj.li_and_ma_equivalent_for_gaussian_background(
                self._background_spectrum.total_count_error)

        else:

            raise NotImplementedError("We haven't put in other significances yet")

        return significance[0]

    @property
    def significance_per_channel(self):
        """
        :return: the significance of the data over background per channel
        """

        sig_obj = Significance(Non=self._current_observed_counts,
                               Noff=self._current_background_counts,
                               alpha=self.exposure / self.background_exposure)

        if self._observed_spectrum.is_poisson and self._background_spectrum.is_poisson:

            # use simple li & ma
            significance = sig_obj.li_and_ma()

        elif self._observed_spectrum.is_poisson and not self._background_spectrum.is_poisson:

            significance = sig_obj.li_and_ma_equivalent_for_gaussian_background(self._current_back_count_errors)

        else:

            raise NotImplementedError("We haven't put in other significances yet")

        return significance

    def write_pha(self):

        raise NotImplementedError("this is in progress")


        # we just need to make a diagonal response and then follow the example in dispersion like

    def view_count_spectrum(self, plot_errors=True, show_bad_channels=True, show_warn_channels=False,
                            significance_level=None):
        """
        View the count and background spectrum. Useful to check energy selections.
        :param plot_errors: plot errors on the counts
        :param show_bad_channels: (bool) show which channels are bad in the native PHA quality
        :return:
        """

        if sum(self._mask) == 0:
            raise RuntimeError("There are no active channels selected to plot!")

        # adding the rebinner: j. michael.

        # In the future read colors from config file

        # First plot the counts

        # find out the type of observation

        if self._observation_noise_model == 'poisson':

            # Observed counts
            observed_counts = copy.copy(self._current_observed_counts)

            cnt_err = np.sqrt(observed_counts)

            if self._background_noise_model == 'poisson':

                background_counts = copy.copy(self._current_background_counts)

                background_errors = np.sqrt(background_counts)

            elif self._background_noise_model == 'ideal':

                background_counts = copy.copy(self._current_scaled_background_counts)

                background_errors = np.zeros_like(background_counts)

            elif self._background_noise_model == 'gaussian':

                background_counts = copy.copy(self._current_background_counts)

                background_errors = copy.copy(self._current_back_count_errors)

            else:

                raise RuntimeError("This is a bug")

            # convert to rates, ugly, yes

            observed_counts /= self._observed_spectrum.exposure
            background_counts /= self._background_spectrum.exposure
            cnt_err /= self._observed_spectrum.exposure
            background_errors /= self._background_spectrum.exposure


        else:

            if self._background_noise_model is None:
                observed_counts = copy.copy(self._current_observed_counts)

                cnt_err = copy.copy(self._current_observed_count_errors)

                # convert to rates, ugly, yes

            observed_counts /= self._observed_spectrum.exposure
            background_counts = np.zeros(observed_counts.shape)
            cnt_err /= self._observed_spectrum.exposure
            background_errors = np.zeros(observed_counts.shape)

        # Make the plots
        fig, ax = plt.subplots()

        # Get the energy boundaries

        energy_min, energy_max = self.energy_boundaries

        energy_width = energy_max - energy_min

        # plot counts and background for the currently selected data

        channel_plot(ax, energy_min, energy_max, observed_counts,
                     color=threeML_config['ogip']['counts color'], lw=1.5, alpha=1, label="Total")

        channel_plot(ax, energy_min, energy_max, background_counts,
                     color=threeML_config['ogip']['background color'], alpha=.8, label="Background")

        mean_chan = np.mean([energy_min, energy_max], axis=0)

        # if asked, plot the errors

        if plot_errors:
            ax.errorbar(mean_chan,
                        observed_counts / energy_width,
                        yerr=cnt_err / energy_width,
                        fmt='',
                        # markersize=3,
                        linestyle='',
                        elinewidth=.7,
                        alpha=.9,
                        capsize=0,
                        # label=data._name,
                        color=threeML_config['ogip']['counts color'])

            if self._background_noise_model is not None:
                ax.errorbar(mean_chan,
                            background_counts / energy_width,
                            yerr=background_errors / energy_width,
                            fmt='',
                            # markersize=3,
                            linestyle='',
                            elinewidth=.7,
                            alpha=.9,
                            capsize=0,
                            # label=data._name,
                            color=threeML_config['ogip']['background color'])

        # Now plot and fade the non-used channels
        non_used_mask = (~self._mask)

        if non_used_mask.sum() > 0:

            # Get un-rebinned versions of all arrays, so we can directly apply the mask
            energy_min_unrebinned, energy_max_unrebinned = np.array(self._observed_spectrum.starts), np.array(
                self._observed_spectrum.stops)
            energy_width_unrebinned = energy_max_unrebinned - energy_min_unrebinned
            observed_rate_unrebinned = self._observed_counts / self.exposure
            observed_rate_unrebinned_err = np.sqrt(self._observed_counts) / self.exposure

            channel_plot(ax,
                         energy_min_unrebinned[non_used_mask],
                         energy_max_unrebinned[non_used_mask],
                         observed_rate_unrebinned[non_used_mask],
                         color=threeML_config['ogip']['counts color'], lw=1.5, alpha=1)

            if self._background_noise_model is not None:

                background_rate_unrebinned = self._background_counts / self.background_exposure
                background_rate_unrebinned_err = np.sqrt(self._background_counts) / self.background_exposure

                channel_plot(ax, energy_min_unrebinned[non_used_mask],
                             energy_max_unrebinned[non_used_mask],
                             background_rate_unrebinned[non_used_mask],
                             color=threeML_config['ogip']['background color'], alpha=.8)
            else:

                background_rate_unrebinned = np.zeros_like(observed_rate_unrebinned)
                background_rate_unrebinned_err = np.zeros_like(observed_rate_unrebinned_err)

            if plot_errors:
                mean_chan_unrebinned = np.mean([energy_min_unrebinned, energy_max_unrebinned], axis=0)

                ax.errorbar(mean_chan_unrebinned[non_used_mask],
                            observed_rate_unrebinned[non_used_mask] / energy_width_unrebinned[non_used_mask],
                            yerr=observed_rate_unrebinned_err[non_used_mask] / energy_width_unrebinned[non_used_mask],
                            fmt='',
                            # markersize=3,
                            linestyle='',
                            elinewidth=.7,
                            alpha=.9,
                            capsize=0,
                            # label=data._name,
                            color=threeML_config['ogip']['counts color'])

                if self._background_noise_model is not None:
                    ax.errorbar(mean_chan_unrebinned[non_used_mask],
                                background_rate_unrebinned[non_used_mask] / energy_width_unrebinned[non_used_mask],
                                yerr=background_rate_unrebinned_err[non_used_mask] / energy_width_unrebinned[
                                    non_used_mask],
                                fmt='',
                                # markersize=3,
                                linestyle='',
                                elinewidth=.7,
                                alpha=.9,
                                capsize=0,
                                # label=data._name,
                                color=threeML_config['ogip']['background color'])

            # make some nice top and bottom plot ranges

            top = max([max(background_rate_unrebinned / energy_width_unrebinned),
                       max(observed_rate_unrebinned / energy_width_unrebinned)]) * 1.5

            bottom = min([min(background_rate_unrebinned / energy_width_unrebinned),
                          min(observed_rate_unrebinned / energy_width_unrebinned)]) * 0.8

            # plot the deselected regions

            disjoint_patch_plot(ax,
                                energy_min_unrebinned,
                                energy_max_unrebinned,
                                top,
                                bottom,
                                ~self._mask,
                                color='k',
                                alpha=.4)

            # plot the bad quality channels if requested

            if show_bad_channels:

                if self._verbose and sum(self._observed_spectrum.quality.bad) > 0:
                    print('bad channels shown in red hatching\n')

                disjoint_patch_plot(ax,
                                    energy_min_unrebinned,
                                    energy_max_unrebinned,
                                    top,
                                    bottom,
                                    self._observed_spectrum.quality.bad,
                                    color='none',
                                    edgecolor='#FE3131',
                                    hatch='/',
                                    alpha=.95
                                    )

            if show_warn_channels:

                if self._verbose and sum(self._observed_spectrum.quality.warn) > 0:
                    print('warned channels shown in purple hatching\n')

                disjoint_patch_plot(ax,
                                    energy_min_unrebinned,
                                    energy_max_unrebinned,
                                    top,
                                    bottom,
                                    self._observed_spectrum.quality.bad,
                                    color='none',
                                    edgecolor='#C79BFE',
                                    hatch='/',
                                    alpha=.95
                                    )

            if significance_level is not None:

                if self._verbose:
                    print('channels below the significance threshold shown in red\n')

                significance_mask = self.significance_per_channel < significance_level

                disjoint_patch_plot(ax,
                                    energy_min_unrebinned,
                                    energy_max_unrebinned,
                                    top,
                                    bottom,
                                    significance_mask,
                                    color='red',
                                    alpha=.3
                                    )

        ax.set_xlabel("Energy (keV)")
        ax.set_ylabel("Rate (counts s$^{-1}$ keV$^{-1}$)")
        ax.set_xlim(left=self._observed_spectrum.absolute_start, right=self._observed_spectrum.absolute_stop)
        ax.legend()

        def __repr__(self):

            return self._output().to_string()

    def _output(self):
        # type: () -> pd.Series

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
        # obs['response'] = self._observed_spectrum.response_file

        return pd.Series(data=obs, index=obs.keys())

    def display(self):

        display(self._output().to_frame())

    def __repr__(self):

        return self._output().to_string()
