import collections
import copy
from contextlib import contextmanager
import matplotlib.pyplot as plt
import numpy as np
from astromodels.core.parameter import Parameter
from astromodels.functions.functions import Uniform_prior
from astromodels.utils.valid_variable import is_valid_variable_name
from astromodels import clone_model

from threeML.io.file_utils import file_existing_and_readable, sanitize_filename
from threeML.io.step_plot import step_plot
from threeML.exceptions.custom_exceptions import custom_warnings
from threeML.plugin_prototype import PluginPrototype, set_external_property
from threeML.plugins.OGIP.likelihood_functions import poisson_log_likelihood_ideal_bkg
from threeML.plugins.OGIP.likelihood_functions import poisson_observed_gaussian_background
from threeML.plugins.OGIP.likelihood_functions import poisson_observed_poisson_background
from threeML.plugins.OGIP.pha import PHA
from threeML.plugins.OGIP.response import Response
from threeML.utils.binner import Rebinner
from threeML.plugins.OGIP.pha import PHAContainer, PHAWrite
from threeML.config.config import threeML_config


__instrument_name = "All OGIP-compliant instruments"

# This defines the known noise models for source and/or background spectra
_known_noise_models = ['poisson', 'gaussian', 'ideal']


class OGIPLike(PluginPrototype):
    def __init__(self, name, pha_file, bak_file=None, rsp_file=None, arf_file=None, spectrum_number=None, verbose=True):

        # Just a toggle for verbosity
        self._verbose = bool(verbose)

        assert is_valid_variable_name(name), "Name %s is not a valid name for a plugin. You must use a name which is " \
                                             "a valid python identifier: no spaces, no operators (+,-,/,*), " \
                                             "it cannot start with a number, no special characters" % name

        # Read the pha file (or the PHAContainer instance)

        self._pha = self._get_pha_instance(pha_file, spectrum_number=spectrum_number)  # type: PHA

        # Get the required background file, response and (if present) arf_file either from the
        # calling sequence or the file.
        # NOTE: if one of the file is specified in the calling sequence, it will be used whether or not there is an
        # equivalent specification in the header. This allows the user to override the content of the header of the
        # PHA file, if needed

        if bak_file is None:

            bak_file = self._pha.background_file

            assert bak_file is not None, "No background file provided, and the PHA file does not specify one."

        # Get a PHA instance with the background

        self._bak = self._get_pha_instance(bak_file, file_type='background', spectrum_number=spectrum_number)

        # Now handle the response

        if rsp_file is None:

            rsp_file = self._pha.response_file

            assert rsp_file is not None, "No response file provided, and the PHA file does not specify one."

        if arf_file is None:

            # Note that this could be None as well, if there is no ancillary file specified in the header

            arf_file = self._pha.ancillary_file

        # Read in the response

        if isinstance(rsp_file, str):

            self._rsp = Response(rsp_file, arf_file=arf_file)

        else:

            # Assume a fully formed Response class
            self._rsp = rsp_file

        # Make sure that data and background have the same number of channels

        assert self._pha.n_channels == self._bak.n_channels, "Data file and background file have different " \
                                                             "number of channels"

        # Precomputed observed and background counts (for speed)

        self._observed_counts = self._pha.rates * self._pha.exposure  # type: np.ndarray
        self._background_counts = self._bak.rates * self._bak.exposure  # type: np.ndarray
        self._scaled_background_counts = self._get_expected_background_counts_scaled()  # type: np.ndarray

        # Init everything else to None
        self._like_model = None
        self._rebinner = None

        # Now auto-probe the statistic to use
        if self._pha.is_poisson():

            if self._bak.is_poisson():

                self.observation_noise_model = 'poisson'
                self.background_noise_model = 'poisson'

                self._back_counts_errors = None

                assert np.all(self._observed_counts >= 0), "Error in PHA: negative counts!"

                assert np.all(self._background_counts >= 0), "Error in background spectrum: negative counts!"

            else:

                self.observation_noise_model = 'poisson'
                self.background_noise_model = 'gaussian'

                self._back_counts_errors = self._bak.rate_errors * self._bak.exposure  # type: np.ndarray

                idx = (self._back_counts_errors == 0)  # type: np.ndarray

                assert np.all(self._back_counts_errors[idx] == self._background_counts[idx]), \
                    "Error in background spectrum: if the error on the background is zero, " \
                    "also the expected background must be zero"

                assert np.all(self._background_counts >= 0), "Error in background spectrum: negative background!"

        else:

            raise NotImplementedError("Gaussian observation is not yet supported")

        # Initialize a mask that selects all the data.
        # We will initially use the quality mask for the PHA file
        # and set any quality greater than 0 to False. We want to save
        # the native quality so that we can warn the user if they decide to
        # select channels that were flagged as bad.


        self._native_quality = self._pha.quality

        assert len(self._native_quality) == len(
                self._observed_counts), "The PHA quality column and rates column are not the same size."

        self._mask = np.asarray(np.ones(self._pha.n_channels), np.bool)

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

        super(OGIPLike, self).__init__(name, nuisance_parameters)

        # The following vectors are the ones that will be really used for the computation. At the beginning they just
        # point to the original ones, but if a rebinner is used and/or a mask is created through set_active_measurements,
        # they will contain the rebinned and/or masked versions

        self._current_observed_counts = self._observed_counts
        self._current_background_counts = self._background_counts
        self._current_scaled_background_counts = self._scaled_background_counts
        self._current_back_counts_errors = self._back_counts_errors

        # This will be used to keep track of how many syntethic datasets have been generated
        self._n_synthetic_datasets = 0

        self._tstart = None
        self._tstop = None

        # This is so far not a simulated data set
        self._simulation_storage = None

        # set the mask to the native quality
        self._mask = self._quality_to_mask()
        # Apply the mask
        self._apply_mask_to_original_vectors()

    def _get_pha_instance(self, pha_file_or_container, *args, **kwargs):

        # If the user didn't provide them, read the needed files from the keywords in the PHA file
        # It is possible a user passed a PHAContainer from an EventList. In this case, this will fail
        # resulting in an attribute error. We will check for this and if it fails, and then try to load the
        # PHAContainer

        if isinstance(pha_file_or_container, str):

            # This is supposed to be a filename

            pha_file = sanitize_filename(pha_file_or_container)

            assert file_existing_and_readable(pha_file.split("{")[0]), "File %s not existing or not " \
                                                                       "readable" % pha_file

            pha = PHA(pha_file, *args, **kwargs)

        else:

            # Assume this is a PHAContainer or a subclass (or some other object with the same interface)

            pha = PHA(pha_file_or_container, *args, **kwargs)

        return pha

    def get_pha_files(self):

        info = {}

        # we want to pass copies so that
        # the user doesn't grab the instance
        # and try to modify things. protection
        info['pha'] = copy.copy(self._pha)
        info['bak'] = copy.copy(self._bak)
        info['rsp'] = copy.copy(self._rsp)

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

        * Using native PHA qaulity:

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

            self._mask = self._quality_to_mask()

        else:

            # otherwise, we will start out with all channels deselected
            # and turn the on/off by the arguments

            self._mask = np.zeros(self._pha.n_channels, dtype=bool)

        if 'all' in args:

            # Just make sure than no further selections have been made.

            assert len(args) == 1, "If you specify 'all', you cannot specify more than one energy range."

            # Convert the mask to all True (we use all channels)
            self._mask[:] = True



        elif 'reset' in args:

            # Convert the to native PHA masking specified in quality

            assert len(args) == 1, "If you specify 'reset', you cannot specify more than one energy range."

            self._mask = self._quality_to_mask()






        else:

            for arg in args:

                selections = arg.replace(" ", "").split("-")

                # We need to find out if it is a channel or and energy being requested

                idx = np.empty(2, dtype=int)
                for i, s in enumerate(selections):

                    if s[0].lower() == 'c':

                        assert int(s[1:]) <= self._pha.n_channels, "%s is larger than the number of channels: %d" % (
                            s, self._pha.n_channels)
                        idx[i] = int(s[1:])

                    else:

                        idx[i] = self._rsp.energy_to_channel(float(s))

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

                        assert int(s[1:]) <= self._pha.n_channels, "%s is larger than the number of channels: %d" % (
                            s, self._pha.n_channels)
                        idx[i] = int(s[1:])

                    else:

                        idx[i] = self._rsp.energy_to_channel(float(s))

                assert idx[0] < idx[
                    1], "The channel and energy selection (%s) are out of order and translate to %s-%s" % (
                    selections, idx[0], idx[1])

                # we do the opposite of the exclude command!
                self._mask[idx[0]:idx[1] + 1] = False

                if self._verbose:
                    print("Range %s translates to excluding channels %s-%s" % (arg, idx[0], idx[1]))

        if self._verbose:
            print("Now using %s channels out of %s" % (np.sum(self._mask), self._pha.n_channels))

        # Apply the mask
        self._apply_mask_to_original_vectors()

        # if the user did not specify use_quality, they may have selected channels which
        # are marked BAD (5) in the native PHA file. We want to warn them in this case only (or maybe in all cases?)

        if not use_quality:

            number_of_native_good_channels = sum(self._quality_bad_to_mask())
            number_of_user_good_channels = sum(self._mask)

            if number_of_user_good_channels > number_of_native_good_channels:

                # we have more good channels than specified in the PHA file
                # so we need to figure out which channels these are where excluded

                deselected_channels = []
                for i in xrange(self._pha.n_channels):

                    if not self._quality_bad_to_mask()[i] and self._mask[i]:

                        deselected_channels.append(i)

                custom_warnings.warn("You have opted to use channels which are flagged BAD in the PHA file.")

                if self._verbose:

                    custom_warnings.warn("These channels are:")

                    for i in deselected_channels:

                        custom_warnings.warn("channel:%d" % i)

    def _quality_to_mask(self):
        """
        Convert the quality array to a channel mask.
        Any channel with quality greater than 0


        :return: boolean array channel maske
        """

        return self._native_quality == 0

    def _quality_bad_to_mask(self):
        """
        Convert the quality array to a channel mask
        for channels marked 5

        :return: boolean array channel maske
        """

        return self._native_quality <= 2

    def _apply_mask_to_original_vectors(self):

        # Apply the mask

        self._current_observed_counts = self._observed_counts[self._mask]
        self._current_background_counts = self._background_counts[self._mask]
        self._current_scaled_background_counts = self._scaled_background_counts[self._mask]

        if self._back_counts_errors is not None:

            self._current_back_counts_errors = self._back_counts_errors[self._mask]

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

    def get_simulated_dataset(self, new_name=None):
        """
        Returns another OGIPLike instance where data have been obtained by randomizing the current expectation from the
        model, as well as from the background (depending on the respective noise models)

        :return: an OGIPLike instance
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

            source_model_counts = self._rsp.convolve() * self.exposure

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
                    randomized_source_rate = randomized_source_counts / self.exposure

                    # Randomize expectations for the background

                    randomized_background_counts = np.random.poisson(background_model_counts)
                    randomized_background_rate = randomized_background_counts / self.background_exposure

                    randomized_background_rate_err = None

                elif self._background_noise_model == 'ideal':

                    # Randomize expectations for the source

                    randomized_source_counts = np.random.poisson(source_model_counts + self._background_counts)
                    randomized_source_rate = randomized_source_counts / self.exposure

                    # No randomization for the background in this case

                    randomized_background_rate = self._background_counts / self.background_exposure

                    randomized_background_rate_err = None

                elif self._background_noise_model == 'gaussian':

                    # Since we use a profile likelihood, the background model is conditional on the source model, so let's
                    # get it from the likelihood function
                    _, background_model_counts = self._loglike_poisson_obs_gaussian_bkg()

                    # Randomize expectations for the source

                    randomized_source_counts = np.random.poisson(source_model_counts + background_model_counts)
                    randomized_source_rate = randomized_source_counts / self.exposure

                    # Now randomize the expectations.

                    # We cannot generate variates with zero sigma. They variates from those channel will always be zero
                    # This is a limitation of this whole idea. However, remember that by construction an error of zero
                    # it is only allowed when the background counts are zero as well.
                    idx = (self._back_counts_errors > 0)

                    randomized_background_counts = np.zeros_like(background_model_counts)

                    randomized_background_counts[idx] = np.random.normal(loc=background_model_counts[idx],
                                                                         scale=self._back_counts_errors[idx])

                    # Issue a warning if the generated background is less than zero, and fix it by placing it at zero

                    idx = (randomized_background_counts < 0)  # type: np.ndarray

                    negative_background_n = np.sum(idx)

                    if negative_background_n > 0:

                        custom_warnings.warn("Generated background has negative counts "
                                             "in %i channels. Fixing them to zero" % (negative_background_n))

                        randomized_background_counts[idx] = 0

                    # Now compute rates and errors

                    randomized_background_rate = randomized_background_counts / self.background_exposure

                    randomized_background_rate_err = copy.copy(self._back_counts_errors) / self.background_exposure

                else:

                    raise RuntimeError("This is a bug")

            else:

                raise NotImplementedError("Not yet implemented")

            n_channels = original_mask.shape[0]

            if self.observation_noise_model == 'poisson':

                is_obs_poisson = True

            else:

                is_obs_poisson = False

            pha = PHAContainer(rates=randomized_source_rate,
                               n_channels=n_channels,
                               exposure=self.exposure,
                               is_poisson=is_obs_poisson,
                               response_file=None,  # We will specify it later
                               ancillary_file=None,  # We will specify it later
                               quality=self._native_quality,
                               mission=self._pha.mission,
                               instrument=self._pha.instrument
                               )

            if self.background_noise_model == 'poisson':

                is_bkg_poisson = True

            else:

                is_bkg_poisson = False

            bak = PHAContainer(rates=randomized_background_rate,
                               rate_errors=randomized_background_rate_err,
                               n_channels=n_channels,
                               exposure=self.background_exposure,
                               is_poisson=is_bkg_poisson,
                               response_file=None,
                               ancillary_file=None,
                               quality=self._native_quality,
                               mission=self._pha.mission,
                               instrument=self._pha.instrument
                               )

            # Now create another instance of OGIPLike with the randomized data we just generated

            new_ogip_like = OGIPLike(new_name,
                                     pha_file=pha,
                                     bak_file=bak,
                                     rsp_file=self._rsp,  # Use the currently loaded response so we don't need to
                                     # re-read from disk (way faster!)
                                     arf_file=None,  # The ARF is None because if present has been already read in
                                     # the self._rsp class
                                     verbose=self._verbose)

            # Apply the same selections as the current data set
            if original_rebinner is not None:

                # Apply rebinning, which also applies the mask
                new_ogip_like._apply_rebinner(original_rebinner)

            else:

                # Only apply the mask
                new_ogip_like._mask = original_mask
                new_ogip_like._apply_mask_to_original_vectors()

            # We want to store the simulated parameters so that the user
            # can recall them later

            new_ogip_like._simulation_storage = clone_model(self._like_model)

            # TODO: nuisance parameters

            return new_ogip_like

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

        rebinner = Rebinner(self._background_counts, min_number_of_counts, self._mask)

        self._apply_rebinner(rebinner)

    def _apply_rebinner(self, rebinner):

        self._rebinner = rebinner

        # Apply the rebinning to everything.
        # NOTE: the output of the .rebin method are the vectors with the mask *already applied*

        (self._current_observed_counts,
         self._current_background_counts,
         self._current_scaled_background_counts) = self._rebinner.rebin(self._observed_counts,
                                                                        self._background_counts,
                                                                        self._scaled_background_counts)

        if self._back_counts_errors is not None:
            # NOTE: the output of the .rebin method are the vectors with the mask *already applied*

            self._current_back_counts_errors, = self._rebinner.rebin_errors(self._back_counts_errors)

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

        scale_factor = self._pha.scale_factor / self._bak.scale_factor

        # The expected number of counts is the rate in the background file multiplied by its exposure, renormalized
        # by the scale factor.
        # (see http://heasarc.gsfc.nasa.gov/docs/asca/abc_backscal.html)

        bkg_counts = self._bak.rates * self._pha.exposure * scale_factor

        return bkg_counts

    def get_folded_model(self):
        """
        The model folded by the current defined response. Note that it only returns the folded model for the
        currently active channels/measurements

        :return: array of folded model
        """

        if self._rebinner is not None:

            folded_model, = self._rebinner.rebin(self._rsp.convolve() * self._pha.exposure)

        else:

            folded_model = self._rsp.convolve()[self._mask] * self._pha.exposure

        return self._nuisance_parameter.value * folded_model

    def _loglike_poisson_obs_ideal_bkg(self):

        # In this likelihood the background becomes part of the model, which means that
        # the uncertainty in the background is completely neglected

        model_counts = self.get_folded_model()

        loglike, _ = poisson_log_likelihood_ideal_bkg(self._current_observed_counts,
                                                      self._current_scaled_background_counts,
                                                      model_counts)

        return np.sum(loglike), None

    @property
    def scale_factor(self):
        """
        Ratio between the source and the background exposure and area

        :return:
        """

        return self._pha.exposure / self._bak.exposure * self._pha.scale_factor / self._bak.scale_factor

    @property
    def tstart(self):
        return self._tstart

    @property
    def tstop(self):
        return self._tstop

    def _loglike_poisson_obs_poisson_bkg(self):

        # Scale factor between source and background spectrum

        model_counts = self.get_folded_model()

        loglike, bkg_model = poisson_observed_poisson_background(self._current_observed_counts,
                                                                 self._current_background_counts,
                                                                 self.scale_factor,
                                                                 model_counts)

        return np.sum(loglike), bkg_model

    def _loglike_poisson_obs_gaussian_bkg(self):

        expected_model_counts = self.get_folded_model()

        loglike, bkg_model = poisson_observed_gaussian_background(self._current_observed_counts,
                                                                  self._current_background_counts,
                                                                  self._current_back_counts_errors,
                                                                  expected_model_counts)

        return np.sum(loglike), bkg_model

    def _set_background_noise_model(self, new_model):

        # Do not make differences between upper and lower cases
        new_model = new_model.lower()

        assert new_model in _known_noise_models, "Noise model %s not recognized. " \
                                                 "Allowed models are: %s" % (new_model, ", ".join(_known_noise_models))

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

            raise NotImplementedError("Not yet implemented")

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

        # Get the differential flux function, and the integral function, which will be then convoluted with the
        # response

        differential_flux, integral = self._get_diff_flux_and_integral()

        self._rsp.set_function(differential_flux, integral)

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
    def background_exposure(self):
        """
        Exposure of the background spectrum
        """
        return self._bak.exposure

    @property
    def exposure(self):
        """
        Exposure of the source spectrum
        """

        return self._pha.exposure

    @property
    def energy_boundaries(self, mask=True):
        """
        Energy boundaries of channels currently in use (rebinned, if a rebinner is active)

        :return: (energy_min, energy_max)
        """

        energies = self._rsp.ebounds.T

        energy_min, energy_max = energies

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
    def ogip_grouping(self):

        if self._rebinner is None:

            return np.ones_like(self._observed_counts)

        else:

            return self._rebinner.grouping

    @property
    def ogip_quality(self):
        """
        The quality of the current mask, not of the native quality
        of the PHA file


        :return:
        """

        quality = np.zeros_like(self._observed_counts)
        quality[~self._mask] = 2

        return quality

    def view_count_spectrum(self, plot_errors=True, show_bad_channels=True):
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

                background_errors = copy.copy(self._current_back_counts_errors)

            else:

                raise RuntimeError("This is a bug")

        else:

            raise NotImplementedError("Not yet implemented")

        # convert to rates, ugly, yes

        observed_counts /= self._pha.exposure
        background_counts /= self._bak.exposure
        cnt_err /= self._pha.exposure
        background_errors /= self._bak.exposure

        # Make the plots
        fig, ax = plt.subplots()

        # Get the energy boundaries

        energy_min, energy_max = self.energy_boundaries

        energy_width = energy_max - energy_min

        # plot counts and background for the currently selected data

        channel_plot(ax, energy_min, energy_max, observed_counts,
                     color='#377eb8', lw=1.5, alpha=1, label="Total")

        channel_plot(ax, energy_min, energy_max, background_counts,
                     color='#e41a1c', alpha=.8, label="Background")

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
                        color='#377eb8')

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
                        color='#e41a1c')

        # Now plot and fade the non-used channels
        non_used_mask = (~self._mask)

        if non_used_mask.sum() > 0:

            # Get un-rebinned versions of all arrays, so we can directly apply the mask
            energy_min_unrebinned, energy_max_unrebinned = self._rsp.ebounds.T
            energy_width_unrebinned = energy_max_unrebinned - energy_min_unrebinned
            observed_rate_unrebinned = self._observed_counts / self.exposure
            observed_rate_unrebinned_err = np.sqrt(self._observed_counts) / self.exposure
            background_rate_unrebinned = self._background_counts / self.background_exposure
            background_rate_unrebinned_err = np.sqrt(self._background_counts) / self.background_exposure

            channel_plot(ax,
                         energy_min_unrebinned[non_used_mask],
                         energy_max_unrebinned[non_used_mask],
                         observed_rate_unrebinned[non_used_mask],
                         color=threeML_config['ogip']['counts color'], lw=1.5, alpha=1)

            channel_plot(ax, energy_min_unrebinned[non_used_mask],
                         energy_max_unrebinned[non_used_mask],
                         background_rate_unrebinned[non_used_mask],
                         color=threeML_config['ogip']['background color'], alpha=.8)

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

                ax.errorbar(mean_chan_unrebinned[non_used_mask],
                            background_rate_unrebinned[non_used_mask] / energy_width_unrebinned[non_used_mask],
                            yerr=background_rate_unrebinned_err[non_used_mask] / energy_width_unrebinned[non_used_mask],
                            fmt='',
                            # markersize=3,
                            linestyle='',
                            elinewidth=.7,
                            alpha=.9,
                            capsize=0,
                            # label=data._name,
                            color=threeML_config['ogip']['background color'])

            excluded_channel_plot(ax, energy_min_unrebinned, energy_max_unrebinned,
                                  observed_rate_unrebinned,
                                  background_rate_unrebinned,
                                  self._mask,
                                  self._quality_bad_to_mask(),
                                  show_bad_channels)

        ax.set_xlabel("Energy (keV)")
        ax.set_ylabel("Rate (counts s$^{-1}$ keV$^{-1}$)")
        ax.set_xlim(left=self._rsp.ebounds[0, 0], right=self._rsp.ebounds[-1, 1])
        ax.legend()

    def write_pha(self, file_name, overwrite=False):
        """
        Create a pha file of the current pha selections


        :param file_name: output file name (excluding extension)
        :return: None
        """

        pha_writer = PHAWrite(self)

        pha_writer.write(file_name, overwrite=overwrite)

    def display_rsp(self):

        self._rsp.plot_matrix()

def channel_plot(ax, chan_min, chan_max, counts, **kwargs):
    chans = np.array(zip(chan_min, chan_max))
    width = chan_max - chan_min

    step_plot(chans, counts / width, ax, **kwargs)
    ax.set_xscale('log')
    ax.set_yscale('log')

    return ax


def excluded_channel_plot(ax, chan_min, chan_max, counts, bkg, mask, bad_mask, show_bad_channels):
    # Figure out the best limit

    width = chan_max - chan_min

    top = max([max(bkg / width), max(counts / width)])
    top = top + top * .5
    bottom = min([min(bkg / width), min(counts / width)])
    bottom = bottom - bottom * .2

    # Find the contiguous regions that are deselected
    slices = slice_disjoint((~mask).nonzero()[0])

    for region in slices:
        ax.fill_between([chan_min[region[0]], chan_max[region[1]]],
                        bottom,
                        top,
                        color='k',
                        alpha=.5)

    if show_bad_channels and sum(bad_mask) < len(bad_mask):

        # Find the contiguous regions that are deselected
        slices = slice_disjoint((~bad_mask).nonzero()[0])

        for region in slices:
            ax.fill_between([chan_min[region[0]], chan_max[region[1]]],
                            bottom,
                            top,
                            color='none',
                            edgecolor='limegreen',
                            hatch='/',
                            alpha=1.)

    ax.set_ylim(bottom, top)


def slice_disjoint(arr):
    """
    Returns an array of disjoint indicies.

    Args:
        arr:

    Returns:

    """

    slices = []
    startSlice = arr[0]
    counter = 0
    for i in range(len(arr) - 1):
        if arr[i + 1] > arr[i] + 1:
            endSlice = arr[i]
            slices.append([startSlice, endSlice])
            startSlice = arr[i + 1]
            counter += 1
    if counter == 0:
        return [[arr[0], arr[-1]]]
    if endSlice != arr[-1]:
        slices.append([startSlice, arr[-1]])
    return slices
