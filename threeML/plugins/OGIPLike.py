import collections
import copy

import matplotlib.pyplot as plt
import numpy as np
from astromodels.parameter import Parameter
from astromodels.functions.functions import Uniform_prior
from astromodels.utils.valid_variable import is_valid_variable_name
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

from threeML.io.file_utils import file_existing_and_readable, sanitize_filename
from threeML.io.step_plot import step_plot
from threeML.plugin_prototype import PluginPrototype
from threeML.plugins.OGIP.likelihood_functions import poisson_log_likelihood_ideal_bkg
from threeML.plugins.OGIP.likelihood_functions import poisson_observed_gaussian_background
from threeML.plugins.OGIP.likelihood_functions import poisson_observed_poisson_background
from threeML.plugins.OGIP.pha import PHA
from threeML.plugins.OGIP.response import Response
from threeML.utils.Binner import Rebinner

__instrument_name = "All OGIP-compliant instruments"

# This defines the known noise models for source and/or background spectra
_known_noise_models = ['poisson', 'gaussian', 'ideal']



class OGIPLike(PluginPrototype):

    def __init__(self, name, pha_file, bak_file=None, rsp_file=None, arf_file=None):

        assert is_valid_variable_name(name), "Name %s is not a valid name for a plugin. You must use a name which is " \
                                             "a valid python identifier: no spaces, no operators (+,-,/,*), " \
                                             "it cannot start with a number, no special characters" % name

        # If the user didn't provide them, read the needed files from the keywords in the PHA file
        # It is possible a user passed a PHAContainer from an EventList. In this case, this will fail
        # resulting in an attribute error. We will check for this and if it fails, and then try to load the
        # PHAContainer

        try:

            pha_file = sanitize_filename(pha_file)

            self._pha = PHA(pha_file)

        except(AttributeError):

            try:

                if pha_file.is_container:

                    self._pha = PHA(pha_file)

                else:

                    raise RuntimeError("Should never arrive here. Your PHA file is improper")

            except:

                # Catch everything

                raise RuntimeError("Your PHA file is invalid.")

        # Get the required background file, response and (if present) arf_file either from the
        # calling sequence or the file.
        # NOTE: if one of the file is specified in the calling sequence, it will be used whether or not there is an
        # equivalent specification in the header. This allows the user to override the content of the header of the
        # PHA file, if needed

        if bak_file is None:

            bak_file = self._pha.background_file

            assert bak_file is not None, "No background file provided, and the PHA file does not contain one."

            bak_file = sanitize_filename(bak_file)

        if rsp_file is None:

            rsp_file = self._pha.response_file

            assert rsp_file is not None, "No response file provided, and the PHA file does not contain one."

            rsp_file = sanitize_filename(bak_file)

        if arf_file is None:

            # Note that his could be None as well, if there is no ancillary file specified in the header

            arf_file = self._pha.ancillary_file

            if arf_file is not None:

                arf_file = sanitize_filename(arf_file)

        # Now make sure response and background file exist (the ancillary file is not always present, and will be
        # treated separately)

        try:

            assert file_existing_and_readable(bak_file.split("{")[0]), "Background file %s not existing or not " \
                                                                       "readable" % bak_file

        except(AttributeError):

            try:

                if bak_file.is_container:
                    pass

            except:

                RuntimeError("Background file type is unrecognizeable")

        assert file_existing_and_readable(rsp_file.split("{")[0]), "Response file %s not existing or not " \
                                                                   "readable" % rsp_file

        self._bak = PHA(bak_file, file_type='background')
        self._rsp = Response(rsp_file, arf_file=arf_file)

        # Make sure that data and background have the same number of channels

        assert self._pha.n_channels == self._bak.n_channels, "Data file and background file have different " \
                                                             "number of channels"

        # Precomputed observed and background counts (for speed)

        self._observed_counts = self._pha.rates * self._pha.exposure
        self._background_counts = self._bak.rates * self._bak.exposure
        self._scaled_background_counts = self._get_expected_background_counts_scaled()

        # Init everything else to None
        self._like_model = None
        self._rebinner = None

        # Now auto-probe the statistic to use
        if self._pha.is_poisson():

            if self._bak.is_poisson():

                self.observation_noise_model = 'poisson'
                self.background_noise_model = 'poisson'

                self._background_errors = None

                assert np.all(self._observed_counts >= 0), "Error in PHA: negative counts!"

                assert np.all(self._background_counts >= 0), "Error in background spectrum: negative counts!"

            else:

                self.observation_noise_model = 'poisson'
                self.background_noise_model = 'gaussian'

                self._background_errors = self._bak.rate_errors * self._bak.exposure

                idx = self._background_errors == 0

                assert np.all(self._background_errors[idx] == self._background_counts[idx]), \
                    "Error in background spectrum: if the error on the background is zero, " \
                    "also the expected background must be zero"

                assert np.all(self._background_counts >= 0), "Error in background spectrum: negative background!"

        else:

            raise NotImplementedError("Gaussian observation is not yet supported")

        # Initialize a mask that selects all the data

        self._mask = np.asarray(np.ones(self._pha.n_channels), np.bool)

        # Print the autoprobed noise models

        print("Auto-probed noise models:")
        print("- observation: %s" % self.observation_noise_model)
        print("- background: %s" % self.background_noise_model)

        # Now create the nuisance parameter for the effective area correction, which is fixed
        # by default. This factor multiplies the model so that it can account for calibration uncertainties on the
        # global effective area. By default it is limited to stay within 20%

        self._nuisance_parameter = Parameter("cons_%s" % name, 1.0, min_value = 0.8, max_value = 1.2, delta=0.05,
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
        self._current_background_errors = self._background_errors

    def set_active_measurements(self, *args):
        """
        Set the measurements to be used during the analysis.
        Use as many ranges as you need,
        specified as 'emin-emax'. Energies are in keV. Example:

        set_active_measurements('10-12.5','56.0-100.0')

        which will set the energy range 10-12.5 keV and 56-100 keV to be
        used in the analysis

        If working on data which has been rebinned, the selection will be
        adjusted properly

        """

        # To implement this we will use an array of boolean index,
        # which will filter
        # out the non-used channels during the logLike

        # Now build the new mask: values for which the mask is 0 will be masked

        # We will build the high res mask even if we are
        # already rebinned so that it can be saved

        assert self._rebinner is None, "You cannot select active measurements if you have a rebinning active. " \
                                       "Remove it first with remove_rebinning"

        self._mask = np.zeros(self._pha.n_channels, dtype=bool)

        for arg in args:

            ee = map(float, arg.replace(" ", "").split("-"))
            emin, emax = sorted(ee)

            idx1 = self._rsp.energy_to_channel(emin)
            idx2 = self._rsp.energy_to_channel(emax)

            self._mask[idx1:idx2 + 1] = True

            print("Range %s translates to channels %s-%s" % (arg, idx1, idx2))

        print("Now using %s channels out of %s" % (np.sum(self._mask), self._pha.n_channels))

        # Apply the mask
        self._apply_mask_to_original_vectors()

    def _apply_mask_to_original_vectors(self):

        # Apply the mask

        self._current_observed_counts = self._observed_counts[self._mask]
        self._current_background_counts = self._background_counts[self._mask]
        self._current_scaled_background_counts = self._scaled_background_counts[self._mask]

        if self._background_errors is not None:

            self._current_background_errors = self._background_errors[self._mask]

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

        self._rebinner = Rebinner(self._background_counts, min_number_of_counts, self._mask)

        # Apply the rebinning to everything.
        # NOTE: the output of the .rebin method are the vectors with the mask *already applied*

        (self._current_observed_counts,
         self._current_background_counts,
         self._current_scaled_background_counts) = self._rebinner.rebin(self._observed_counts,
                                                                        self._background_counts,
                                                                        self._scaled_background_counts)

        if self._background_errors is not None:

            # NOTE: the output of the .rebin method are the vectors with the mask *already applied*

            self._current_background_errors, = self._rebinner.rebin_errors(self._background_errors)

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

        loglike = poisson_log_likelihood_ideal_bkg(self._current_observed_counts,
                                                   self._current_scaled_background_counts,
                                                   model_counts)

        return loglike

    def _loglike_poisson_obs_poisson_bkg(self):

        # Scale factor between source and background spectrum

        scale_factor = self._pha.exposure / self._bak.exposure * self._pha.scale_factor / self._bak.scale_factor

        model_counts = self.get_folded_model()

        loglike = poisson_observed_poisson_background(self._current_observed_counts,
                                                      self._current_background_counts,
                                                      scale_factor,
                                                      model_counts)

        return loglike

    def _loglike_poisson_obs_gaussian_bkg(self):

        expected_model_counts = self.get_folded_model()

        loglike = poisson_observed_gaussian_background(self._current_observed_counts,
                                                       self._current_background_counts,
                                                       self._current_background_errors,
                                                       expected_model_counts)

        return loglike

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

    def get_log_like(self):

        if self._observation_noise_model == 'poisson':

            if self._background_noise_model == 'poisson':

                loglike = self._loglike_poisson_obs_poisson_bkg()

            elif self._background_noise_model == 'ideal':

                loglike = self._loglike_poisson_obs_ideal_bkg()

            elif self._background_noise_model == 'gaussian':

                loglike = self._loglike_poisson_obs_gaussian_bkg()

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

    def view_count_spectrum(self, plot_errors=True):
        """
        View the count and background spectrum. Useful to check energy selections.
        Args:
            plot_errors: bool to choose error plotting

        Returns:
            none

        """

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

                background_errors = copy.copy(self._current_background_errors)

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
                         color='#377eb8', lw=1.5, alpha=1)

            channel_plot(ax, energy_min_unrebinned[non_used_mask],
                         energy_max_unrebinned[non_used_mask],
                         background_rate_unrebinned[non_used_mask],
                         color='#e41a1c', alpha=.8)

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
                            color='#377eb8')

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
                            color='#e41a1c')

            excluded_channel_plot(ax, energy_min_unrebinned, energy_max_unrebinned, self._mask,
                                  observed_rate_unrebinned,
                                  background_rate_unrebinned)

        ax.set_xlabel("Energy (keV)")
        ax.set_ylabel("Rate (counts s$^{-1}$ keV$^{-1}$)")
        ax.set_xlim(left=self._rsp.ebounds[0, 0], right=self._rsp.ebounds[-1, 1])
        ax.legend()


def display_model_counts(*args, **kwargs):
    """

    Display the fitted model count spectrum of one or more OGIP plugins

    :param args: one or more instances of OGIP plugin
    :param min_rate: (optional) rebin to keep this minimum rate in each channel (if possible). If one number is
    provided, the same minimum rate is used for each dataset, otherwise a list can be provided with the minimum rate
    for each dataset
    :param data_cmap: (optional) the color map used to extract automatically the colors for the data
    :param model_cmap: (optional) the color map used to extract automatically the colors for the models
    :param data_colors: (optional) a tuple or list with the color for each dataset
    :param model_colors: (optional) a tuple or list with the color for each folded model
    :param show_legend: (optional) if True (default), shows a legend
    :return: figure instance
    """

    # default settings

    min_rates = [1e-99] * len(args)

    data_cmap = plt.cm.rainbow
    model_cmap = plt.cm.nipy_spectral_r

    # Legend is on by default
    show_legend = True

    # Default colors

    data_colors = map(lambda x:data_cmap(x), np.linspace(0.0, 1.0, len(args)))
    model_colors = map(lambda x:model_cmap(x), np.linspace(0.0, 1.0, len(args)))

    # Now override defaults according to the optional keywords, if present

    if 'show_legend' in kwargs:

        show_legend = bool(kwargs.pop('show_legend'))

    if 'min_rate' in kwargs:

        min_rate = kwargs.pop('min_rate')

        # If min_rate is a floating point, use the same for all datasets, otherwise use the provided ones

        try:

            min_rate = float(min_rate)

        except TypeError:

            min_rates = list(min_rate)

            assert len(min_rates) >= len(args), "If you provide different minimum rates for each data set, you need" \
                                                "to provide an iterable of the same length of the number of datasets"

    if 'data_cmap' in kwargs:

        data_cmap = kwargs.pop('data_cmap')
        data_colors = map(lambda x: data_cmap(x), np.linspace(0.0, 1.0, len(args)))

    if 'model_cmap' in kwargs:

        model_cmap = kwargs.pop('model_cmap')
        model_colors = map(lambda x: model_cmap(x), np.linspace(0.0, 1.0, len(args)))

    if 'data_colors' in kwargs:

        data_colors = kwargs.pop('data_colors')

        assert len(data_colors) >= len(args), "You need to provide at least a number of data colors equal to the " \
                                               "number of datasets"

    if 'model_colors' in kwargs:

        model_colors = kwargs.pop('model_colors')

        assert len(model_colors) >= len(args), "You need to provide at least a number of model colors equal to the " \
                                                "number of datasets"

    fig, (ax, ax1) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    #divider = make_axes_locatable(ax)

    #ax1 = divider.append_axes('bottom', size=1.75, pad=0., sharex=ax)

    # go thru the detectors
    for data, data_color, model_color, min_rate in zip(args, data_colors, model_colors, min_rates):

        # NOTE: we use the original (unmasked) vectors because we need to rebin ourselves the data later on

        energy_min, energy_max = data._rsp.ebounds.T

        # figure out the type of data

        if data._observation_noise_model == 'poisson':

            # Observed counts
            observed_counts = data._observed_counts

            cnt_err = np.sqrt(observed_counts)

            if data._background_noise_model == 'poisson':

                background_counts = data._background_counts

                # Gehrels weighting, a little bit better approximation when statistic is low
                # (and inconsequential when statistic is high)

                background_errors = 1 + np.sqrt(background_counts + 0.75)

            elif data._background_noise_model == 'ideal':

                background_counts = data._scaled_background_counts

                background_errors = np.zeros_like(background_counts)

            elif data._background_noise_model == 'gaussian':

                background_counts = data._background_counts

                background_errors = data._background_errors

            else:

                raise RuntimeError("This is a bug")

        else:

            raise NotImplementedError("Not yet implemented")

        chan_width = energy_max - energy_min

        # get the expected counts
        # NOTE: _rsp.convolve() returns already the rate (counts / s)
        expected_model_rate = data._nuisance_parameter.value * data._rsp.convolve() # * data.exposure  / data.exposure

        # calculate all the correct quantites

        # since we compare to the model rate... background subtract but with proper propagation

        src_rate = (observed_counts / data.exposure - background_counts / data.background_exposure)

        src_rate_err = np.sqrt((cnt_err / data.exposure) ** 2 +
                               (background_errors / data.background_exposure) ** 2)

        # rebin on the source rate
        this_rebinner = Rebinner(src_rate, min_rate, data._mask)

        # get the rebinned counts
        new_rate, new_model_rate = this_rebinner.rebin(src_rate, expected_model_rate)
        new_err, = this_rebinner.rebin_errors(src_rate_err)

        # adjust channels
        new_energy_min, new_energy_max = this_rebinner.get_new_start_and_stop(energy_min, energy_max)
        new_chan_width = new_energy_max - new_energy_min


        mean_energy = np.mean([new_energy_min, new_energy_max], axis=0)
        delta_energy = new_energy_max - new_energy_min

        ax.errorbar(mean_energy,
                    new_rate / new_chan_width,
                    yerr=new_err / new_chan_width,
                    xerr=delta_energy/2.0,
                    fmt='.',
                    markersize=3,
                    linestyle='',
                    # elinewidth=.5,
                    alpha=.9,
                    capsize=0,
                    label=data._name,
                    color=data_color)

        step_plot(np.asarray(zip(energy_min[data._mask], energy_max[data._mask])),
                  expected_model_rate[data._mask] / chan_width[data._mask],
                  ax, alpha=.8,
                  label='%s Model' % data._name, color=model_color)

        # Residuals
        # ax1 = fig.add_subplot(gs[1])


        # is this the best way to do residuals?
        residuals = (new_model_rate - new_rate) / new_model_rate

        ax1.axhline(0, linestyle='--', color='k')
        ax1.errorbar(mean_energy,
                     residuals,
                     yerr=new_err / new_model_rate,
                     capsize=0,
                     fmt='.',
                     markersize=3,
                     color=data_color)

    if show_legend:

        ax.legend(fontsize='x-small', loc=0)

    ax.set_ylabel("Background-subtracted rate\n(counts s$^{-1}$ keV$^{-1}$)")

    ax.set_xscale('log')
    ax.set_yscale('log', nonposy='clip')

    ax1.set_xscale("log")

    locator = MaxNLocator(prune='upper', nbins=5)
    ax1.yaxis.set_major_locator(locator)

    ax1.set_xlabel("Energy\n(keV)")
    ax1.set_ylabel("Residuals\n($\sigma$)")

    # This takes care of making space for all labels around the figure

    fig.tight_layout()

    # Now remove the space between the two subplots
    # NOTE: this must be placed *after* tight_layout, otherwise it will be ineffective

    fig.subplots_adjust(hspace=0)

    return fig


def channel_plot(ax, chan_min, chan_max, counts, **kwargs):

    chans = np.array(zip(chan_min, chan_max))
    width = chan_max - chan_min

    step_plot(chans, counts / width, ax, **kwargs)
    ax.set_xscale('log')
    ax.set_yscale('log')

    return ax


def excluded_channel_plot(ax, chan_min, chan_max, mask, counts, bkg):
    # Figure out the best limit

    width = chan_max - chan_min

    top = max([max(bkg / width), max(counts / width)])
    top = top + top * .5
    bottom = min([min(bkg / width), min(counts / width)])
    bottom = bottom - bottom * .2

    # Find the contiguous regions
    slices = slice_disjoint((~mask).nonzero()[0])

    for region in slices:
        ax.fill_between([chan_min[region[0]], chan_max[region[1]]],
                        bottom,
                        top,
                        color='k',
                        alpha=.5)

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
