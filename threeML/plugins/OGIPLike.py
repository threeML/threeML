from threeML.plugin_prototype import PluginPrototype
from threeML.plugins.OGIP.pha import PHA
from threeML.plugins.OGIP.response import Response
from threeML.io.file_utils import file_existing_and_readable
from threeML.io.step_plot import step_plot
from threeML.plugins.gammaln import logfactorial
from threeML.plugins.OGIP.likelihood_functions import poisson_log_likelihood_ideal_bkg
from threeML.plugins.OGIP.likelihood_functions import poisson_observed_poisson_background
from threeML.plugins.OGIP.likelihood_functions import poisson_observed_gaussian_background

from threeML.config.config import threeML_config

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy

__instrument_name = "All OGIP-compliant instruments"

# This defines the known noise models for source and/or background spectra
_known_noise_models = ['poisson', 'gaussian', 'ideal']


class NotEnoughtCounts(RuntimeError):
    pass


class PrivateMember(RuntimeError):
    pass


class Rebinner(object):
    def __init__(self, vector_to_rebin_on, min_counts):

        tot_counts = np.sum(vector_to_rebin_on)

        if tot_counts < min_counts:

            raise NotEnoughtCounts("Vector contains %s counts, cannot rebin at %s counts per bin" % (tot_counts,
                                                                                                     min_counts))

        self._edges = [-1]
        n = 0

        for index, b in enumerate(vector_to_rebin_on):

            n += b

            if n >= min_counts:

                self._edges.append(index)

                n = 0

        if self._edges[-1] != vector_to_rebin_on.shape[0] - 1:
            self._edges.append(vector_to_rebin_on.shape[0] - 1)

        assert len(self._edges) >= 2

        self._min_counts = min_counts

    @property
    def n_bins(self):
        """
        Returns the number of bins defined.

        :return:
        """

        return len(self._edges) - 1

    def rebin(self, *vectors):

        rebinned_vectors = []

        for vector in vectors:

            rebinned_vector = []

            for low_bound, hi_bound in zip(self._edges[:-1], self._edges[1:]):

                rebinned_vector.append(np.sum(vector[low_bound + 1:hi_bound + 1]))

            # Vector might not contain counts, so we use a relative comparison

            assert abs(np.sum(rebinned_vector) / np.sum(rebinned_vector) - 1) < 1e-4

            rebinned_vectors.append(np.array(rebinned_vector))

        return rebinned_vectors

    def rebin_errors(self, *vectors):

        rebinned_vectors = []

        for vector in vectors:

            rebinned_vector = []

            for low_bound, hi_bound in zip(self._edges[:-1], self._edges[1:]):

                rebinned_vector.append(np.sqrt(np.sum(vector[low_bound + 1:hi_bound + 1] ** 2)))

            # Vector might not contain counts, so we use a relative comparison

            assert abs(np.sum(rebinned_vector) / np.sum(rebinned_vector) - 1) < 1e-4

            rebinned_vectors.append(np.array(rebinned_vector))

        return rebinned_vectors

    @property
    def min_counts(self):

        return self._min_counts

    @min_counts.setter
    def min_counts(self, val):

        raise PrivateMember("Cannot manually set min_counts of rebinner!")

    @min_counts.getter
    def min_counts(self):

        return self._min_counts



    @property
    def edges(self):

        return self._edges

    @edges.getter
    def edges(self):

        # return the low and high bins
        return np.array(self._edges[:-1]) + 1, np.array(self._edges[1:])

    @edges.setter
    def edges(self, val):

        raise PrivateMember("Cannot manually set edges of rebinner!")




class OGIPLike(PluginPrototype):
    def __init__(self, name, pha_file, bak_file=None, rsp_file=None, arf_file=None):

        self._name = str(name)

        # If the user didn't provide them, read the needed files from the keywords in the PHA file

        self._pha = PHA(pha_file)

        # Get the required background file, response and (if present) arf_file either from the
        # calling sequence or the file.
        # NOTE: if one of the file is specified in the calling sequence, it will be used whether or not there is an
        # equivalent specification in the header. This allows the user to override the content of the header of the
        # PHA file, if needed

        if bak_file is None:

            bak_file = self._pha.background_file

            assert bak_file is not None, "No background file provided, and the PHA file does not contain one."

        if rsp_file is None:

            rsp_file = self._pha.response_file

            assert rsp_file is not None, "No response file provided, and the PHA file does not contain one."

        if arf_file is None:

            # Note that his could be None as well, if there is no ancillary file specified in the header

            arf_file = self._pha.ancillary_file

        # Now make sure response and background file exist (the ancillary file is not always present, and will be
        # treated separately

        try:
            assert file_existing_and_readable(
                    bak_file.split("{")[0]), "Background file %s not existing or not readable" % bak_file
        except(AttributeError):
            try:
                if bak_file.is_container:
                    pass
            except:
                RuntimeError("Background file type is unrecognizeable")

        assert file_existing_and_readable(
                rsp_file.split("{")[0]), "Response file %s not existing or not readable" % rsp_file

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

        # Print the autoprobed noise models

        print("Auto-probed noise models:")
        print("- observation: %s" % self.observation_noise_model)
        print("- background: %s" % self.background_noise_model)

    def set_active_measurements(self, *args):
        '''Set the measurements to be used during the analysis.
        Use as many ranges as you need,
        specified as 'emin-emax'. Energies are in keV. Example:

        setActiveMeasurements('10-12.5','56.0-100.0')

        which will set the energy range 10-12.5 keV and 56-100 keV to be
        used in the analysis'''

        # To implement this we will use an array of boolean index,
        # which will filter
        # out the non-used channels during the logLike

        # Now build the new mask: values for which the mask is 0 will be masked

        mask = np.zeros(self._pha.n_channels, dtype=bool)

        # Now parse the input ranges

        for arg in args:

            ee = map(float, arg.replace(" ", "").split("-"))
            emin, emax = sorted(ee)

            idx1 = self._rsp.energy_to_channel(emin)
            idx2 = self._rsp.energy_to_channel(emax)

            mask[idx1:idx2 + 1] = True
            print("Range %s translates to channels %s-%s" % (arg, idx1, idx2))

        self._mask = np.array(mask, np.bool)

        # Check if the spectrum was rebinned on the background.



        print("Now using %s channels out of %s" % (np.sum(self._mask), self._pha.n_channels))

        if self._rebinner is not None:

            print("Previous rebinning Detecting. Now rebinning.")
            self._rebinner = Rebinner(self._background_counts[self._mask], self._rebinner.min_counts)
            print("Using %s bins" % self._rebinner.n_bins)

    def view_count_spectrum(self, plot_errors=True):
        '''
        View the count and background spectrum. Useful to check energy selections.

        '''

        # adding the rebinner: j. michael.

        # First plot the counts

        chans = self._rsp.ebounds[self._mask].T
        chan_min, chan_max = chans

        chan_width = chan_max - chan_min

        if self._observation_noise_model == 'poisson':

            # Observed counts
            observed_counts = copy.copy(self._observed_counts[self._mask])

            cnt_err = np.sqrt(observed_counts)

            if self._background_noise_model == 'poisson':

                background_counts = copy.copy(self._background_counts[self._mask])

                background_errors = np.sqrt(background_counts)

            elif self._background_noise_model == 'ideal':

                background_counts = copy.copy(self._scaled_background_counts[self._mask])

                background_errors = np.zeros_like(background_counts)

            elif self._background_noise_model == 'gaussian':

                background_counts = copy.copy(self._background_counts[self._mask])

                background_errors = copy.copy(self._background_errors[self._mask])

            else:

                raise RuntimeError("This is a bug")

        else:

            raise NotImplementedError("Not yet implemented")

        if self._rebinner is not None:

            observed_counts, background_counts = self._rebinner.rebin(observed_counts, background_counts)
            cnt_err, background_errors = self._rebinner.rebin(cnt_err, background_errors)

            lo, hi = self._rebinner.edges

            chan_min = chan_min[lo]
            chan_max = chan_max[hi]

            chan_width = chan_max - chan_min

            # convert to rates

        observed_counts /= self._pha.exposure
        background_counts /= self._bak.exposure
        cnt_err /= self._pha.exposure
        background_errors /= self._bak.exposure



        _ = channel_plot(chan_min, chan_max, observed_counts,
                         color='#377eb8', lw=1.5, alpha=1, label="Total")
        ax = channel_plot(chan_min, chan_max, background_counts,
                          color='#e41a1c', alpha=.8, label="Background")

        mean_chan = np.mean([chan_min, chan_max], axis=0)

        if plot_errors:
            ax.errorbar(mean_chan,
                        observed_counts / chan_width,
                        yerr=cnt_err / chan_width,
                        fmt='',
                        # markersize=3,
                        linestyle='',
                        elinewidth=.7,
                        alpha=.9,
                        capsize=0,
                        # label=data._name,
                        color='#377eb8')

            ax.errorbar(mean_chan,
                        background_counts / chan_width,
                        yerr=background_errors / chan_width,
                        fmt='',
                        # markersize=3,
                        linestyle='',
                        elinewidth=.7,
                        alpha=.9,
                        capsize=0,
                        # label=data._name,
                        color='#e41a1c')

        # Now fade the non-used channels
        excluded_channel_plot(self._rsp.ebounds[:, 0], self._rsp.ebounds[:, 1], self._mask,
                              self._observed_counts / self._pha.exposure,
                              self._background_counts / self._bak.exposure, ax)



        ax.set_xlabel("Energy (keV)")
        ax.set_ylabel("Rate (counts s$^{-1}$ keV$^{-1}$)")
        ax.set_xlim(left=self._rsp.ebounds[0, 0], right=self._rsp.ebounds[-1, 1])
        ax.legend()

    def _get_expected_background_counts_scaled(self):
        """
        Get the background counts expected in the source interval and in the source region, based on the observed
        background.

        :return:
        """

        # The scale factor is the ratio between the collection area of the source spectrum and the
        # background spectrum. It is used for example for the typical aperture-photometry method used in
        # X-ray astronomy, where the background region has a different size with respect to the source region

        scale_factor = self._pha.scale_factor / self._bak.scale_factor

        # The expected number of counts is the rate in the background file multiplied by its exposure, renormalized
        # by the scale factor.
        # (see http://heasarc.gsfc.nasa.gov/docs/asca/abc_backscal.html)

        bkg_counts = self._bak.rates * self._pha.exposure * scale_factor

        return bkg_counts

    def _loglike_poisson_obs_ideal_bkg(self):

        # Observed counts
        observed_counts = self._observed_counts[self._mask]

        # Model predicted counts
        # In this likelihood the background becomes part of the model, which means that
        # the uncertainty in the background is completely neglected

        bkg_counts = self._scaled_background_counts[self._mask]

        model_counts = self._rsp.convolve()[self._mask] * self._pha.exposure

        if self._rebinner is not None:

            new_counts, new_model, new_bkg = self._rebinner.rebin(observed_counts, model_counts, bkg_counts)

            loglike = poisson_observed_poisson_background(new_counts, new_bkg, new_model)

        else:

            loglike = poisson_observed_poisson_background(observed_counts, bkg_counts, model_counts)

        return loglike

    def _loglike_poisson_obs_poisson_bkg(self):

        # Observed counts
        observed_counts = self._observed_counts[self._mask]

        # Model predicted counts
        # In this likelihood the background becomes part of the model, which means that
        # the uncertainty in the background is completely neglected

        bkg_counts = self._background_counts[self._mask]

        # Scale factor between source and background spectrum

        scale_factor = self._pha.exposure / self._bak.exposure * self._pha.scale_factor / self._bak.scale_factor

        model_counts = self._rsp.convolve()[self._mask] * self._pha.exposure

        # if a rebinner is active, use it

        if self._rebinner is not None:

            new_counts, new_model, new_bkg = self._rebinner.rebin(observed_counts, model_counts, bkg_counts)

            loglike = poisson_observed_poisson_background(new_counts, new_bkg, scale_factor, new_model)

        else:

            loglike = poisson_observed_poisson_background(observed_counts, bkg_counts, scale_factor, model_counts)

        # loglike = poisson_observed_poisson_background(observed_counts, bkg_counts, scale_factor, model_counts)

        # print("predicted: %s (bkg: %s, model: %s), observed: %s" % (np.sum(new_model) + np.sum(new_bkg * scale_factor),
        #                                                             np.sum(new_bkg * scale_factor),
        #                                                             np.sum(new_model),
        #                                                             np.sum(new_counts)))

        return loglike

    def _loglike_poisson_obs_gaussian_bkg(self):

        # Observed counts
        observed_counts = self._observed_counts[self._mask]

        # Model predicted counts
        # In this likelihood the background becomes part of the model, which means that
        # the uncertainty in the background is completely neglected

        background_counts = self._background_counts[self._mask]

        background_errors = self._background_errors[self._mask]

        expected_model_counts = self._rsp.convolve()[self._mask] * self._pha.exposure

        if self._rebinner is not None:

            new_counts, new_model, new_bkg, new_bkg_err = self._rebinner.rebin(observed_counts, expected_model_counts,
                                                                               background_counts, background_errors)

            loglike = poisson_observed_gaussian_background(new_counts, new_bkg, new_bkg_err, new_model)

        else:

            loglike = poisson_observed_gaussian_background(observed_counts, background_counts,
                                                           background_errors, expected_model_counts)

        return loglike

    def rebin_on_background(self, min_number_of_counts):
        """
        Rebin the spectrum guaranteeing the provided minimum number of counts in each background bin. This is usually
        required for spectra with very few background counts to make the profile likelihood meaningful. Of course this
        is not relevant if you treat the background as ideal.

        The observed spectrum will be rebinned in the same fashion as the background spectrum.

        To neutralize this completely, use "remove_rebinning"

        :param min_number_of_counts: the minimum number of counts in each bin
        :return: none
        """

        self._rebinner = Rebinner(self._background_counts[self._mask], min_number_of_counts)

        print("Using %s bins" % self._rebinner.n_bins)

    def remove_rebinning(self):
        """
        Remove the rebinning scheme set with rebin_on_background.

        :return:
        """

        self._rebinner = None

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

    def get_name(self):

        return self._name

    def get_nuisance_parameters(self):

        return {}

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


def display_model_counts(*args, **kwargs):
    """

    Display the fitted model count spectrum

    Returns:

    """

    try:

        min_rate = kwargs.pop('min_rate')

    except:

        min_rate = 1

    fig = plt.figure(666)

    # gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])

    # gs.update(hspace=0)

    ax = fig.add_subplot(111)
    # ax = plt.subplot(gs[0])
    divider = make_axes_locatable(ax)
    ax1 = divider.append_axes('bottom', size=.75, pad=0., sharex=ax)

    color = np.linspace(0., 1., len(args))
    color_itr = 0

    data_cmap = plt.cm.rainbow
    model_cmap = plt.cm.nipy_spectral_r

    for data in args:

        chans = data._rsp.ebounds[data._mask].T
        chan_min, chan_max = chans

        if data._observation_noise_model == 'poisson':

            # Observed counts
            observed_counts = data._observed_counts[data._mask]

            cnt_err = np.sqrt(observed_counts)

            if data._background_noise_model == 'poisson':

                background_counts = data._background_counts[data._mask]

                background_errors = np.sqrt(background_counts)

            elif data._background_noise_model == 'ideal':

                background_counts = data._scaled_background_counts[data._mask]

                background_errors = np.zeros_like(background_counts)

            elif data._background_noise_model == 'gaussian':

                background_counts = data._background_counts[data._mask]

                background_errors = data._background_errors[data._mask]

            else:

                raise RuntimeError("This is a bug")

        else:

            raise NotImplementedError("Not yet implemented")

        chan_width = chan_max - chan_min

        expected_model_rate = data._rsp.convolve()[data._mask] / chan_width  #* data._pha.exposure

        # I must fit this! the back exposure is wrong

        src_rate = (observed_counts / data._pha.exposure - background_counts / data._bak.exposure) / (chan_width)
        src_rate_err = np.sqrt((cnt_err / (data._pha.exposure * chan_width)) ** 2 + (
            background_errors / (data._bak.exposure * chan_width)) ** 2)

        this_rebinner = Rebinner(src_rate, min_rate)

        new_rate, new_model_rate = this_rebinner.rebin(src_rate, expected_model_rate)
        new_err, = this_rebinner.rebin_errors(src_rate_err)

        lo, hi = this_rebinner.edges
        chan_min = chan_min[lo]
        chan_max = chan_max[hi]



        mean_chan = np.mean([chan_min, chan_max], axis=0)

        ax.errorbar(mean_chan,
                    new_rate,
                    yerr=new_err,
                    fmt='.',
                    markersize=3,
                    linestyle='',
                    # elinewidth=.5,
                    alpha=.9,
                    capsize=0,
                    label=data._name,
                    color=data_cmap(color[color_itr]))

        step_plot(np.asarray(zip(chan_min, chan_max)), new_model_rate,
                  ax, alpha=.8,
                  label='%s Model' % data._name, color=model_cmap(color[color_itr]))

        # Residuals
        # ax1 = fig.add_subplot(gs[1])

        residuals = (new_model_rate - new_rate) / new_model_rate

        ax1.axhline(0, linestyle='--', color='k')
        ax1.errorbar(mean_chan,
                     residuals,
                     yerr=new_err / new_model_rate,
                     capsize=0,
                     fmt='.',
                     markersize=3,
                     color=data_cmap(color[color_itr]))

        color_itr += 1

    ax.legend(fontsize='x-small')

    # ax.set_xlabel("Energy (keV)")
    ax.set_ylabel(r"Rate (counts s$^{-1}$ keV$^{-1}$)")

    ax.set_xscale('log')
    ax.set_yscale('log', nonposy='clip')

    # ax.set_xlim(left=data._rsp.ebounds[0, 0], right=data._rsp.ebounds[-1, 1])


    ax1.set_xscale("log")

    locator = MaxNLocator(prune='upper', nbins=5)
    ax1.yaxis.set_major_locator(locator)

    ax1.set_xlabel("Energy (MeV)")
    ax1.set_ylabel("Residuals")

    # ax.set_xticks([])


def channel_plot(chan_min, chan_max, counts, **kwargs):
    chans = np.array(zip(chan_min, chan_max))
    width = chan_max - chan_min
    fig = plt.figure(666)
    ax = fig.add_subplot(111)
    step_plot(chans, counts / width, ax, **kwargs)
    ax.set_xscale('log')
    ax.set_yscale('log')

    return ax


def excluded_channel_plot(chan_min, chan_max, mask, counts, bkg, ax):
    # Figure out the best limit
    chans = np.array(zip(chan_min, chan_max))
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
    slices = []
    startSlice = 0
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
