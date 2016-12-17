import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from threeML.plugins.OGIPLike import OGIPLike
from threeML.utils.binner import Rebinner
from threeML.io.step_plot import step_plot
from threeML.utils.stats_tools import Significance
from threeML.exceptions.custom_exceptions import custom_warnings

from threeML.config.config import threeML_config

NO_REBIN = 1e-99

def display_ogip_model_counts(analysis, data=(), **kwargs):
    """

    Display the fitted model count spectrum of one or more OGIP plugins

    NOTE: all parameters passed as keyword arguments that are not in the list below, will be passed as keyword arguments
    to the plt.subplots() constructor. So for example, you can specify the size of the figure using figsize = (20,10)

    :param args: one or more instances of OGIP plugin
    :param min_rate: (optional) rebin to keep this minimum rate in each channel (if possible). If one number is
    provided, the same minimum rate is used for each dataset, otherwise a list can be provided with the minimum rate
    for each dataset
    :param data_cmap: (str) (optional) the color map used to extract automatically the colors for the data
    :param model_cmap: (str) (optional) the color map used to extract automatically the colors for the models
    :param data_colors: (optional) a tuple or list with the color for each dataset
    :param model_colors: (optional) a tuple or list with the color for each folded model
    :param show_legend: (optional) if True (default), shows a legend
    :param step: (optional) if True (default), show the folded model as steps, if False, the folded model is plotted
    with linear interpolation between each bin
    :return: figure instance


    """

    # If the user supplies a subset of the data, we will use that

    if not data:

        data_keys = analysis.data_list.keys()

    else:

        data_keys = data

    # Now we want to make sure that we only grab OGIP plugins

    new_data_keys = []

    for key in data_keys:

        # Make sure it is a valid key
        if key in analysis.data_list.keys():

            if isinstance(analysis.data_list[key], OGIPLike):

                new_data_keys.append(key)

            else:

                custom_warnings.warn("Dataset %s is not of the OGIP kind. Cannot be plotted by "
                                     "display_ogip_model_counts" % key)

    if not new_data_keys:

        RuntimeError(
                'There were no valid OGIP data requested for plotting. Please use the detector names in the data list')

    data_keys = new_data_keys

    # default settings

    # Default is to show the model with steps
    step = True

    data_cmap = plt.get_cmap(threeML_config['ogip']['data plot cmap'])  # plt.cm.rainbow
    model_cmap = plt.get_cmap(threeML_config['ogip']['model plot cmap'])  # plt.cm.nipy_spectral_r

    # Legend is on by default
    show_legend = True

    # Default colors

    data_colors = map(lambda x: data_cmap(x), np.linspace(0.0, 1.0, len(data_keys)))
    model_colors = map(lambda x: model_cmap(x), np.linspace(0.0, 1.0, len(data_keys)))

    # Now override defaults according to the optional keywords, if present

    if 'show_legend' in kwargs:

        show_legend = bool(kwargs.pop('show_legend'))

    if 'step' in kwargs:

        step = bool(kwargs.pop('step'))


    if 'min_rate' in kwargs:

        min_rate = kwargs.pop('min_rate')

        # If min_rate is a floating point, use the same for all datasets, otherwise use the provided ones

        try:

            min_rate = float(min_rate)

            min_rates = [min_rate] * len(data_keys)

        except TypeError:

            min_rates = list(min_rate)

            assert len(min_rates) >= len(
                    data_keys), "If you provide different minimum rates for each data set, you need" \
                                "to provide an iterable of the same length of the number of datasets"

    else:

        # This is the default (no rebinning)

        min_rates = [NO_REBIN] * len(data_keys)

    if 'data_cmap' in kwargs:

        data_cmap = plt.get_cmap(kwargs.pop('data_cmap'))
        data_colors = map(lambda x: data_cmap(x), np.linspace(0.0, 1.0, len(data_keys)))

    if 'model_cmap' in kwargs:

        model_cmap = kwargs.pop('model_cmap')
        model_colors = map(lambda x: model_cmap(x), np.linspace(0.0, 1.0, len(data_keys)))

    if 'data_colors' in kwargs:

        data_colors = kwargs.pop('data_colors')

        assert len(data_colors) >= len(data_keys), "You need to provide at least a number of data colors equal to the " \
                                                   "number of datasets"

    if 'model_colors' in kwargs:

        model_colors = kwargs.pop('model_colors')

        assert len(model_colors) >= len(
                data_keys), "You need to provide at least a number of model colors equal to the " \
                            "number of datasets"

    # Now we need to set the model back to the best or median fit depending
    # on the type of analysis

    # GV: I would leave this to the user: if he/she wants to see the best fit and she did something to the model
    # between the fit and when this function is called, then he/she can restore manually the fit.
    # This way it is possible to use this method even before a fit, for example to
    # check if the chosen normalizations have anything to do with the data.

    # if analysis.analysis_type == 'mle':
    #
    #     analysis.restore_best_fit()
    #
    # else:
    #
    #     analysis.restore_median_fit()

    fig, (ax, ax1) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [2, 1]}, **kwargs)

    # go thru the detectors
    for key, data_color, model_color, min_rate in zip(data_keys, data_colors, model_colors, min_rates):

        # NOTE: we use the original (unmasked) vectors because we need to rebin ourselves the data later on

        data = analysis.data_list[key]

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

                background_errors = data._back_counts_errors

            else:

                raise RuntimeError("This is a bug")

        else:

            raise NotImplementedError("Not yet implemented")

        chan_width = energy_max - energy_min

        # get the expected counts
        # NOTE: _rsp.convolve() returns already the rate (counts / s)
        expected_model_rate = data._nuisance_parameter.value * data._rsp.convolve()  # * data.exposure  / data.exposure

        # calculate all the correct quantites

        # since we compare to the model rate... background subtract but with proper propagation
        src_rate = (observed_counts / data.exposure - background_counts / data.background_exposure)

        src_rate_err = np.sqrt((cnt_err / data.exposure) ** 2 +
                               (background_errors / data.background_exposure) ** 2)

        # rebin on the source rate

        # Create a rebinner if either a min_rate has been given, or if the current data set has no rebinned on its own

        if (min_rate is not NO_REBIN) or (data._rebinner is None):

            this_rebinner = Rebinner(src_rate, min_rate, data._mask)

        else:

            # Use the rebinner already in the data
            this_rebinner = data._rebinner

        # get the rebinned counts
        new_rate, new_model_rate = this_rebinner.rebin(src_rate, expected_model_rate)
        new_err, = this_rebinner.rebin_errors(src_rate_err)

        # adjust channels
        new_energy_min, new_energy_max = this_rebinner.get_new_start_and_stop(energy_min, energy_max)
        new_chan_width = new_energy_max - new_energy_min

        # mean_energy = np.mean([new_energy_min, new_energy_max], axis=0)

        # For each bin find the weighted average of the channel center
        mean_energy = []
        delta_energy = [[],[]]
        mean_energy_unrebinned = (energy_max + energy_min)/2.0

        for e_min, e_max in zip(new_energy_min, new_energy_max):

            # Find all channels in this rebinned bin
            idx = (mean_energy_unrebinned >= e_min) & (mean_energy_unrebinned <= e_max)

            # Find the rates for these channels
            r = src_rate[idx]

            if r.max() == 0:

                # All empty, cannot weight
                this_mean_energy = (e_min + e_max) / 2.0

            else:

                # Do the weighted average of the mean energies
                weights = r / np.sum(r)

                this_mean_energy = np.average(mean_energy_unrebinned[idx], weights=weights)

            # Compute "errors" for X (which aren't really errors, just to mark the size of the bin)

            delta_energy[0].append(this_mean_energy - e_min)
            delta_energy[1].append(e_max - this_mean_energy)
            mean_energy.append(this_mean_energy)

        ax.errorbar(mean_energy,
                    new_rate / new_chan_width,
                    yerr=new_err / new_chan_width,
                    xerr=delta_energy,
                    fmt='.',
                    markersize=3,
                    linestyle='',
                    # elinewidth=.5,
                    alpha=.9,
                    capsize=0,
                    label=data._name,
                    color=data_color)

        if step:

            step_plot(np.asarray(zip(new_energy_min, new_energy_max)),
                      new_model_rate / new_chan_width,
                      ax, alpha=.8,
                      label='%s Model' % data._name, color=model_color)

        else:

            # We always plot the model un-rebinned here

            # Mask the array so we don't plot the model where data have been excluded
            # y = expected_model_rate / chan_width
            y = np.ma.masked_where(~data._mask, expected_model_rate / chan_width)

            x = np.mean([energy_min, energy_max], axis=0)

            ax.plot(x, y, alpha=.8, label='%s Model' % data._name, color=model_color)

        # Residuals

        # we need to get the rebinned counts
        rebinned_observed_counts, = this_rebinner.rebin(observed_counts)

        # the rebinned counts expected from the model
        rebinned_model_counts = new_model_rate * data.exposure

        # and also the rebinned background

        rebinned_background_counts, = this_rebinner.rebin(background_counts)
        rebinned_background_errors, = this_rebinner.rebin_errors(background_errors)

        significance_calc = Significance(rebinned_observed_counts,
                                         rebinned_background_counts + rebinned_model_counts / data.scale_factor,
                                         data.scale_factor)

        # Divide the various cases

        if data._observation_noise_model == 'poisson':

            if data._background_noise_model == 'poisson':

                # We use the Li-Ma formula to get the significance (sigma)

                residuals = significance_calc.li_and_ma()

            elif data._background_noise_model == 'ideal':

                residuals = significance_calc.known_background()

            elif data._background_noise_model == 'gaussian':

                residuals = significance_calc.li_and_ma_equivalent_for_gaussian_background(rebinned_background_errors)

            else:

                raise RuntimeError("This is a bug")

        else:

            raise NotImplementedError("Not yet implemented")


        ax1.axhline(0, linestyle='--', color='k')
        ax1.errorbar(mean_energy,
                     residuals,
                     yerr=np.ones_like(residuals),
                     capsize=0,
                     fmt='.',
                     markersize=3,
                     color=data_color)

    if show_legend:

        ax.legend(fontsize='x-small', loc=0)

    ax.set_ylabel("Net rate\n(counts s$^{-1}$ keV$^{-1}$)")

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

