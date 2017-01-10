__author__ = 'grburgess'

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import quantity_support

from threeML.config.config import threeML_config
from threeML.io.calculate_flux import _setup_analysis_dictionaries
from threeML.io.plotting.cmap_cycle import cmap_intervals

def plot_point_source_spectra(*analyses, **kwargs):
    """

    plotting routine for fitted point source spectra


    :param analyses: fitted JointLikelihood or BayesianAnalysis objects
    :param sources_to_use: (optional) list of PointSource string names to plot from the analysis
    :param energy_unit: (optional) astropy energy unit in string form (can also be frequency)
    :param flux_unit: (optional) astropy flux unit in string form
    :param ene_min: (optional) minimum energy to plot
    :param ene_max: (optional) maximum energy to plot
    :param num_ene: (optional) number of energies to plot
    :param use_components: (optional) True or False to plot the spectral components
    :param components_to_use: (optional) list of string names of the components to plot: including 'total'
    will also plot the total spectrum
    :param sum_sources: (optional) some all the MLE and Bayesian sources
    :param show_contours: (optional) True or False to plot the contour region
    :param plot_style_kwargs: (optional) dictionary of MPL plot styling for the best fit curve
    :param contour_style_kwargs: (optional) dictionary of MPL plot styling for the contour regions
    :param fit_cmap: MPL color map to iterate over for plotting multiple analyses
    :param contour_cmap: MPL color map to iterate over for plotting contours for  multiple analyses
    :return:
    """

    # allow matplotlib to plot quantities to the access

    quantity_support()

    # get the default color maps
    if 'fit_cmap' in kwargs:

        fit_cmap = kwargs.pop('fit_cmap')

    else:

        fit_cmap = threeML_config['model plot']['point source plot']['fit cmap']

    if 'contour_cmap' in kwargs:

        contour_cmap = kwargs.pop('contour_cmap')

    else:

        contour_cmap = threeML_config['model plot']['point source plot']['contour cmap']

    if 'sigma' in kwargs:

        sigma = kwargs.pop('sigma')

    else:

        sigma = 1

    if 'energy_unit' in kwargs:

        energy_unit = kwargs.pop('energy_unit')

    else:

        energy_unit = 'keV'

    if 'flux_unit' in kwargs:

        flux_unit = kwargs.pop('flux_unit')

    else:

        flux_unit = '1/(cm2 s keV)'

    if 'ene_min' in kwargs:

        ene_min = kwargs.pop('ene_min')

    else:

        ene_min = 10.

    if 'ene_max' in kwargs:

        ene_max = kwargs.pop('ene_max')

    else:

        ene_max = 40000

    if 'num_ene' in kwargs:

        num_ene = kwargs.pop('num_ene')

    else:

        num_ene = 200

    if 'fraction_of_samples' in kwargs:

        fraction_of_samples = kwargs.pop('fraction_of_samples')

    else:

        fraction_of_samples = 0.1

    if 'use_components' in kwargs:

        use_components = kwargs.pop('use_components')

    else:

        use_components = False

    if 'components_to_use' in kwargs:

        components_to_use = kwargs.pop('components_to_use')

    else:

        components_to_use = []

    if 'sum_sources' in kwargs:

        sum_sources = kwargs.pop('sum_sources')

    else:

        sum_sources = False

    if 'show_contours' in kwargs:

        show_contours = kwargs.pop('show_contours')

    else:

        show_contours = True

    if 'plot_style_kwargs' in kwargs:

        plot_style_kwargs = kwargs.pop('plot_style_kwargs')

    else:

        plot_style_kwargs = threeML_config['model plot']['point source plot']['plot style']

    if 'contour_style_kwargs' in kwargs:

        contour_style_kwargs = kwargs.pop('contour_style_kwargs')

    else:

        contour_style_kwargs = threeML_config['model plot']['point source plot']['contour style']

    if 'show_legend' in kwargs:

        show_legend = kwargs.pop('show_legend')

    else:

        show_legend = True

    if 'legend_kwargs' in kwargs:

        legend_kwargs = kwargs.pop('legend_kwargs')

    else:

        legend_kwargs = threeML_config['model plot']['point source plot']['legend style']
    # first extract the sources to plots



    energy_range = np.logspace(np.log10(ene_min), np.log10(ene_max), num_ene)

    mle_analyses, bayesian_analyses, num_sources_to_plot, sources_to_use, duplicate_keys = _setup_analysis_dictionaries(
        analyses,
        energy_range,
        energy_unit,
        flux_unit,
        use_components,
        components_to_use,
        sigma,
        fraction_of_samples,
        differential=True,
        **kwargs)

    # we are now ready to plot.
    # all calculations have been made.

    fig, ax = plt.subplots()

    color_fit = cmap_intervals(num_sources_to_plot, fit_cmap)
    color_contour = cmap_intervals(num_sources_to_plot,contour_cmap)
    color_itr = 0

    energy_range = energy_range * u.Unit(energy_unit)

    # if we are not going to sum sources

    if not sum_sources:

        # go thru the mle analysis and plot spectra

        for key in mle_analyses.keys():

            if key in sources_to_use:

                # we won't assume to plot the total until the end

                plot_total = False

                if use_components:

                    # if this source has no components or none that we wish to plot
                    # then we will plot the total spectrum after this

                    if (not mle_analyses[key]['components'].keys()) or ('total' in components_to_use):
                        plot_total = True

                    for component in mle_analyses[key]['components'].keys():

                        # extract the information and plot it

                        best_fit = mle_analyses[key]['components'][component].spectrum
                        positive_error = best_fit + mle_analyses[key]['components'][component].error_region
                        negative_error = best_fit - mle_analyses[key]['components'][component].error_region

                        pos_mask = np.logical_and(best_fit > 0,
                                                  mle_analyses[key]['components'][component].error_region > 0)

                        label = "%s: %s" % (key, component)

                        # this is where we keep track of duplicates

                        if key in duplicate_keys:
                            label = "%s: MLE" % label

                        ax.loglog(energy_range[pos_mask],
                                  best_fit[pos_mask],
                                  color=color_fit[color_itr],
                                  label=label,
                                  **plot_style_kwargs)

                        if show_contours:
                            ax.fill_between(energy_range[pos_mask],
                                            negative_error[pos_mask],
                                            positive_error[pos_mask],
                                            facecolor=color_contour[color_itr],
                                            **contour_style_kwargs

                                            )

                        color_itr += 1

                else:

                    plot_total = True

                if plot_total:

                    # it ends up that we need to plot the total spectrum
                    # which is just a repeat of the process

                    best_fit = mle_analyses[key]['fitted point source'].spectrum
                    positive_error = best_fit + mle_analyses[key]['fitted point source'].error_region
                    negative_error = best_fit - mle_analyses[key]['fitted point source'].error_region

                    pos_mask = np.logical_and(best_fit > 0, mle_analyses[key]['fitted point source'].error_region > 0)

                    label = "%s" % key

                    if key in duplicate_keys:
                        label = "%s: MLE" % label

                    ax.loglog(energy_range[pos_mask],
                              best_fit[pos_mask],
                              color=color_fit[color_itr],
                              label=label,
                              **plot_style_kwargs)

                    if show_contours:
                        ax.fill_between(energy_range[pos_mask],
                                        negative_error[pos_mask],
                                        positive_error[pos_mask],
                                        facecolor=color_contour[color_itr],
                                        **contour_style_kwargs)

                    color_itr += 1

        # we will do the exact same thing for the bayesian analyses

        for key in bayesian_analyses.keys():

            if key in sources_to_use:

                plot_total = False

                if use_components:

                    if (not bayesian_analyses[key]['components'].keys()) or ('total' in components_to_use):
                        plot_total = True

                    for component in bayesian_analyses[key]['components'].keys():

                        best_fit = bayesian_analyses[key]['components'][component].spectrum
                        positive_error = bayesian_analyses[key]['components'][component].error_region[:, 1]
                        negative_error = bayesian_analyses[key]['components'][component].error_region[:, 0]

                        label = "%s: %s" % (key, component)

                        if key in duplicate_keys:
                            label = "%s: Bayesian" % label

                        ax.loglog(energy_range,
                                  best_fit,
                                  color=color_fit[color_itr],
                                  label=label,
                                  **plot_style_kwargs)

                        if show_contours:
                            ax.fill_between(energy_range,
                                            negative_error,
                                            positive_error,
                                            facecolor=color_contour[color_itr],
                                            **contour_style_kwargs

                                            )

                        color_itr += 1




                else:

                    plot_total = True

                if plot_total:

                    best_fit = bayesian_analyses[key]['fitted point source'].spectrum
                    positive_error = bayesian_analyses[key]['fitted point source'].error_region[:, 1]
                    negative_error = bayesian_analyses[key]['fitted point source'].error_region[:, 0]

                    label = "%s" % key

                    if key in duplicate_keys:
                        label = "%s: Bayesian" % label

                    ax.loglog(energy_range,
                              best_fit,
                              color=color_fit[color_itr],
                              label=label,
                              **plot_style_kwargs)

                    if show_contours:
                        ax.fill_between(energy_range,
                                        negative_error,
                                        positive_error,
                                        facecolor=color_contour[color_itr],
                                        **contour_style_kwargs

                                        )

                    color_itr += 1

        if show_legend:
            ax.legend(**legend_kwargs)

        ax.set_xlim(ene_min, ene_max)

    else:


        for key in mle_analyses.keys():

            if key in sources_to_use:

                # we won't assume to plot the total until the end

                plot_total = False

                if use_components:

                    # if this source has no components or none that we wish to plot
                    # then we will plot the total spectrum after this

                    if (not mle_analyses[key]['components'].keys()) or ('total' in components_to_use):
                        plot_total = True

                    for component in mle_analyses[key]['components'].keys():

                        # extract the information and plot it

                        best_fit = mle_analyses[key]['components'][component].spectrum
                        positive_error = best_fit + mle_analyses[key]['components'][component].error_region
                        negative_error = best_fit - mle_analyses[key]['components'][component].error_region

                        pos_mask = np.logical_and(best_fit > 0,
                                                  mle_analyses[key]['components'][component].error_region > 0)

                        label = "%s: %s" % (key, component)

                        # this is where we keep track of duplicates

                        if key in duplicate_keys:
                            label = "%s: MLE" % label

                        ax.loglog(energy_range[pos_mask],
                                  best_fit[pos_mask],
                                  color=color_fit[color_itr],
                                  label=label,
                                  **plot_style_kwargs)

                        if show_contours:
                            ax.fill_between(energy_range[pos_mask],
                                            negative_error[pos_mask],
                                            positive_error[pos_mask],
                                            facecolor=color_contour[color_itr],
                                            **contour_style_kwargs

                                            )

                        color_itr += 1

                else:

                    plot_total = True

                if plot_total:

                    # it ends up that we need to plot the total spectrum
                    # which is just a repeat of the process

                    best_fit = mle_analyses[key]['fitted point source'].spectrum
                    positive_error = best_fit + mle_analyses[key]['fitted point source'].error_region
                    negative_error = best_fit - mle_analyses[key]['fitted point source'].error_region

                    pos_mask = np.logical_and(best_fit > 0, mle_analyses[key]['fitted point source'].error_region > 0)

                    label = "%s" % key

                    if key in duplicate_keys:
                        label = "%s: MLE" % label

                    ax.loglog(energy_range[pos_mask],
                              best_fit[pos_mask],
                              color=color_fit[color_itr],
                              label=label,
                              **plot_style_kwargs)

                    if show_contours:
                        ax.fill_between(energy_range[pos_mask],
                                        negative_error[pos_mask],
                                        positive_error[pos_mask],
                                        facecolor=color_contour[color_itr],
                                        **contour_style_kwargs)

                    color_itr += 1



