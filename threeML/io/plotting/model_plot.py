__author__ = 'grburgess'

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import quantity_support

from threeML.config.config import threeML_config
from threeML.io.calculate_flux import _setup_analysis_dictionaries, _collect_sums_into_dictionaries
from threeML.io.plotting.cmap_cycle import cmap_intervals




def plot_point_source_spectra(*analysis_results, **kwargs):
    """

    plotting routine for fitted point source spectra


    :param analysis_results: fitted JointLikelihood or BayesianAnalysis objects
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

    _defaults = {'fit_cmap': threeML_config['model plot']['point source plot']['fit cmap'],
                 'contour_cmap': threeML_config['model plot']['point source plot']['contour cmap'],
                 'confidence_level': 0.68,
                 'equal_tailed': True,
                 'best_fit': 'median',
                 'energy_unit': 'keV',
                 'flux_unit': '1/(keV s cm2)',
                 'ene_min': 10.,
                 'ene_max': 1E4,
                 'num_ene': 100,
                 'use_components': False,
                 'components_to_use': [],
                 'sources_to_use': [],
                 'sum_sources': False,
                 'show_contours': True,
                 'plot_style_kwargs': threeML_config['model plot']['point source plot']['plot style'],
                 'contour_style_kwargs': threeML_config['model plot']['point source plot']['contour style'],
                 'show_legend': True,
                 'legend_kwargs': threeML_config['model plot']['point source plot']['legend style']

                 }



    for key, value in kwargs.iteritems():

        if key in _defaults:

            _defaults[key] = value


    energy_range = np.logspace(np.log10(_defaults['ene_min']), np.log10(_defaults['ene_max']), _defaults['num_ene'])

    mle_analyses, bayesian_analyses, num_sources_to_plot, duplicate_keys = _setup_analysis_dictionaries(
        analysis_results,
        energy_range,
        _defaults['energy_unit'],
        _defaults['flux_unit'],
        _defaults['use_components'],
        _defaults['components_to_use'],
        _defaults['confidence_level'],
        _defaults['equal_tailed'],
        differential=True,
        sources_to_use=_defaults['sources_to_use'])

    # we are now ready to plot.
    # all calculations have been made.

    fig, ax = plt.subplots()

    energy_range = energy_range * u.Unit(_defaults['energy_unit'])

    # if we are not going to sum sources

    if not _defaults['sum_sources']:

        color_fit = cmap_intervals(num_sources_to_plot, _defaults['fit_cmap'])
        color_contour = cmap_intervals(num_sources_to_plot, _defaults['contour_cmap'])
        color_itr = 0

        # go thru the mle analysis and plot spectra

        for key in mle_analyses.keys():



            # we won't assume to plot the total until the end

            plot_total = False

            if _defaults['use_components']:

                # if this source has no components or none that we wish to plot
                # then we will plot the total spectrum after this

                if (not mle_analyses[key]['components'].keys()) or ('total' in _defaults['components_to_use']):
                    plot_total = True

                for component in mle_analyses[key]['components'].keys():

                    # extract the information and plot it

                    if _defaults['best_fit'] == 'average':

                        best_fit = mle_analyses[key]['components'][component].average

                    else:

                        best_fit = mle_analyses[key]['components'][component].median

                    positive_error = mle_analyses[key]['components'][component].upper_error
                    negative_error = mle_analyses[key]['components'][component].lower_error

                    neg_mask = negative_error <= 0

                    # replace with small number

                    negative_error[neg_mask] = min(best_fit) * 0.9

                    label = "%s: %s" % (key, component)

                    # this is where we keep track of duplicates

                    if key in duplicate_keys:
                        label = "%s: MLE" % label

                    ax.loglog(energy_range,
                              best_fit,
                              color=color_fit[color_itr],
                              label=label,
                              **_defaults['plot_style_kwargs'])

                    if _defaults['show_contours']:
                        ax.fill_between(energy_range,
                                        negative_error,
                                        positive_error,
                                        facecolor=color_contour[color_itr],
                                        **_defaults['contour_style_kwargs']

                                        )

                    color_itr += 1

            else:

                plot_total = True

            if plot_total:

                # it ends up that we need to plot the total spectrum
                # which is just a repeat of the process

                if _defaults['best_fit'] == 'average':

                    best_fit = mle_analyses[key]['fitted point source'].average

                else:

                    best_fit = mle_analyses[key]['fitted point source'].median

                positive_error = mle_analyses[key]['fitted point source'].upper_error
                negative_error = mle_analyses[key]['fitted point source'].lower_error

                neg_mask = negative_error <= 0

                # replace with small number

                negative_error[neg_mask] = min(best_fit) * 0.9

                label = "%s" % key

                if key in duplicate_keys:
                    label = "%s: MLE" % label

                ax.loglog(energy_range,
                          best_fit,
                          color=color_fit[color_itr],
                          label=label,
                          **_defaults['plot_style_kwargs'])

                if _defaults['show_contours']:
                    ax.fill_between(energy_range,
                                    negative_error,
                                    positive_error,
                                    facecolor=color_contour[color_itr],
                                    **_defaults['contour_style_kwargs'])

                color_itr += 1

        # we will do the exact same thing for the bayesian analyses

        for key in bayesian_analyses.keys():


            plot_total = False

            if _defaults['use_components']:

                if (not bayesian_analyses[key]['components'].keys()) or ('total' in _defaults['components_to_use']):
                    plot_total = True

                for component in bayesian_analyses[key]['components'].keys():

                    if _defaults['best_fit'] == 'average':

                        best_fit = bayesian_analyses[key]['components'][component].average

                    else:

                        best_fit = bayesian_analyses[key]['components'][component].median


                    positive_error = bayesian_analyses[key]['components'][component].upper_error
                    negative_error = bayesian_analyses[key]['components'][component].lower_error

                    label = "%s: %s" % (key, component)

                    if key in duplicate_keys:
                        label = "%s: Bayesian" % label

                    ax.loglog(energy_range,
                              best_fit,
                              color=color_fit[color_itr],
                              label=label,
                              **_defaults['plot_style_kwargs'])

                    if _defaults['show_contours']:
                        ax.fill_between(energy_range,
                                        negative_error,
                                        positive_error,
                                        facecolor=color_contour[color_itr],
                                        **_defaults['contour_style_kwargs']

                                        )

                    color_itr += 1




            else:

                plot_total = True

            if plot_total:

                if _defaults['best_fit'] == 'average':

                    best_fit = bayesian_analyses[key]['fitted point source'].average

                else:

                    best_fit = bayesian_analyses[key]['fitted point source'].median


                positive_error = bayesian_analyses[key]['fitted point source'].upper_error
                negative_error = bayesian_analyses[key]['fitted point source'].lower_error

                label = "%s" % key

                if key in duplicate_keys:
                    label = "%s: Bayesian" % label

                ax.loglog(energy_range,
                          best_fit,
                          color=color_fit[color_itr],
                          label=label,
                          **_defaults['plot_style_kwargs'])

                if _defaults['show_contours']:
                    ax.fill_between(energy_range,
                                    negative_error,
                                    positive_error,
                                    facecolor=color_contour[color_itr],
                                    **_defaults['contour_style_kwargs']

                                    )

                color_itr += 1

        if _defaults['show_legend']:
            ax.legend(**_defaults['legend_kwargs'])

        ax.set_xlim(_defaults['ene_min'], _defaults['ene_max'])

    else:
        # now we sum sources instead


        # we keep MLE and Bayes apart because it makes no
        # sense to sum them together

        total_analysis_mle, component_sum_dict_mle, num_sources_to_plot = _collect_sums_into_dictionaries(mle_analyses,
                                                                                                          _defaults['use_components'],
                                                                                                          _defaults['components_to_use'])




        total_analysis_bayes, component_sum_dict_bayes, num_sources_to_plot_bayes = _collect_sums_into_dictionaries(
            bayesian_analyses,
            _defaults['use_components'],
            _defaults['components_to_use'])


        num_sources_to_plot += num_sources_to_plot_bayes

        color_fit = cmap_intervals(num_sources_to_plot, _defaults['fit_cmap'])
        color_contour = cmap_intervals(num_sources_to_plot, _defaults['contour_cmap'])
        color_itr = 0

        if _defaults['use_components'] and component_sum_dict_mle.keys():

            # we have components to plot

            for component, values in component_sum_dict_mle.iteritems():

                summed_analysis = sum(values)

                if _defaults['best_fit'] == 'average':

                    best_fit = summed_analysis.average

                else:

                    best_fit = summed_analysis.median



                positive_error = summed_analysis.upper_error

                negative_error = summed_analysis.lower_error

                neg_mask = negative_error <= 0

                # replace with small number

                negative_error[neg_mask] = min(best_fit) * 0.9

                ax.loglog(energy_range,
                          best_fit,
                          color=color_fit[color_itr],
                          label="%s: MLE" % component,
                          **_defaults['plot_style_kwargs'])

                if _defaults['show_contours']:
                    ax.fill_between(energy_range,
                                    negative_error,
                                    positive_error,
                                    facecolor=color_contour[color_itr],
                                    **_defaults['contour_style_kwargs'])

                color_itr += 1

        if total_analysis_mle:

            # we will sum and plot the total
            # analysis


            summed_analysis = sum(total_analysis_mle)

            if _defaults['best_fit'] == 'average':

                best_fit = summed_analysis.average

            else:

                best_fit = summed_analysis.median

            positive_error = best_fit + summed_analysis.upper_error

            negative_error = best_fit - summed_analysis.lower_error

            neg_mask = negative_error <= 0

            # replace with small number

            negative_error[neg_mask] = min(best_fit) * 0.9

            ax.loglog(energy_range,
                      best_fit,
                      color=color_fit[color_itr],
                      label="total: MLE",
                      **_defaults['plot_style_kwargs'])

            if _defaults['show_contours']:
                ax.fill_between(energy_range,
                                negative_error,
                                positive_error,
                                facecolor=color_contour[color_itr],
                                **_defaults['contour_style_kwargs'])

            color_itr += 1

        if _defaults['use_components'] and component_sum_dict_bayes.keys():

            # we have components to plot

            for component, values in component_sum_dict_bayes.iteritems():

                summed_analysis = sum(values)

                if _defaults['best_fit'] == 'average':

                    best_fit = summed_analysis.average

                else:

                    best_fit = summed_analysis.median

                positive_error = summed_analysis.upper_error

                negative_error = summed_analysis.lower_error

                ax.loglog(energy_range,
                          best_fit,
                          color=color_fit,
                          label="%s: Bayesian" % component,
                          **_defaults['plot_style_kwargs'])

                if _defaults['show_contours']:
                    ax.fill_between(energy_range,
                                    negative_error,
                                    positive_error,
                                    facecolor=color_contour[color_itr],
                                    **_defaults['contour_style_kwargs'])

                color_itr += 1

        if total_analysis_bayes:

            # we will sum and plot the total
            # analysis


            summed_analysis = sum(total_analysis_bayes)

            if _defaults['best_fit'] == 'average':

                best_fit = summed_analysis.average

            else:

                best_fit = summed_analysis.median

            positive_error = summed_analysis.upper_error

            negative_error = summed_analysis.lower_error

            ax.loglog(energy_range,
                      best_fit,
                      color=color_fit[color_itr],
                      label="total: Bayesian",
                      **_defaults['plot_style_kwargs'])

            if _defaults['show_contours']:
                ax.fill_between(energy_range,
                                negative_error,
                                positive_error,
                                facecolor=color_contour[color_itr],
                                **_defaults['contour_style_kwargs'])

        if _defaults['show_legend']:
            ax.legend(**_defaults['legend_kwargs'])

        ax.set_xlim(_defaults['ene_min'], _defaults['ene_max'])
        color_itr += 1


