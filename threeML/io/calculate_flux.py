__author__ = 'grburgess'

from threeML.io.rich_display import display
from threeML.utils.fitted_objects.fitted_point_sources import FittedPointSourceSpectralHandler

import numpy as np
import pandas as pd



def _setup_analysis_dictionaries(analysis_results, energy_range, energy_unit, flux_unit, use_components, components_to_use,
                                 confidence_level, equal_tailed,differential, sources_to_use):
    """
    helper function to pull out analysis details that are common to flux and plotting functions


    :param analysis_results:
    :param energy_range:
    :param energy_unit:
    :param flux_unit:
    :param use_components:
    :param components_to_use:
    :param confidence_level:
    :param fraction_of_samples:
    :param differential:
    :param sources_to_use:
    :return:
    """

    bayesian_analyses = {}
    mle_analyses = {}

    # first we split up the bayesian and mle analysis

    mle_sources = {}
    bayes_sources = {}

    for analysis in analysis_results:

        for source_name, source in analysis.optimized_model.point_sources.iteritems():


            if source_name in sources_to_use or not sources_to_use:

                if analysis.analysis_type == "MLE":

                    # keep track of duplicate sources

                    mle_sources.setdefault(source_name, []).append(1)

                    if len(mle_sources[source_name]) > 1:
                        name = "%s_%d" % (source_name, len(mle_sources[source_name]))

                    else:

                        name = source_name

                    try:

                        comps = [c.name for c in source.spectrum.main.composite.functions]

                    except:

                        comps = []


                    mle_analyses[name] = {'source': source_name, 'analysis': analysis, 'component_names': comps}


                else:

                    bayes_sources.setdefault(source_name, []).append(1)

                    # keep track of duplicate sources

                    if len(bayes_sources[source_name]) > 1:
                        name = "%s_%d" % (source_name, len(bayes_sources[source_name]))

                    else:

                        name = source_name

                    try:

                        comps = [c.name for c in source.spectrum.main.composite.functions]

                    except:

                        comps = []

                    bayesian_analyses[name] = {'source': source_name, 'analysis': analysis, 'component_names': comps}



    # keep track of the number of sources we will use

    num_sources_to_use = 0

    # go through the MLE analysis and build up some fitted sources

    for key in mle_analyses.keys():

        # if we want to use this source

        if not use_components or ('total' in components_to_use) or (not mle_analyses[key]['component_names']):

            mle_analyses[key]['fitted point source'] = FittedPointSourceSpectralHandler(mle_analyses[key]['analysis'],
                                                                                        mle_analyses[key]['source'],
                                                                                        energy_range,
                                                                                        energy_unit,
                                                                                        flux_unit,
                                                                                        confidence_level,
                                                                                        equal_tailed=equal_tailed,
                                                                                        is_differential_flux=differential)

            num_sources_to_use += 1

        # see if there are any components to use

        if use_components:


            num_components_to_use = 0

            component_dict = {}



            for component in mle_analyses[key]['component_names']:

                # if we want to plot all the components

                if not components_to_use:

                    component_dict[component] = FittedPointSourceSpectralHandler(mle_analyses[key]['analysis'],
                                                                                 mle_analyses[key]['source'],
                                                               energy_range,
                                                               energy_unit,
                                                               flux_unit,
                                                               confidence_level,
                                                               equal_tailed,
                                                               component=component,
                                                               is_differential_flux=differential)

                    num_components_to_use += 1

                else:

                    # otherwise pick off only the ones of interest

                    if component in components_to_use:
                        component_dict[component] = FittedPointSourceSpectralHandler(mle_analyses[key]['analysis'],
                                                                   mle_analyses[key]['source'],
                                                                   energy_range,
                                                                   energy_unit,
                                                                   flux_unit,
                                                                   confidence_level,
                                                                    equal_tailed,
                                                                   component=component,
                                                                   is_differential_flux=differential)

                        num_components_to_use += 1

            # save these to the dict

            mle_analyses[key]['components'] = component_dict

        # keep track of how many components we need to plot

        if use_components:

            num_sources_to_use += num_components_to_use

            if 'total' in components_to_use:
                num_sources_to_use += 1

        else:

            num_sources_to_use += 1

    # repeat for the bayes analyses

    for key in bayesian_analyses.keys():

        # if we have a source to use

        if not use_components or ('total' in components_to_use) or (not bayesian_analyses[key]['component_names']):


            bayesian_analyses[key]['fitted point source'] = FittedPointSourceSpectralHandler(bayesian_analyses[key]['analysis'],
                                                                                             bayesian_analyses[key]['source'],
                                                                                             energy_range,
                                                                                             energy_unit,
                                                                                             flux_unit,
                                                                                             confidence_level,
                                                                                             equal_tailed,
                                                                                             is_differential_flux=differential)

            num_sources_to_use += 1

        # if we want to use components

        if use_components:

            num_components_to_use = 0

            component_dict = {}



            for component in bayesian_analyses[key]['component_names']:

                # extracting all components

                if not components_to_use:
                    component_dict[component] = FittedPointSourceSpectralHandler(bayesian_analyses[key]['analysis'],
                                                                    bayesian_analyses[key]['source'],
                                                                    energy_range,
                                                                    energy_unit,
                                                                    flux_unit,
                                                                    confidence_level,
                                                                    equal_tailed,
                                                                    component=component,
                                                                    is_differential_flux=differential)

                    num_components_to_use += 1

                # or just some of them

                if component in components_to_use:
                    component_dict[component] = FittedPointSourceSpectralHandler(bayesian_analyses[key]['analysis'],
                                                                    bayesian_analyses[key]['source'],
                                                                    energy_range,
                                                                    energy_unit,
                                                                    flux_unit,
                                                                    confidence_level,
                                                                    equal_tailed,
                                                                    component=component,
                                                                    is_differential_flux=differential)

                    num_components_to_use += 1

            bayesian_analyses[key]['components'] = component_dict

        # keep track of everything we added on

        if use_components and num_components_to_use > 0:

            num_sources_to_use += num_components_to_use

            if 'total' in components_to_use:
                num_sources_to_use += 1

        else:

            num_sources_to_use += 1

    # we may have the same source in a bayesian and mle analysis.
    # we want to plot them, but make sure to label them differently.
    # so let's keep track of them

    duplicate_keys = []

    for key in mle_analyses.keys():

        if key in bayesian_analyses.keys():
            duplicate_keys.append(key)



    return mle_analyses, bayesian_analyses, num_sources_to_use, duplicate_keys


def _collect_sums_into_dictionaries(analyses, use_components, components_to_use):
    """

    :param analyses:
    :param use_components:
    :param components_to_use:
    :return:
    """

    total_analysis = []

    component_sum_dict = {}

    num_sources_to_use = 0

    for key in analyses.keys():



        # we won't assume to plot the total until the end

        use_total = False

        if use_components:

            # append all the components we want to sum to their
            # own key

            if (not analyses[key]['components'].keys()) or ('total' in components_to_use):
                use_total = True

            for component in analyses[key]['components'].keys():
                component_sum_dict.setdefault(component, []).append(analyses[key]['components'][component])

        else:

            use_total = True

        if use_total:
            # append the total spectrum

            total_analysis.append(analyses[key]['fitted point source'])

    if use_components:

        for key, values in component_sum_dict.iteritems():
            num_sources_to_use += len(values)

    num_sources_to_use += len(total_analysis)

    return total_analysis, component_sum_dict, num_sources_to_use




def calculate_point_source_flux(ene_min, ene_max, *analyses, **kwargs):
    """

    :param ene_min: lower energy bound for int
    :param ene_max:
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

    :return:
    """

    _defaults = {
        'confidence_level': 0.68,
        'equal_tailed': True,
        'best_fit': 'median',
        'energy_unit': 'keV',
        'flux_unit': 'erg/(s cm2)',
        'ene_min': 10.,
        'ene_max': 1E4,
        'num_ene': 100,
        'use_components': False,
        'components_to_use': [],
        'sources_to_use': [],
        'sum_sources': False,
        'show_contours': True,

    }


    for key, value in kwargs.iteritems():

        if key in _defaults:

            _defaults[key] = value



    # set up the integral limits

    energy_range = np.array([_defaults['ene_min'], _defaults['ene_max']])

    mle_analyses, bayesian_analyses, _, _ = _setup_analysis_dictionaries(analyses,
                                                                                         energy_range,
                                                                                         _defaults['energy_unit'],
                                                                                         _defaults['flux_unit'],
                                                                                         _defaults['use_components'],
                                                                                         _defaults['components_to_use'],
                                                                                         _defaults['confidence_level'],
                                                                                         _defaults['equal_tailed'],
                                                                                         differential=False,
                                                                                         sources_to_use=_defaults['sources_to_use'])

    out = []

    if not _defaults['sum_sources']:

        fluxes = []
        p_errors = []
        n_errors = []
        labels = []

        # go thru the mle analysis and get the fluxes
        for key in mle_analyses.keys():

            # we won't assume to plot the total until the end

            get_total = False

            if _defaults['use_components']:

                # if this source has no components or none that we wish to plot
                # then we will get the total flux after this

                if (not mle_analyses[key]['components'].keys()) or ('total' in _defaults['components_to_use']):
                    get_total = True

                for component in mle_analyses[key]['components'].keys():
                    # extract the information and plot it

                    if _defaults['best_fit'] == 'average':

                        best_fit = mle_analyses[key]['components'][component].average[0, 0]

                    else:

                        best_fit = mle_analyses[key]['components'][component].median[0, 0]



                    positive_error = mle_analyses[key]['components'][component].upper_error[0, 0]

                    negative_error = mle_analyses[key]['components'][component].lower_error[0, 0]


                    label = "%s: %s" % (key, component)

                    fluxes.append(best_fit)
                    p_errors.append(positive_error)
                    n_errors.append(negative_error)
                    labels.append(label)


            else:

                get_total = True

            if get_total:
                # it ends up that we need to plot the total spectrum
                # which is just a repeat of the process

                if _defaults['best_fit'] == 'average':

                    best_fit = mle_analyses[key]['fitted point source'].average[0, 0]

                else:

                    best_fit = mle_analyses[key]['fitted point source'].median[0, 0]




                positive_error = mle_analyses[key]['fitted point source'].upper_error[0, 0]

                negative_error = mle_analyses[key]['fitted point source'].lower_error[0, 0]


                label = "%s: total" % key

                fluxes.append(best_fit)
                p_errors.append(positive_error)
                n_errors.append(negative_error)
                labels.append(label)

        if fluxes:
            # now make a data frame



            mle_df = pd.DataFrame({'flux': fluxes, 'negative error': n_errors, 'positive error': p_errors}, index=labels)
            mle_df = mle_df[['flux', 'negative error', 'positive error']]
            out.append(mle_df)

            display(mle_df)

        else:

            out.append(None)

        # now do the bayesian side

        fluxes = []
        n_errors=[]
        p_errors=[]
        distributions = []
        labels = []

        for key in bayesian_analyses.keys():

            get_total = False

            if _defaults['use_components']:

                if (not bayesian_analyses[key]['components'].keys()) or ('total' in _defaults['components_to_use']):
                    get_total = True

                for component in bayesian_analyses[key]['components'].keys():

                    if _defaults['best_fit'] == 'average':

                        best_fit = bayesian_analyses[key]['components'][component].average[0, 0]

                    else:

                        best_fit = bayesian_analyses[key]['components'][component].median[0, 0]



                    positive_error = bayesian_analyses[key]['components'][component].upper_error[0, 0]
                    negative_error = bayesian_analyses[key]['components'][component].lower_error[0, 0]

                    dist = bayesian_analyses[key]['components'][component].samples[0,0]

                    label = "%s: %s" % (key, component)

                    fluxes.append(best_fit)
                    p_errors.append(positive_error)
                    n_errors.append(negative_error)
                    distributions.append(dist)
                    labels.append(label)

            else:

                get_total = True

            if get_total:

                if _defaults['best_fit'] == 'average':

                    best_fit = bayesian_analyses[key]['fitted point source'].average[0, 0]

                else:

                    best_fit = bayesian_analyses[key]['fitted point source'].median[0, 0]


                positive_error = bayesian_analyses[key]['fitted point source'].upper_error[0, 0]
                negative_error = bayesian_analyses[key]['fitted point source'].lower_error[0, 0]
                dist = bayesian_analyses[key]['fitted point source'].samples[0, 0]
                label = "%s: total" % key

                fluxes.append(best_fit)
                p_errors.append(positive_error)
                n_errors.append(negative_error)
                distributions.append(dist)
                labels.append(label)

        if fluxes:
            # now make a data frame


            bayes_df = pd.DataFrame({'flux': fluxes, 'negative error': n_errors ,'positive error': p_errors, 'flux distribution': distributions},
                                    index=labels)

            bayes_df = bayes_df[['flux', 'negative error', 'positive error', 'flux distribution']]
            out.append(bayes_df)

            display(bayes_df)

        else:

            out.append(None)






    else:

        # instead we now sum the fluxes
        # we keep bayes and mle apart


        total_analysis_mle, component_sum_dict_mle, _ = _collect_sums_into_dictionaries(mle_analyses,
                                                                                        _defaults['use_components'],
                                                                                        _defaults['components_to_use'])

        total_analysis_bayes, component_sum_dict_bayes, _ = _collect_sums_into_dictionaries(
            bayesian_analyses,
            _defaults['use_components'],
            _defaults['components_to_use'])

        fluxes = []
        n_errors = []
        p_errors = []
        labels = []

        if _defaults['use_components'] and component_sum_dict_mle.keys():



            # we have components to calculate

            for component, values in component_sum_dict_mle.iteritems():
                summed_analysis = sum(values)

                if _defaults['best_fit'] == 'average':

                    best_fit = summed_analysis.average[0,0]

                else:

                    best_fit = summed_analysis.median[0,0]

                positive_error = summed_analysis.upper_error[0,0]

                negative_error = summed_analysis.lower_error[0,0]


                label = component

                fluxes.append(best_fit)
                p_errors.append(positive_error)
                n_errors.append(negative_error)
                labels.append(label)

        if total_analysis_mle:

            summed_analysis = sum(total_analysis_mle)

            if _defaults['best_fit'] == 'average':

                best_fit = summed_analysis.average[0, 0]

            else:

                best_fit = summed_analysis.median[0, 0]


            positive_error = summed_analysis.upper_error[0, 0]

            negative_error = summed_analysis.lower_error[0,0]

            label = 'total'

            fluxes.append(best_fit)
            p_errors.append(positive_error)
            n_errors.append(negative_error)
            labels.append(label)

        if fluxes:
            # now make a data frame


            mle_df = pd.DataFrame({'flux': fluxes, 'negative error':n_errors, 'positive error': p_errors}, index=labels)
            mle_df = mle_df[['flux', 'negative error', 'positive error']]
            out.append(mle_df)

            display(mle_df)

        else:

            out.append(None)

        # now do the bayesian side

        fluxes = []
        n_errors = []
        p_errors= []
        distributions = []
        labels = []


        if _defaults['use_components'] and component_sum_dict_bayes.keys():

            # we have components to plot

            for component, values in component_sum_dict_bayes.iteritems():
                summed_analysis = sum(values)

                if _defaults['best_fit'] == 'average':

                    best_fit = summed_analysis.average[0, 0]

                else:

                    best_fit = summed_analysis.median[0,0]


                positive_error = summed_analysis.upper_error[0, 0]
                negative_error = summed_analysis.lower_error[0, 0]

                dist = summed_analysis.samples[0,0]

                label = component

                fluxes.append(best_fit)
                p_errors.append(positive_error)
                n_errors.append(negative_error)
                distributions.append(dist)
                labels.append(label)

        if total_analysis_bayes:

            summed_analysis = sum(total_analysis_bayes)

            if _defaults['best_fit'] == 'average':

                best_fit = summed_analysis.average[0, 0]

            else:

                best_fit = summed_analysis.median[0,0]

            positive_error = summed_analysis.upper_error[0, 0]
            negative_error = summed_analysis.lower_error[0, 0]
            dist = summed_analysis.samples[0, 0]

            label = 'total'

            fluxes.append(best_fit)
            p_errors.append(positive_error)
            n_errors.append(negative_error)
            distributions.append(dist)
            labels.append(label)

        if fluxes:
            # now make a data frame


            bayes_df = pd.DataFrame({'flux': fluxes, 'negative error': n_errors, 'positive error':p_errors,'flux distribution': distributions},
                                    index=labels)

            bayes_df = bayes_df[['flux', 'negative error', 'positive error' ,'flux distribution']]
            out.append(bayes_df)

            display(bayes_df)
        else:

            out.append(None)


    return out
