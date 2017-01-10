__author__ = 'grburgess'

from threeML.io.rich_display import display
from threeML.utils.fitted_objects.fitted_point_sources import MLEPointSource, BayesianPointSource

import numpy as np
import pandas as pd


#def _check_double_source_names(sources):




def _setup_analysis_dictionaries(analyses,energy_range,energy_unit, flux_unit, use_components,components_to_use, sigma, fraction_of_samples,differential,**kwargs):
    """
    helper function to pull out analysis details that are common to flux and plotting functions


    :param analyses:
    :param energy_range:
    :param energy_unit:
    :param flux_unit:
    :param use_components:
    :param components_to_use:
    :param sigma:
    :param fraction_of_samples:
    :param differential:
    :param kwargs:
    :return:
    """

    bayesian_analyses = {}
    mle_analyses = {}

    # first we split up the bayesian and mle analysis

    mle_sources = {}
    bayes_sources = {}


    for analysis in analyses:

        for name, source in analysis.likelihood_model.point_sources.iteritems():

            if analysis.analysis_type == "mle":

                # keep track of duplicate sources

                mle_sources.setdefault(name, []).append(1)

                if len(mle_sources[name]) > 1:

                    name = "%s_%d" % (name, len(mle_sources[name]))

                mle_analyses[name] = {'source': source, 'analysis': analysis}


            else:

                bayes_sources.setdefault(name, []).append(1)

                # keep track of duplicate sources

                if len(bayes_sources[name]) > 1:

                    name = "%s_%d" % (name, len(bayes_sources[name]))

                bayesian_analyses[name] = {'source': source, 'analysis': analysis}


    sources_to_use = set(bayesian_analyses.keys() + mle_analyses.keys())

    # decided if we only want to use a few sources

    if "sources_to_use" in kwargs:

        sources_to_plot_tmp = kwargs.pop('sources_to_use')

        tmp = []

        for source_key in sources_to_plot_tmp:

            if source_key in sources_to_use:

                tmp.append(source_key)

            else:
                pass
                # warn

        sources_to_use = tmp

    # keep track of the number of sources we will use

    num_sources_to_use = 0

    # go through the MLE analysis and build up some fitted sources

    for key in mle_analyses.keys():

        # if we want to use this source

        if key in sources_to_use:

            mle_analyses[key]['fitted point source'] = MLEPointSource(mle_analyses[key]['analysis'],
                                                                      mle_analyses[key]['source'],
                                                                      energy_range,
                                                                      energy_unit,
                                                                      flux_unit,
                                                                      sigma,
                                                                      is_differential_flux=differential)

            # see if there are any components to use

            if use_components:

                num_components_to_use = 0

                component_dict = {}

                if mle_analyses[key]['fitted point source'].components is not None:

                    for component in mle_analyses[key]['fitted point source'].components:

                        # if we want to plot all the components

                        if not components_to_use:

                            component_dict[component] = MLEPointSource(mle_analyses[key]['analysis'],
                                                                       mle_analyses[key]['source'],
                                                                       energy_range,
                                                                       energy_unit,
                                                                       flux_unit,
                                                                       sigma,
                                                                       component=component,
                                                                       is_differential_flux=differential)

                            num_components_to_use += 1

                        else:

                            # otherwise pick off only the ones of interest

                            if component in components_to_use:
                                component_dict[component] = MLEPointSource(mle_analyses[key]['analysis'],
                                                                           mle_analyses[key]['source'],
                                                                           energy_range,
                                                                           energy_unit,
                                                                           flux_unit,
                                                                           sigma,
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

        if key in sources_to_use:

            bayesian_analyses[key]['fitted point source'] = BayesianPointSource(bayesian_analyses[key]['analysis'],
                                                                                bayesian_analyses[key]['source'],
                                                                                energy_range,
                                                                                energy_unit,
                                                                                flux_unit,
                                                                                sigma,
                                                                                fraction_of_samples=fraction_of_samples,
                                                                                is_differential_flux=differential)

            # if we want to use components

            if use_components:

                num_components_to_use = 0

                component_dict = {}

                if bayesian_analyses[key]['fitted point source'].components is not None:

                    for component in bayesian_analyses[key]['fitted point source'].components:

                        # extracting all components

                        if not components_to_use:
                            component_dict[component] = BayesianPointSource(bayesian_analyses[key]['analysis'],
                                                                            bayesian_analyses[key]['source'],
                                                                            energy_range,
                                                                            energy_unit,
                                                                            flux_unit,
                                                                            sigma,
                                                                            component=component,
                                                                            fraction_of_samples=fraction_of_samples,
                                                                            is_differential_flux=differential)

                            num_components_to_use += 1

                        # or just some of them

                        if component in components_to_use:
                            component_dict[component] = BayesianPointSource(bayesian_analyses[key]['analysis'],
                                                                            bayesian_analyses[key]['source'],
                                                                            energy_range,
                                                                            energy_unit,
                                                                            flux_unit,
                                                                            sigma,
                                                                            component=component,
                                                                            fraction_of_samples=fraction_of_samples,
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

    return mle_analyses, bayesian_analyses, num_sources_to_use, sources_to_use, duplicate_keys


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

    if 'energy_unit' in kwargs:

        energy_unit = kwargs.pop('energy_unit')

    else:

        energy_unit = 'keV'

    if 'flux_unit' in kwargs:

        flux_unit = kwargs.pop('flux_unit')

    else:

        flux_unit = 'erg/(cm2 s)'

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

    if 'sigma' in kwargs:

        sigma = kwargs.pop('sigma')

    else:

        sigma = 1

    if 'fraction_of_samples' in kwargs:

        fraction_of_samples = kwargs.pop('fraction_of_samples')

    else:

        fraction_of_samples = 0.1

    # set up the integral limits

    energy_range = np.array([ene_min, ene_max])

    mle_analyses, bayesian_analyses, _, sources_to_use, _ = _setup_analysis_dictionaries(analyses,
                                                                         energy_range,
                                                                         energy_unit,
                                                                         flux_unit,
                                                                         use_components,
                                                                         components_to_use,
                                                                         sigma,
                                                                         fraction_of_samples,
                                                                         differential=False,
                                                                         **kwargs)



    out = []

    if not sum_sources:

        fluxes = []
        errors = []
        labels = []

        # go thru the mle analysis and get the fluxes
        for key in mle_analyses.keys():

            # we won't assume to plot the total until the end

            get_total = False

            if use_components:

                # if this source has no components or none that we wish to plot
                # then we will get the total flux after this

                if (not mle_analyses[key]['components'].keys()) or ('total' in components_to_use):
                    get_total = True

                for component in mle_analyses[key]['components'].keys():
                    # extract the information and plot it

                    best_fit = mle_analyses[key]['components'][component].spectrum[0, 0]
                    error = mle_analyses[key]['components'][component].error_region[0, 0]
                    label = "%s: %s" % (key, component)

                    fluxes.append(best_fit)
                    errors.append(error)
                    labels.append(label)


            else:

                get_total = True

            if get_total:
                # it ends up that we need to plot the total spectrum
                # which is just a repeat of the process

                best_fit = mle_analyses[key]['fitted point source'].spectrum[0, 0]
                error = mle_analyses[key]['fitted point source'].error_region[0, 0]
                label = "%s: total" % key

                fluxes.append(best_fit)
                errors.append(error)
                labels.append(label)

        if fluxes:
            # now make a data frame

            mle_df = pd.DataFrame({'flux': fluxes, 'error': errors}, index=labels)
            mle_df = mle_df[['flux','error']]
            out.append(mle_df)

            display(mle_df)

        # now do the bayesian side

        fluxes = []
        errors = []
        distributions = []
        labels = []

        for key in bayesian_analyses.keys():

            if key in sources_to_use:

                get_total = False

                if use_components:

                    if (not bayesian_analyses[key]['components'].keys()) or ('total' in components_to_use):
                        get_total = True

                    for component in bayesian_analyses[key]['components'].keys():
                        best_fit = bayesian_analyses[key]['components'][component].spectrum[0, 0]
                        positive_error = bayesian_analyses[key]['components'][component].error_region[0, 1]
                        negative_error = bayesian_analyses[key]['components'][component].error_region[0, 0]
                        dist = bayesian_analyses[key]['components'][component].raw_chains[0]
                        label = "%s: %s" % (key, component)

                        fluxes.append(best_fit)
                        errors.append([negative_error, positive_error])
                        distributions.append(dist)
                        labels.append(label)

                else:

                    get_total = True

                if get_total:
                    best_fit = bayesian_analyses[key]['fitted point source'].spectrum[0, 0]
                    positive_error = bayesian_analyses[key]['fitted point source'].error_region[0, 1]
                    negative_error = bayesian_analyses[key]['fitted point source'].error_region[0, 0]
                    dist = bayesian_analyses[key]['fitted point source'].raw_chains[0]
                    label = "%s: total" % key

                    fluxes.append(best_fit)
                    errors.append([negative_error, positive_error])
                    distributions.append(dist)
                    labels.append(label)

        if fluxes:
            # now make a data frame

            bayes_df = pd.DataFrame({'flux': fluxes, 'credible region': errors, 'flux distribution': distributions},
                                    index=labels)

            bayes_df = bayes_df[['flux', 'credible region', 'flux distribution']]
            out.append(bayes_df)

            display(bayes_df)






    else:

        raise NotImplementedError('Summing fluxes does is not possible yet.')

    return out
