from builtins import range

__author__ = "grburgess"

# from threeML.io.rich_display import display
from threeML.utils.fitted_objects.fitted_point_sources import (
    FittedPointSourceSpectralHandler,
)
from threeML.exceptions.custom_exceptions import custom_warnings

import numpy as np
import pandas as pd
import collections


def _setup_analysis_dictionaries(
    analysis_results,
    energy_range,
    energy_unit,
    flux_unit,
    use_components,
    components_to_use,
    confidence_level,
    equal_tailed,
    differential,
    sources_to_use,
    include_extended,
):
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
    :param include_extended:
    :return:
    """

    bayesian_analyses = collections.OrderedDict()
    mle_analyses = collections.OrderedDict()

    # first we split up the bayesian and mle analysis

    mle_sources = collections.OrderedDict()
    bayes_sources = collections.OrderedDict()

    for analysis in analysis_results:

        items = (
            list(analysis.optimized_model.point_sources.items())
            if not include_extended
            else list(analysis.optimized_model.sources.items())
        )

        for source_name, source in items:

            if source_name in sources_to_use or not sources_to_use:

                if analysis.analysis_type == "MLE":

                    # keep track of duplicate sources

                    mle_sources.setdefault(source_name, []).append(1)

                    if len(mle_sources[source_name]) > 1:
                        name = "%s_%d" % (source_name, len(mle_sources[source_name]))

                    else:

                        name = source_name

                    try:

                        comps = [
                            c.name for c in source.spectrum.main.composite.functions
                        ]

                    except:

                        comps = []

                    # duplicate components
                    comps = [
                        "%s_n%i" % (s, suffix) if num > 1 else s
                        for s, num in list(collections.Counter(comps).items())
                        for suffix in range(1, num + 1)
                    ]

                    mle_analyses[name] = {
                        "source": source_name,
                        "analysis": analysis,
                        "component_names": comps,
                    }

                else:

                    bayes_sources.setdefault(source_name, []).append(1)

                    # keep track of duplicate sources

                    if len(bayes_sources[source_name]) > 1:
                        name = "%s_%d" % (source_name, len(bayes_sources[source_name]))

                    else:

                        name = source_name

                    try:

                        comps = [
                            c.name for c in source.spectrum.main.composite.functions
                        ]

                    except:

                        comps = []

                    # duplicate components
                    comps = [
                        "%s_n%i" % (s, suffix) if num > 1 else s
                        for s, num in list(collections.Counter(comps).items())
                        for suffix in range(1, num + 1)
                    ]

                    bayesian_analyses[name] = {
                        "source": source_name,
                        "analysis": analysis,
                        "component_names": comps,
                    }

    # keep track of the number of sources we will use

    num_sources_to_use = 0

    # go through the MLE analysis and build up some fitted sources

    for key in list(mle_analyses.keys()):

        # if we want to use this source

        if (
            not use_components
            or ("total" in components_to_use)
            or (not mle_analyses[key]["component_names"])
        ):
            mle_analyses[key]["fitted point source"] = FittedPointSourceSpectralHandler(
                mle_analyses[key]["analysis"],
                mle_analyses[key]["source"],
                energy_range,
                energy_unit,
                flux_unit,
                confidence_level,
                equal_tailed=equal_tailed,
                is_differential_flux=differential,
            )

            num_sources_to_use += 1

        # see if there are any components to use

        if use_components:

            num_components_to_use = 0

            component_dict = {}

            for component in mle_analyses[key]["component_names"]:

                # if we want to plot all the components

                if not components_to_use:

                    component_dict[component] = FittedPointSourceSpectralHandler(
                        mle_analyses[key]["analysis"],
                        mle_analyses[key]["source"],
                        energy_range,
                        energy_unit,
                        flux_unit,
                        confidence_level,
                        equal_tailed,
                        component=component,
                        is_differential_flux=differential,
                    )

                    num_components_to_use += 1

                else:

                    # otherwise pick off only the ones of interest

                    if component in components_to_use:
                        component_dict[component] = FittedPointSourceSpectralHandler(
                            mle_analyses[key]["analysis"],
                            mle_analyses[key]["source"],
                            energy_range,
                            energy_unit,
                            flux_unit,
                            confidence_level,
                            equal_tailed,
                            component=component,
                            is_differential_flux=differential,
                        )

                        num_components_to_use += 1

            # save these to the dict

            mle_analyses[key]["components"] = component_dict

        # keep track of how many components we need to plot

        if use_components:

            num_sources_to_use += num_components_to_use

            if "total" in components_to_use:
                num_sources_to_use += 1

        # else:
        #
        #     num_sources_to_use += 1

    # repeat for the bayes analyses

    for key in list(bayesian_analyses.keys()):

        # if we have a source to use

        if (
            not use_components
            or ("total" in components_to_use)
            or (not bayesian_analyses[key]["component_names"])
        ):
            bayesian_analyses[key][
                "fitted point source"
            ] = FittedPointSourceSpectralHandler(
                bayesian_analyses[key]["analysis"],
                bayesian_analyses[key]["source"],
                energy_range,
                energy_unit,
                flux_unit,
                confidence_level,
                equal_tailed,
                is_differential_flux=differential,
            )

            num_sources_to_use += 1

        # if we want to use components

        if use_components:

            num_components_to_use = 0

            component_dict = {}

            for component in bayesian_analyses[key]["component_names"]:

                # extracting all components

                if not components_to_use:
                    component_dict[component] = FittedPointSourceSpectralHandler(
                        bayesian_analyses[key]["analysis"],
                        bayesian_analyses[key]["source"],
                        energy_range,
                        energy_unit,
                        flux_unit,
                        confidence_level,
                        equal_tailed,
                        component=component,
                        is_differential_flux=differential,
                    )

                    num_components_to_use += 1

                # or just some of them

                if component in components_to_use:
                    component_dict[component] = FittedPointSourceSpectralHandler(
                        bayesian_analyses[key]["analysis"],
                        bayesian_analyses[key]["source"],
                        energy_range,
                        energy_unit,
                        flux_unit,
                        confidence_level,
                        equal_tailed,
                        component=component,
                        is_differential_flux=differential,
                    )

                    num_components_to_use += 1

            bayesian_analyses[key]["components"] = component_dict

        # keep track of everything we added on

        if use_components and num_components_to_use > 0:

            num_sources_to_use += num_components_to_use

            if "total" in components_to_use:
                num_sources_to_use += 1
        #
        # else:
        #
        #     num_sources_to_use += 1

    # we may have the same source in a bayesian and mle analysis.
    # we want to plot them, but make sure to label them differently.
    # so let's keep track of them

    duplicate_keys = []

    for key in list(mle_analyses.keys()):

        if key in list(bayesian_analyses.keys()):
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

    component_sum_dict = collections.OrderedDict()

    num_sources_to_use = 0

    for key in list(analyses.keys()):

        # we won't assume to plot the total until the end

        use_total = False

        if use_components:

            # append all the components we want to sum to their
            # own key

            if (not list(analyses[key]["components"].keys())) or (
                "total" in components_to_use
            ):
                use_total = True

            for component in list(analyses[key]["components"].keys()):
                component_sum_dict.setdefault(component, []).append(
                    analyses[key]["components"][component]
                )

        else:

            use_total = True

        if use_total:
            # append the total spectrum

            total_analysis.append(analyses[key]["fitted point source"])

    if use_components:

        for key, values in list(component_sum_dict.items()):
            num_sources_to_use += len(values)

    num_sources_to_use += len(total_analysis)

    return total_analysis, component_sum_dict, num_sources_to_use


def _append_best_fit_and_errors(
    samples, _defaults, label, fluxes, p_errors, n_errors, labels
):
    if _defaults["best_fit"] == "average":

        best_fit = samples.average[0, 0]

    else:

        best_fit = samples.median[0, 0]

    positive_error = samples.upper_error[0, 0]

    negative_error = samples.lower_error[0, 0]

    fluxes.append(best_fit)
    p_errors.append(positive_error)
    n_errors.append(negative_error)
    labels.append(label)


def _compute_output(analyses, _defaults, out):
    fluxes = []
    p_errors = []
    n_errors = []
    labels = []

    # go thru the mle analysis and get the fluxes
    for key in list(analyses.keys()):

        # we won't assume to plot the total until the end

        get_total = False

        if _defaults["use_components"]:

            # if this source has no components or none that we wish to plot
            # then we will get the total flux after this

            if (not list(analyses[key]["components"].keys())) or (
                "total" in _defaults["components_to_use"]
            ):
                get_total = True

            for component in list(analyses[key]["components"].keys()):
                # extract the information and plot it

                samples = analyses[key]["components"][component]

                label = "%s: %s" % (key, component)

                _append_best_fit_and_errors(
                    samples, _defaults, label, fluxes, p_errors, n_errors, labels
                )

        else:

            get_total = True

        if get_total:
            # it ends up that we need to plot the total spectrum
            # which is just a repeat of the process

            samples = analyses[key]["fitted point source"]

            label = "%s: total" % key

            _append_best_fit_and_errors(
                samples, _defaults, label, fluxes, p_errors, n_errors, labels
            )

    if fluxes:
        # now make a data frame

        mle_df = pd.DataFrame(
            {"flux": fluxes, "low bound": n_errors, "hi bound": p_errors}, index=labels
        )
        mle_df = mle_df[["flux", "low bound", "hi bound"]]
        mle_df = mle_df[["flux", "low bound", "hi bound"]]
        out.append(mle_df)

        # display(mle_df)

    else:

        out.append(None)


def _compute_output_with_components(_defaults, component_sum_dict, total_analysis, out):

    fluxes = []
    n_errors = []
    p_errors = []
    labels = []

    if _defaults["use_components"] and list(component_sum_dict.keys()):

        # we have components to calculate

        for component, values in list(component_sum_dict.items()):

            summed_analysis = sum(values)

            if _defaults["best_fit"] == "average":

                best_fit = summed_analysis.average[0, 0]

            else:

                best_fit = summed_analysis.median[0, 0]

            positive_error = summed_analysis.upper_error[0, 0]

            negative_error = summed_analysis.lower_error[0, 0]

            label = component

            fluxes.append(best_fit)
            p_errors.append(positive_error)
            n_errors.append(negative_error)
            labels.append(label)

    if total_analysis:

        summed_analysis = sum(total_analysis)

        if _defaults["best_fit"] == "average":

            best_fit = summed_analysis.average[0, 0]

        else:

            best_fit = summed_analysis.median[0, 0]

        positive_error = summed_analysis.upper_error[0, 0]

        negative_error = summed_analysis.lower_error[0, 0]

        label = "total"

        fluxes.append(best_fit)
        p_errors.append(positive_error)
        n_errors.append(negative_error)
        labels.append(label)

    if fluxes:
        # now make a data frame

        df = pd.DataFrame(
            {"flux": fluxes, "low bound": n_errors, "hi bound": p_errors}, index=labels
        )
        df = df[["flux", "low bound", "hi bound"]]
        out.append(df)

        # display(df)

    else:

        out.append(None)


def calculate_point_source_flux(*args, **kwargs):

    custom_warnings.warn(
        "The use of calculate_point_source_flux is deprecated. Please use the .get_point_source_flux()"
        " method of the JointLikelihood.results or the BayesianAnalysis.results member. For example:"
        " jl.results.get_point_source_flux()."
    )

    return _calculate_point_source_flux(*args, **kwargs)


def _calculate_point_source_flux(ene_min, ene_max, *analyses, **kwargs):
    """

    :param ene_min: lower energy bound for the flux
    :param ene_max: upper energy bound for the flux
    :param analyses: fitted JointLikelihood or BayesianAnalysis objects
    :param sources_to_use: (optional) list of PointSource string names to plot from the analysis
    :param energy_unit: (optional) astropy energy unit in string form (can also be frequency)
    :param flux_unit: (optional) astropy flux unit in string form
    :param ene_min: (optional) minimum energy to plot
    :param ene_max: (optional) maximum energy to plot
    :param use_components: (optional) True or False to plot the spectral components
    :param components_to_use: (optional) list of string names of the components to plot: including 'total'
    will also plot the total spectrum
    :param include_extended: (optional) if True, plot extended source spectra (spatially integrated) as well.
    
    :return: mle_dataframe, bayes_dataframe
    """

    _defaults = {
        "confidence_level": 0.68,
        "equal_tailed": True,
        "best_fit": "median",
        "energy_unit": "keV",
        "flux_unit": "erg/(s cm2)",
        "ene_min": ene_min,
        "ene_max": ene_max,
        "use_components": False,
        "components_to_use": [],
        "sources_to_use": [],
        "sum_sources": False,
        "include_extended": False,
    }

    for key, value in list(kwargs.items()):

        if key in _defaults:
            _defaults[key] = value

    # set up the integral limits

    energy_range = np.array([_defaults["ene_min"], _defaults["ene_max"]])

    mle_analyses, bayesian_analyses, _, _ = _setup_analysis_dictionaries(
        analyses,
        energy_range,
        _defaults["energy_unit"],
        _defaults["flux_unit"],
        _defaults["use_components"],
        _defaults["components_to_use"],
        _defaults["confidence_level"],
        _defaults["equal_tailed"],
        differential=False,
        sources_to_use=_defaults["sources_to_use"],
        include_extended=_defaults["include_extended"],
    )

    out = []

    if not _defaults["sum_sources"]:

        # Process the MLE analyses

        _compute_output(mle_analyses, _defaults, out)

        # now do the bayesian side

        _compute_output(bayesian_analyses, _defaults, out)

    else:

        # instead we now sum the fluxes
        # we keep bayes and mle apart

        total_analysis_mle, component_sum_dict_mle, _ = _collect_sums_into_dictionaries(
            mle_analyses, _defaults["use_components"], _defaults["components_to_use"]
        )

        _compute_output_with_components(
            _defaults, component_sum_dict_mle, total_analysis_mle, out
        )

        # now do the bayesian side

        (
            total_analysis_bayes,
            component_sum_dict_bayes,
            _,
        ) = _collect_sums_into_dictionaries(
            bayesian_analyses,
            _defaults["use_components"],
            _defaults["components_to_use"],
        )

        _compute_output_with_components(
            _defaults, component_sum_dict_bayes, total_analysis_bayes, out
        )

    return out
