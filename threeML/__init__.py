# We import matplotlib first, because we need control on the backend
# Indeed, if no DISPLAY variable is set, matplotlib 2.0 crashes (at the moment, 05/26/2017)
import os
import warnings
from importlib import import_module
from importlib.util import find_spec

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

_public = {}


# Import everything from astromodels
_astromodels = {
    "Model": ("astromodels.core.model", "Model"),
    "Log_uniform_prior": ("astromodels.functions.priors", "Log_uniform_prior"),
    "Uniform_prior": ("astromodels.functions.priors", "Uniform_prior"),
    "PointSource": ("astromodels.sources", "PointSource"),
    "ExtendedSource": ("astromodels.sources", "ExtendedSource"),
}
_public.update(_astromodels)

_config = {
    k: ("threeML.config", k)
    for k in [
        "get_current_configuration_copy",
        "show_configuration",
        "threeML_config",
    ]
}

_public.update(_config)
_io = {k: ("threeML.io", k) for k in ["setup_logger"]}
_public.update(_io)
_mini = {
    k: ("threeML.minimizer.minimization", k)
    for k in [
        "GlobalMinimization",
        "LocalMinimization",
        "_minimizers",
    ]
}
_public.update(_mini)

_plugins = {"PluginPrototype": ("threeML.plugin_prototype", "PluginPrototype")}
_public.update(_plugins)
_astropy = {"u": ("astropy", "units")}
_public.update(_astropy)
_analysis_results = {
    k: ("threeML.analysis_results", k)
    for k in [
        "convert_fits_analysis_result_to_hdf",
        "load_analysis_results",
        "load_analysis_results_hdf",
    ]
}
_public.update(_analysis_results)

# Import catalogs
_catalogs = {
    k: ("threeML.catalogs", k)
    for k in [
        "FermiGBMBurstCatalog",
        "FermiGBMTriggerCatalog",
        "FermiLATSourceCatalog",
        "FermiLLEBurstCatalog",
        "FermiPySourceCatalog",
        "SwiftGRBCatalog",
    ]
}
_public.update(_catalogs)

_ios = {
    "activate_logs": ("threeML.io", "activate_logs"),
    "activate_progress_bars": ("threeML.io", "activate_progress_bars"),
    "activate_warnings": ("threeML.io", "activate_warnings"),
    "debug_mode": ("threeML.io", "debug_mode"),
    "loud_mode": ("threeML.io", "loud_mode"),
    "quiet_mode": ("threeML.io", "quiet_mode"),
    "silence_logs": ("threeML.io", "silence_logs"),
    "silence_progress_bars": ("threeML.io", "silence_progress_bars"),
    "silence_warnings": ("threeML.io", "silence_warnings"),
    "toggle_progress_bars": ("threeML.io", "toggle_progress_bars"),
    "update_logging_level": ("threeML.io", "update_logging_level"),
    "plot_tte_lightcurve": (
        "threeML.io.plotting.light_curve_plots",
        "plot_tte_lightcurve",
    ),
    "plot_point_source_spectra": (
        "threeML.io.plotting.model_plot",
        "plot_point_source_spectra",
    ),
    "plot_spectra": ("threeML.io.plotting.model_plot", "plot_spectra"),
    "display_photometry_model_magnitudes": (
        "threeML.io.plotting.post_process_data_plots",
        "display_photometry_model_magnitudes",
    ),
    "display_spectrum_model_counts": (
        "threeML.io.plotting.post_process_data_plots",
        "display_spectrum_model_counts",
    ),
    "interval_to_errors": ("threeML.io.uncertainty_formatter", "interval_to_errors"),
    "get_threeML_style": ("threeML.io", "get_threeML_style"),
    "calculate_point_source_flux": (
        "threeML.io.calculate_flux",
        "calculate_point_source_flux",
    ),
}
_public.update(_ios)

_data_builders = {
    "TimeSeriesBuilder": ("threeML.utils.data_builders", "TimeSeriesBuilder"),
    "TransientLATDataBuilder": (
        "threeML.utils.data_builders",
        "TransientLATDataBuilder",
    ),
}
_public.update(_data_builders)
_data_downloaders = {
    "download_LAT_data": ("threeML.utils.data_download.FermiLAT", "download_LAT_data"),
    "download_LLE_trigger_data": (
        "threeML.utils.data_download.FermiLAT",
        "download_LLE_trigger_data",
    ),
    "download_GBM_daily_data": (
        "threeML.utils.data_download.FermiGBM",
        "download_GBM_daily_data",
    ),
    "download_GBM_trigger_data": (
        "threeML.utils.data_download.FermiGBM",
        "download_GBM_trigger_data",
    ),
}
_public.update(_data_downloaders)
_core = {
    "BayesianAnalysis": ("threeML.bayesian.bayesian_analysis", "BayesianAnalysis"),
    "GoodnessOfFit": ("threeML.classicMLE.goodness_of_fit", "GoodnessOfFit"),
    "JointLikelihood": ("threeML.classicMLE.joint_likelihood", "JointLikelihood"),
    "JointLikelihoodSet": (
        "threeML.classicMLE.joint_likelihood_set",
        "JointLikelihoodSet",
    ),
    "JointLikelihoodSetAnalyzer": (
        "threeML.classicMLE.joint_likelihood_set",
        "JointLikelihoodSetAnalyzer",
    ),
    "LikelihoodRatioTest": (
        "threeML.classicMLE.likelihood_ratio_test",
        "LikelihoodRatioTest",
    ),
    "DataList": ("threeML.data_list", "DataList"),
    "parallel_computation": (
        "threeML.parallel.parallel_client",
        "parallel_computation",
    ),
    "step_generator": ("threeML.utils.step_parameter_generator", "step_generator"),
}
_public.update(_core)
# Import GBM  downloader

_deprecated = {}
DEPRECATED_TOPLEVEL = set(_deprecated.keys())

# Export everything (historic behavior), plus convenience names
__all__ = sorted(set(_public.keys()) | {"__version__"})


def __getattr__(name: str):
    # Lazy re-exports

    try:
        mod_name, attr = _public[name]
    except KeyError:
        raise AttributeError(f"module 'threeML' has no attribute {name!r}") from None

    # Emit deprecation for legacy top-level function/prior/template names
    if name in DEPRECATED_TOPLEVEL:
        warnings.warn(
            f"Top-level access 'threeML.{name}' is deprecated; "
            f"use 'from {mod_name} import {name}' instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )
    if name in _astromodels:
        warnings.warn(
            "You are importing x from astromodels as 'from threeML import x'"
            "This is depcrated! - Please use `from astromodels import x'",
            category=DeprecationWarning,
            stacklevel=2,
        )

    try:
        mod = import_module(mod_name)
        val = getattr(mod, attr)
    except ImportError as e:
        # Surface clearer message for optional dependencies or missing submodules
        raise ImportError(
            f"Cannot access 'threeML.{name}' because '{mod_name}' "
            f"could not be imported. This feature may require optional dependencies. "
            f"Original error: {e}"
        ) from e
    except AttributeError as e:
        # Defensive: module imported but symbol missing (internal mismatch)
        raise AttributeError(
            f"'threeML.{name}' could not be resolved from '{mod_name}'."
        ) from e

    globals()[name] = val  # cache
    return val


def __dir__():
    # List everything we export (historic behavior)
    return sorted(__all__)


# Check that the number of threads is set to 1 for all multi-thread libraries
# otherwise numpy operations will be way slower than what they could be, since
# we never perform huge numpy operations, we instead perform many of them. In this
# situation, opening threads introduces overhead with no performance gain. This solution
# allows cores to be used for multi-cpu computation with the parallel client

var_to_check = ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]

from threeML.config import threeML_config
from threeML.io import apply_startup_settings

log = apply_startup_settings(threeML_config)

for var in var_to_check:
    num_threads = os.environ.get(var)

    if num_threads is not None:
        try:
            num_threads = int(num_threads)

        except ValueError:
            if threeML_config.logging.startup_warnings:
                log.warning(
                    "Your env. variable %s is not an integer, which doesn't make sense. Set it to 1 "
                    "for optimum performances." % var,
                    # RuntimeWarning,
                )

    else:
        if threeML_config.logging.startup_warnings:
            log.warning(
                "Env. variable %s is not set. Please set it to 1 for optimal performances in 3ML"
                % var
                #            RuntimeWarning,
            )


def is_plugin_available(name):
    if find_spec(name, "threeML.plugins") is not None:
        return True
    return False
