import logging

# We import matplotlib first, because we need control on the backend
# Indeed, if no DISPLAY variable is set, matplotlib 2.0 crashes (at the moment, 05/26/2017)
import os
import traceback
import warnings

import pandas as pd

pd.set_option("display.max_columns", None)


# Workaround to avoid a segmentation fault with ROOT and a CFITSIO issue
# LEAVE THESE HERE BEFORE ANY THREEML IMPORT
try:
    import ROOT

    ROOT.__doc__
except ImportError:
    pass
try:
    import pyLikelihood

    pyLikelihood.__doc__
except ImportError:
    pass

from pathlib import Path

# Import everything from astromodels
from astromodels import *

from .config import (
    threeML_config,
    show_configuration,
    get_current_configuration_copy,
)

log = logging.getLogger(__name__)

if threeML_config["logging"]["startup_warnings"]:
    log.info("Starting 3ML!")
    log.warning("WARNINGs here are [red]NOT[/red] errors")
    log.warning("but are inform you about optional packages that can be installed")
    log.warning(
        "[red] to disable these messages, turn off start_warning in your config file[/red]"
    )

if os.environ.get("DISPLAY") is None:
    if threeML_config["logging"]["startup_warnings"]:
        log.warning(
            "no display variable set. using backend for graphics without display (agg)"
        )

    import matplotlib as mpl

    mpl.use("Agg")

# Import version (this has to be placed before the import of serialization
# since __version__ needs to be defined at that stage)
from . import _version

__version__ = _version.get_versions()["version"]

import traceback
import importlib.util

# Finally import the serialization machinery
from .io.serialization import *

# Now import the optimizers first (to avoid conflicting libraries problems)
from .minimizer.minimization import (
    GlobalMinimization,
    LocalMinimization,
    _minimizers,
)
from .plugin_prototype import PluginPrototype

# from .exceptions.custom_exceptions import custom_warnings


# This must be here before the automatic import of subpackages,
# otherwise we will incur in weird issues with other packages
# using similar names (for example, the io package)


# Now look for plugins

# This verifies if a module is importable


def is_module_importable(module_full_path):
    try:
        spec = importlib.util.spec_from_file_location(
            module_full_path.stem, module_full_path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load {module_full_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return True, module
    except Exception:
        return False, traceback.format_exc()


plugins_dir = Path(__file__).parent / "plugins"

found_plugins = plugins_dir.glob("*.py")

found_plugins = [f for f in plugins_dir.glob("*.py") if f.name != "__init__.py"]

_working_plugins = {}
_not_working_plugins = {}

# Loop over each candidates plugins and check if it is importable

for i, module_full_path in enumerate(found_plugins):
    plugin_name = module_full_path.stem

    is_importable, result = is_module_importable(module_full_path)

    if not is_importable:
        if threeML_config.logging.startup_warnings:
            log.warning(
                f"Could not import plugin {module_full_path.name}. Do you have the relative instrument software installed "
                "and configured?"
                # custom_exceptions.CannotImportPlugin,
            )

        _not_working_plugins[plugin_name] = result

        continue

    # First get the instrument name
    module = result
    instrument_name = getattr(module, "__instrument_name", None)
    if not instrument_name:
        continue

    try:
        imported_plugin = importlib.import_module(f"threeML.plugins.{plugin_name}")
        plugin_class = getattr(imported_plugin, plugin_name)
        globals()[plugin_name] = plugin_class
    except ImportError:
        log.warning(
            f"Could not import plugin {plugin_name}."
            "Do you have the relative instrument software installed?"
        )
        continue
    _working_plugins[instrument_name] = plugin_name

# Now some convenience functions


def get_available_plugins():
    """Print a list of available plugins.

    :return:
    """
    print("Available plugins:\n")

    for instrument, class_name in _working_plugins.items():
        print(f"{class_name} for {instrument}")


def _display_plugin_traceback(plugin):
    if threeML_config.logging.startup_warnings:
        log.warning("#############################################################")
        log.warning("\nCouldn't import plugin %s" % plugin)
        log.warning("\nTraceback:\n")
        log.warning(_not_working_plugins[plugin])
        log.warning("#############################################################")


def is_plugin_available(plugin):
    """Test whether the plugin for the provided instrument is available.

    :param plugin: the name of the plugin class
    :return: True or False
    """

    if plugin in _working_plugins.values():
        # FIXME
        if plugin == "FermipyLike":
            try:
                _ = FermipyLike.__new__(FermipyLike, test=True)

            except Exception:
                # Do not register it

                _not_working_plugins[plugin] = traceback.format_exc()

                _display_plugin_traceback(plugin)

                return False

        return True

    else:
        if plugin in _not_working_plugins:
            _display_plugin_traceback(plugin)

            return False

        else:
            log.error(f"Plugin {plugin} is not known")
            raise RuntimeError()


# Import the classic Maximum Likelihood Estimation package

import os

import astropy.units as u

# Import the results loader
from threeML.analysis_results import (
    convert_fits_analysis_result_to_hdf,
    load_analysis_results,
    load_analysis_results_hdf,
)

# Import catalogs
from threeML.catalogs import (
    FermiGBMBurstCatalog,
    FermiGBMTriggerCatalog,
    FermiLATSourceCatalog,
    FermiLLEBurstCatalog,
    FermiPySourceCatalog,
    SwiftGRBCatalog,
)
from threeML.io import (
    activate_logs,
    activate_progress_bars,
    activate_warnings,
    debug_mode,
    loud_mode,
    quiet_mode,
    silence_logs,
    silence_progress_bars,
    silence_warnings,
    toggle_progress_bars,
    update_logging_level,
)
from threeML.io.plotting.light_curve_plots import plot_tte_lightcurve
from threeML.io.plotting.model_plot import (
    plot_point_source_spectra,
    plot_spectra,
)
from threeML.io.plotting.post_process_data_plots import (
    display_photometry_model_magnitudes,
    display_spectrum_model_counts,
)

#
from threeML.io.uncertainty_formatter import interval_to_errors

# import time series builder, soon to replace the Fermi plugins
from threeML.utils.data_builders import *
from threeML.utils.data_download.Fermi_GBM.download_GBM_data import (
    download_GBM_daily_data,
    download_GBM_trigger_data,
)

# Import the LAT data downloader
from threeML.utils.data_download.Fermi_LAT.download_LAT_data import (
    download_LAT_data,
)

# Import LLE downloader
from threeML.utils.data_download.Fermi_LAT.download_LLE_data import (
    download_LLE_trigger_data,
)

# Import the Bayesian analysis
from .bayesian.bayesian_analysis import BayesianAnalysis
from .classicMLE.goodness_of_fit import GoodnessOfFit
from .classicMLE.joint_likelihood import JointLikelihood

# Import the joint likelihood set
from .classicMLE.joint_likelihood_set import (
    JointLikelihoodSet,
    JointLikelihoodSetAnalyzer,
)
from .classicMLE.likelihood_ratio_test import LikelihoodRatioTest
from .data_list import DataList
from .io import get_threeML_style, set_threeML_style
from .io.calculate_flux import calculate_point_source_flux
from .parallel.parallel_client import parallel_computation

# Added by JM. step generator for time-resolved fits
from .utils.step_parameter_generator import step_generator

# Now read the configuration and make it available as threeML_config


# Import the plot_style context manager and the function to create new styles


# Import optical filters
# from threeML.plugins.photometry.filter_factory import threeML_filter_library


# Import GBM  downloader


# Check that the number of threads is set to 1 for all multi-thread libraries
# otherwise numpy operations will be way slower than what they could be, since
# we never perform huge numpy operations, we instead perform many of them. In this
# situation, opening threads introduces overhead with no performance gain. This solution
# allows cores to be used for multi-cpu computation with the parallel client

var_to_check = ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]


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


del os
del Path
del warnings
