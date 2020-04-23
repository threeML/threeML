# We import matplotlib first, because we need control on the backend
# Indeed, if no DISPLAY variable is set, matplotlib 2.0 crashes (at the moment, 05/26/2017)
import pandas as pd

pd.set_option("max_columns", None)

import os
import warnings

if os.environ.get("DISPLAY") is None:

    warnings.warn(
        "No DISPLAY variable set. Using backend for graphics without display (Agg)"
    )

    import matplotlib as mpl

    mpl.use("Agg")

# Import version (this has to be placed before the import of serialization
# since __version__ needs to be defined at that stage)
from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

# Finally import the serialization machinery
from .io.serialization import *

from .exceptions.custom_exceptions import custom_warnings

import glob
import imp
import traceback

# Import everything from astromodels
from astromodels import *

# Now import the optimizers first (to avoid conflicting libraries problems)
from .minimizer.minimization import _minimizers, LocalMinimization, GlobalMinimization

# This must be here before the automatic import of subpackages,
# otherwise we will incur in weird issues with other packages
# using similar names (for example, the io package)

from .exceptions import custom_exceptions
from .plugin_prototype import PluginPrototype

try:

    # noinspection PyUnresolvedReferences
    from cthreeML.pyModelInterfaceCache import pyToCppModelInterfaceCache

except ImportError:

    custom_warnings.warn(
        "The cthreeML package is not installed. You will not be able to use plugins which require "
        "the C/C++ interface (currently HAWC)",
        custom_exceptions.CppInterfaceNotAvailable,
    )

# Now look for plugins

# This verifies if a module is importable


def is_module_importable(module_full_path):

    try:

        _ = imp.load_source("__", module_full_path)

    except:

        return False, traceback.format_exc()

    else:

        return True, "%s imported ok" % module_full_path


plugins_dir = os.path.join(os.path.dirname(__file__), "plugins")

found_plugins = glob.glob(os.path.join(plugins_dir, "*.py"))

# Filter out __init__

found_plugins = filter(lambda x: x.find("__init__") < 0, found_plugins)

_working_plugins = {}
_not_working_plugins = {}

# Loop over each candidates plugins and check if it is importable

for i, module_full_path in enumerate(found_plugins):

    plugin_name = os.path.splitext(os.path.basename(module_full_path))[0]

    is_importable, failure_traceback = is_module_importable(module_full_path)

    if not is_importable:

        custom_warnings.warn(
            "Could not import plugin %s. Do you have the relative instrument software installed "
            "and configured?" % os.path.basename(module_full_path),
            custom_exceptions.CannotImportPlugin,
        )

        _not_working_plugins[plugin_name] = failure_traceback

        continue

    else:

        # First get the instrument name
        try:

            exec("from threeML.plugins.%s import __instrument_name" % plugin_name)

        except ImportError:

            # This module does not contain a plugin, continue
            continue

        # Now import the plugin itself

        import_command = "from threeML.plugins.%s import %s" % (
            plugin_name,
            plugin_name,
        )

        try:

            exec(import_command)

        except ImportError:

            pass

        else:

            _working_plugins[__instrument_name] = plugin_name


# Now some convenience functions


def get_available_plugins():
    """
    Print a list of available plugins

    :return:
    """
    print("Available plugins:\n")

    for instrument, class_name in _working_plugins.items():

        print("%s for %s" % (class_name, instrument))


def _display_plugin_traceback(plugin):

    print("#############################################################")
    print("\nCouldn't import plugin %s" % plugin)
    print("\nTraceback:\n")
    print(_not_working_plugins[plugin])
    print("#############################################################")


def is_plugin_available(plugin):
    """
    Test whether the plugin for the provided instrument is available

    :param plugin: the name of the plugin class
    :return: True or False
    """

    if plugin in _working_plugins.values():

        # FIXME
        if plugin == "FermipyLike":

            try:

                _ = FermipyLike.__new__(FermipyLike, test=True)

            except:

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

            raise RuntimeError("Plugin %s is not known" % plugin)


# Import the classic Maximum Likelihood Estimation package

from .classicMLE.joint_likelihood import JointLikelihood

# Import the Bayesian analysis
from .bayesian.bayesian_analysis import BayesianAnalysis

# Import the DataList class

from .data_list import DataList


from threeML.io.plotting.model_plot import plot_spectra, plot_point_source_spectra
from threeML.io.plotting.light_curve_plots import plot_tte_lightcurve
from threeML.io.plotting.post_process_data_plots import (
    display_spectrum_model_counts,
    display_photometry_model_magnitudes,
)

# Import the joint likelihood set
from .classicMLE.joint_likelihood_set import (
    JointLikelihoodSet,
    JointLikelihoodSetAnalyzer,
)
from .classicMLE.likelihood_ratio_test import LikelihoodRatioTest
from .classicMLE.goodness_of_fit import GoodnessOfFit

from .io.calculate_flux import calculate_point_source_flux

# Added by JM. step generator for time-resolved fits
from .utils.step_parameter_generator import step_generator

from .parallel.parallel_client import parallel_computation

#
from threeML.io.uncertainty_formatter import interval_to_errors


# Import optical filters
# from threeML.plugins.photometry.filter_factory import threeML_filter_library

# import time series builder, soon to replace the Fermi plugins
from threeML.utils.data_builders import *

# Import catalogs
from threeML.catalogs import *

# Import GBM  downloader

from threeML.utils.data_download.Fermi_GBM.download_GBM_data import (
    download_GBM_trigger_data,
)

# Import LLE downloader
from threeML.utils.data_download.Fermi_LAT.download_LLE_data import (
    download_LLE_trigger_data,
)

# Now read the configuration and make it available as threeML_config
from .config.config import threeML_config

import astropy.units as u

import os

# Import the LAT data downloader
from threeML.utils.data_download.Fermi_LAT.download_LAT_data import download_LAT_data

# Import the results loader
from threeML.analysis_results import load_analysis_results

# Import the plot_style context manager and the function to create new styles
from .io.plotting.plot_style import (
    plot_style,
    create_new_plotting_style,
    get_available_plotting_styles,
)

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

            custom_warnings.warn(
                "Your env. variable %s is not an integer, which doesn't make sense. Set it to 1 "
                "for optimum performances." % var,
                RuntimeWarning,
            )

    else:

        custom_warnings.warn(
            "Env. variable %s is not set. Please set it to 1 for optimal performances in 3ML"
            % var,
            RuntimeWarning,
        )
