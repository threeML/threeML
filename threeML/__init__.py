import glob
import imp
import os
import sys

from version import __version__


# Import everything from astromodels
from astromodels import *

# Now import the optimizers first (to avoid conflicting libraries problems)
from .minimizer.minimization import _minimizers

# This dynamically loads a module and return it in a variable

def is_module_importable(module_full_path):

    try:

        _ = imp.load_source('__', module_full_path)

    except:

        return False

    else:

        return True



# Import in the current namespace everything under the
# models directory

# This must be here before the automatic import of subpackages,
# otherwise we will incur in weird issues with other packages
# using similar names (for example, the io package)

from .exceptions import custom_exceptions
from .exceptions.custom_exceptions import custom_warnings
from .plugin_prototype import PluginPrototype

try:

    # noinspection PyUnresolvedReferences
    from cthreeML.pyModelInterfaceCache import pyToCppModelInterfaceCache

except ImportError:

    custom_warnings.warn("The cthreeML package is not installed. You will not be able to use plugins which require "
                         "the C/C++ interface (currently HAWC)",
                         custom_exceptions.CppInterfaceNotAvailable)

# Import the classic Maximum Likelihood Estimation package

from .classicMLE.joint_likelihood import JointLikelihood

# Import the Bayesian analysis
from .bayesian.bayesian_analysis import BayesianAnalysis

# Import the DataList class

from data_list import DataList


# Find the directory containing 3ML

threeML_dir = os.path.abspath(os.path.dirname(__file__))

# Import all modules here

sys.path.insert(0, threeML_dir)
mods = [os.path.basename(f)[:-3] for f in glob.glob(os.path.join(threeML_dir, "*.py"))]

# Filter out __init__

modsToImport = filter(lambda x: x.find("__init__") < 0, mods)

# Import everything in current directory

for mod in modsToImport:
    exec ("from %s import *" % mod)

# Now look for plugins

plugins_dir = os.path.join(os.path.dirname(__file__), "plugins")

found_plugins = glob.glob(os.path.join(plugins_dir, "*.py"))


# Filter out __init__

found_plugins = filter(lambda x: x.find("__init__") < 0, found_plugins)

_available_plugins = {}

for i, module_full_path in enumerate(found_plugins):

    # Loop over each candidates plugins

    if not is_module_importable(module_full_path):

        custom_warnings.warn("Could not import plugin %s. Do you have the relative instrument software installed "
                             "and configured?" % os.path.basename(module_full_path),
                             custom_exceptions.CannotImportPlugin)
        continue

    else:

        plugin_name = os.path.splitext(os.path.basename(module_full_path))[0]

        # First get the instrument name
        try:

            exec("from threeML.plugins.%s import __instrument_name" % plugin_name)

        except ImportError:

            # This module does not contain a plugin, continue
            continue

        # Now import the plugin itself

        import_command = "from threeML.plugins.%s import %s" % (plugin_name, plugin_name)

        try:

            exec(import_command)

        except ImportError:

            pass

        else:

            _available_plugins[__instrument_name] = plugin_name


def get_available_plugins():
    """
    Print a list of available plugins

    :return:
    """
    print("Available plugins:\n")

    for instrument, class_name in _available_plugins.iteritems():

        print("%s for %s" % (class_name, instrument))


def is_plugin_available(plugin):
    """
    Test whether the plugin for the provided instrument is available

    :param plugin: the name of the plugin class
    :return: True or False
    """

    if plugin in _available_plugins.values():

        # FIXME
        if plugin == "FermipyLike":

            # Test it
            available = FermipyLike.__new__(FermipyLike, test=True)

            if not available:
                # Do not register it

                return False

        return True

    else:

        return False

# Import the joint likelihood set
from .classicMLE.joint_likelihood_set import JointLikelihoodSet, JointLikelihoodSetAnalyzer
from .classicMLE.likelihood_ratio_test import LikelihoodRatioTest
from .classicMLE.goodness_of_fit import GoodnessOfFit

from .io.plotting import *

from .io.calculate_flux import calculate_point_source_flux

from .utils.stats_tools import ModelComparison


# Added by JM. step generator for time-resolved fits
from .utils.step_parameter_generator import step_generator

from .parallel.parallel_client import parallel_computation

# Import catalogs
from threeML.catalogs import *

# Import GBM  downloader
from threeML.plugins.Fermi_GBM.download_GBM_data import download_GBM_trigger_data, cleanup_downloaded_GBM_data

# Import LAT downloader
from threeML.plugins.Fermi_LAT.download_LAT_data import download_LAT_data

# Import LLE downloader
from threeML.plugins.Fermi_LAT.download_LLE_data import download_LLE_trigger_data, cleanup_downloaded_LLE_data

# Now read the configuration and make it available as threeML_config
from .config.config import threeML_config

import astropy.units as u

import os

# Import the LAT data downloader
from threeML.plugins.Fermi_LAT.download_LAT_data import download_LAT_data

# Import the results loader
from threeML.analysis_results import load_analysis_results

# Check that the number of threads is set to 1 for all multi-thread libraries
# otherwise numpy operations will be way slower than what they could be, since
# we never perform huge numpy operations, we instead perform many of them. In this
# situation, opening threads introduces overhead with no performance gain. This solution
# allows cores to be used for multi-cpu computation with the parallel client

var_to_check = ['OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']

for var in var_to_check:

    num_threads = os.environ.get(var)

    if num_threads is not None:

        try:

            num_threads = int(num_threads)

        except ValueError:

            custom_warnings.warn("Your env. variable %s is not an integer, which doesn't make sense. Set it to 1 "
                                 "for optimum performances." % var, RuntimeWarning)

    else:

        custom_warnings.warn("Env. variable %s is not set. Please set it to 1 for optimal performances in 3ML" % var,
                             RuntimeWarning)
