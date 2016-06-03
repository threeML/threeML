import os
import sys
import glob
import inspect
import imp

# Import the version
from version import __version__

# This dynamically loads a module and return it in a variable

def import_module(module_name):

    # Fast path: see if the module has already been imported.

    try:

        return sys.modules[module_name]

    except KeyError:

        pass

    # If any of the following calls raises an exception,
    # there's a problem we can't handle -- let the caller handle it.

    fp, pathname, description = imp.find_module(module_name)

    try:

        return imp.load_module(module_name, fp, pathname, description)

    except:

        raise

    finally:

        # Since we may exit via an exception, close fp explicitly.

        if fp:

            fp.close()

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
    from cthreeML.pyModelInterface import pyToCppModelInterface

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
sys.path.insert(1, plugins_dir)
found_plugins = glob.glob(os.path.join(plugins_dir, "*.py"))

# Filter out __init__

found_plugins = filter(lambda x: x.find("__init__") < 0, found_plugins)

_available_plugins = {}

for i, plug in enumerate(found_plugins):

    # Loop over each candidates plugins

    try:

        thisPlugin = import_module(os.path.basename(".".join(plug.split(".")[:-1])))

    except ImportError:

        custom_warnings.warn("Could not import plugin %s. Do you have the relative instrument software installed "
                             "and configured?" % plug, custom_exceptions.CannotImportPlugin)
        continue

    # Get the classes within this module

    classes = inspect.getmembers(thisPlugin, lambda x: inspect.isclass(x) and inspect.getmodule(x) == thisPlugin)

    for name, cls in classes:

        if not issubclass(cls, PluginPrototype):

            # This is not a plugin

            pass

        else:

            _available_plugins[thisPlugin.__instrument_name] = cls.__name__

            # import it again in the uppermost namespace

            exec ("from %s import %s" % (os.path.basename(thisPlugin.__name__), cls.__name__))


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

        return True

    else:

        return False

from .parallel.parallel_client import parallel_computation

# Now read the configuration and make it available as threeML_config
from .config.config import threeML_config

# Finally import everything from astromodels
from astromodels import *

import astropy.units as u