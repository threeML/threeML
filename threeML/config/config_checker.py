__author__ = 'drJfunk'

# from threeML.config.config import threeML_config
import matplotlib.colors as colors
import matplotlib.pyplot as plt

from threeML.exceptions.custom_exceptions import custom_warnings
from threeML.exceptions import custom_exceptions

# Note! These must bne updated as new configuration variables
# are introduced!


_color_keys = [['ogip', 'counts color'],
               ['ogip', 'background color'],
               ['gbm', 'lightcurve color'],
               ['gbm', 'selection color'],
               ['gbm', 'background color'],
               ['gbm', 'background selection color'],
               ['mle', 'contour background'],
               ['mle', 'contour level 1'],
               ['mle', 'contour level 2'],
               ['mle', 'contour level 3'],
               ['mle', 'profile color'],
               ['mle', 'profile level 1'],
               ['mle', 'profile level 2'],
               ['mle', 'profile level 3']
               ]

_cmap_keys = [['ogip', 'data plot cmap'],
              ['ogip', 'model plot cmap'],
              ['mle', 'contour cmap'],
              ['model plot', 'fit cmap'],
              ['model plot', 'bayes cmap'],
              ['model plot', 'contour cmap']
              ]

# _parallel_keys = [['parallel', 'IPython profile name'],
#                   ['parallel', 'use-parallel']]


_bool_keys = [['parallel', 'use-parallel']]

_string_keys = [['parallel', 'IPython profile name']]

# This stores all the keys that MUST be in the configuration file

_required_keys = {}

# build a dictionary of the keys
for element in [_bool_keys, _string_keys, _color_keys, _cmap_keys]:

    for key_pair in element:

        _required_keys.setdefault(key_pair[0], []).append(key_pair[1])





def is_matplotlib_cmap(cmap):
    try:

        plt.get_cmap(cmap)

        return True


    except:

        return False


def is_matplotlib_color(color):
    # color_converter = colors.ColorConverter()

    try:

        colors.is_color_like(color)

        return True

    except(ValueError):

        return False


def is_bool(var):
    return type(var) == bool


def is_string(var):
    return type(var) == str



def check_configuration(threeML_config, path):
    """
    A routine to make sure that user specified configurations
    are indeed valid.

    :return:
    """

    configuration_is_ok = True

    # First we check if the proper keys exist

    for top_level in _required_keys:

        if top_level in threeML_config:

            for bottom_level in _required_keys[top_level]:

                if bottom_level in threeML_config[top_level]:

                    continue

                else:

                    configuration_is_ok = False

                    custom_warnings.warn(
                        "Configuration is missing %s in %s. Read from %s" % (bottom_level, top_level, path),
                        custom_exceptions.ConfigurationFileCorrupt)
        else:

            configuration_is_ok = False

            custom_warnings.warn("Configuration is missing %s. Read from %s" % (top_level, path),
                                 custom_exceptions.ConfigurationFileCorrupt)

    # If the required keys are missing, then there is no
    # point seeing if they are ok... so we just return the
    # corruption error for now

    if not configuration_is_ok:

        return configuration_is_ok


    # Now we check if the values of the keys are valid


    for key in _color_keys:

        color_to_try = threeML_config[key[0]][key[1]]

        if not is_matplotlib_color(color_to_try):

            configuration_is_ok = False

            custom_warnings.warn(
                    "The key: %s of %s is not a valid color string. Read from %s" % (key[0], key[1], path),
                    custom_exceptions.ConfigurationFileCorrupt)



    for key in _cmap_keys:

        cmap_to_try = threeML_config[key[0]][key[1]]

        if not is_matplotlib_cmap(cmap_to_try):

            configuration_is_ok = False

            custom_warnings.warn(
                    "The key: %s of %s is not a valid cmap string. Read from %s" % (key[0], key[1], path),
                    custom_exceptions.ConfigurationFileCorrupt)

    for key in _bool_keys:

        key_to_try = threeML_config[key[0]][key[1]]

        if not is_bool(key_to_try):

            configuration_is_ok = False

            custom_warnings.warn(
                    "The key: %s of %s is not a bool. Read from %s" % (key[0], key[1], path),
                    custom_exceptions.ConfigurationFileCorrupt)

    for key in _string_keys:

        key_to_try = threeML_config[key[0]][key[1]]

        if not is_string(key_to_try):

            configuration_is_ok = False

            custom_warnings.warn(
                    "The key: %s of %s is not a str. Read from %s" % (key[0], key[1], path),
                    custom_exceptions.ConfigurationFileCorrupt)








    return configuration_is_ok
