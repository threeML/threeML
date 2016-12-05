__author__ = 'drJfunk'

# from threeML.config.config import threeML_config
import matplotlib.colors as colors
import matplotlib.pyplot as plt

from threeML.exceptions.custom_exceptions import custom_warnings
from threeML.exceptions import custom_exceptions


_color_keys = [['ogip', 'counts color'],
               ['ogip', 'background color'],
               ['gbm', 'lightcurve color'],
               ['gbm', 'selection color'],
               ['gbm', 'background color'],
               ['gbm', 'background selection color']
               ]

_cmap_keys = [['ogip', 'data plot cmap'],
              ['ogip', 'model plot cmap'],
              ]

_parallel_keys = [['parallel', 'IPython profile name'],
                  ['parallel', 'use-parallel']]

# This stores all the keys that MUST be in the configuration file

_required_keys = {}

# build a dictionary of the keys
for element in [_parallel_keys, _color_keys, _cmap_keys]:

    for key_pair in element:

        _required_keys.setdefault(key_pair[0], []).append(key_pair[1])





def is_matplotlib_cmap(cmap):
    try:

        plt.get_cmap(cmap)

        return True


    except:

        return False


def is_matplotlib_color(color):
    color_converter = colors.ColorConverter()

    try:

        color_converter.to_rgb(color)

        return True

    except(ValueError):

        return False


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

    return configuration_is_ok
