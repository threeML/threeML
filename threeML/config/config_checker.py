# from threeML.config.config import threeML_config
import matplotlib.colors as colors
import matplotlib.pyplot as plt

from threeML.exceptions.custom_exceptions import custom_warnings
from threeML.exceptions import custom_exceptions


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

    # First get the configuration file from the distribution root (which is *assumed* to be correct)


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
