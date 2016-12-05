__author__ = 'drJfunk'

# from threeML.config.config import threeML_config
import matplotlib.colors as colors
import matplotlib.pyplot as plt


class ConfigFileWarning(RuntimeWarning):
    pass


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


def check_configuration(threeML_config):
    """
    A routine to make sure that user specified configurations
    are indeed valid.

    :return:
    """

    for key in _color_keys:

        color_to_try = threeML_config[key[0]][key[1]]

        if not is_matplotlib_color(color_to_try):

            raise ConfigFileWarning("The key: %s of %s is not a valid color string" % (key[0], key[1]))

    for key in _cmap_keys:

        cmap_to_try = threeML_config[key[0]][key[1]]

        if not is_matplotlib_cmap(cmap_to_try):

            raise ConfigFileWarning("The key: %s of %s is not a valid cmap string" % (key[0], key[1]))
