__author__ = 'drJfunk'

from threeML.config.config import threeML_config
import matplotlib.colors as colors


class ConfigFileError(RuntimeError):
    pass


_color_keys = [['ogip', 'counts color'],
               ['ogip', 'background color'],
               ['gbm', 'lightcurve color'],
               ['gbm', 'selection color'],
               ['gbm', 'background color'],
               ['gbm', 'background selection color']
               ]

_cmap_keys = [['ogip', 'data plot cmap'], ]


def is_matplotlib_cmap():
    test = True

    return test


def is_matplotlib_color():
    test = True

    return test


def check_configuration():
    """
    A routine to make sure that user specified configurations
    are indeed valid.

    :return:
    """

    for key in _color_keys:

        color_converter = colors.ColorConverter()

        try:
            color_to_try = threeML_config[key[0]][key[1]]

            color_converter.to_rgb(color_to_try)

        except(ValueError):

            raise ConfigFileError("The key: %s of %s is not a valid color string" % (key[0], key[1]))
