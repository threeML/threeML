# This is the regular expression to use when dealing with the output from the uncertainty package

import re


def get_uncertainty_tokens(x):
    """
    Split the given uncertainty in number, error and exponent.

    :param x: an uncertainty instance
    :return: number, error and exponent
    """

    try:

        number, uncertainty, exponent = re.match('\(?(\-?[0-9]+\.?[0-9]*) ([0-9]+\.?[0-9]*)\)?(e[\+|\-][0-9]+)?',
                                                 x.__str__().replace("+/-", " ")).groups()

    except:

        raise RuntimeError("Could not extract number, uncertainty and exponent from %s. "
                           "This is likely a bug." % x.__str__())

    return number, uncertainty, exponent