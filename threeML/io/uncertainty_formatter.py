from __future__ import division
from past.utils import old_div
import uncertainties
import re
import numpy as np


def interval_to_errors(value, low_bound, hi_bound):
    """
    Convert error intervals to errors
    
    :param value: central value 
    :param low_bound: interval low bound
    :param hi_bound: interval high bound
    :return: (error minus, error plus)
    """

    error_plus = hi_bound - value
    error_minus = value - low_bound

    return error_minus, error_plus


def get_uncertainty_tokens(x):
    """
    Split the given uncertainty in number, error and exponent.

    :param x: an uncertainty instance
    :return: number, error and exponent
    """

    try:

        number, uncertainty, exponent = re.match(
            "\(?(\-?[0-9]+\.?[0-9]*) ([0-9]+\.?[0-9]*)\)?(e[\+|\-][0-9]+)?",
            x.__str__().replace("+/-", " ").replace("nan", "0"),
        ).groups()

    except:

        raise RuntimeError(
            "Could not extract number, uncertainty and exponent from %s. "
            "This is likely a bug." % x.__str__()
        )

    return number, uncertainty, exponent


def _order_of_magnitude(value):

    return 10 ** np.floor(np.log10(abs(value)))


def uncertainty_formatter(value, low_bound, hi_bound):
    """
    Gets a value and its error in input, and returns the value, the uncertainty and the common exponent with the proper
    number of significant digits in a string like (4.2 -0.023 +5.23) x 10^5

    :param value:
    :param error: a *positive* value
    :return: string representation of interval
    """

    # Get the errors (instead of the boundaries)

    error_m, error_p = interval_to_errors(value, low_bound, hi_bound)

    # Compute the sign of the errors
    # NOTE: sometimes value is not within low_bound - hi_bound, so these sign might not always
    # be -1 and +1 respectively

    sign_m = _sign(low_bound - value)
    sign_p = _sign(hi_bound - value)

    # Scale the values to the order of magnitude of the value

    order_of_magnitude = max(
        [
            _order_of_magnitude(value),
            _order_of_magnitude(error_m),
            _order_of_magnitude(error_p),
        ]
    )

    scaled_value = old_div(value, order_of_magnitude)
    scaled_error_m = old_div(error_m, order_of_magnitude)
    scaled_error_p = old_div(error_p, order_of_magnitude)

    # Get the uncertainties instance of the scaled values/errors

    x = uncertainties.ufloat(scaled_value, abs(scaled_error_m))

    # Split the uncertainty in number, negative error, and exponent (if any)

    num1, unc1, exponent1 = get_uncertainty_tokens(x)

    # Repeat the same for the other error

    y = uncertainties.ufloat(scaled_value, abs(scaled_error_p))

    num2, unc2, exponent2 = get_uncertainty_tokens(y)

    # Choose the representation of the number with more digits
    # This is necessary for asymmetric intervals where one of the two errors is much larger in magnitude
    # then the others. For example, 1 -0.01 +90. This will choose 1.00 instead of 1,so that the final
    # representation will be 1.00 -0.01 +90

    if len(num1) > len(num2):

        num = num1

    else:

        num = num2

    # Get the exponent of 10 to use for the representation

    expon = int(np.log10(order_of_magnitude))

    if unc1 != unc2:

        # Asymmetric error

        repr1 = "%s%s" % (sign_m, unc1)
        repr2 = "%s%s" % (sign_p, unc2)

        if expon == 0:

            # No need to show any power of 10

            return "%s %s %s" % (num, repr1, repr2)

        elif expon == 1:

            # Display 10 instead of 10^1

            return "(%s %s %s) x 10" % (num, repr1, repr2)

        else:

            # Display 10^expon

            return "(%s %s %s) x 10^%s" % (num, repr1, repr2, expon)

    else:

        # Symmetric error
        repr1 = "+/- %s" % unc1

        if expon == 0:

            return "%s %s" % (num, repr1)

        elif expon == 1:

            return "(%s %s) x 10" % (num, repr1)

        else:

            return "(%s %s) x 10^%s" % (num, repr1, expon)


def _sign(number):

    if number < 0:

        return "-"

    else:

        return "+"
