import numdifftools as nd
import numpy as np

from astromodels import SettingOutOfBounds


class ParameterOnBoundary(RuntimeError):
    pass


class CannotComputeHessian(RuntimeError):
    pass


def get_hessian(function, point, minima, maxima):

    # Define a wrapper because numdifftools expect the function to be f(x) with
    # x a vector, while the likelihood function expect f(x1,x2,x3...)

    # Also, compute the orders of magnitude of the parameters, which we will use to make the values used
    # by numdifftools close to 1

    try:

        n_dim = point.shape[0]
        _ = minima.shape
        _ = maxima.shape

    except AttributeError:

        point = np.array(point, ndmin=1)
        minima = np.array(minima, ndmin=1)
        maxima = np.array(maxima, ndmin=1)

        n_dim = point.shape[0]

    orders_of_magnitude = 10**np.ceil(np.log10(np.abs(point)))

    scaled_point = point / orders_of_magnitude
    scaled_minima = minima / orders_of_magnitude
    scaled_maxima = maxima / orders_of_magnitude

    def wrapper(x):

        scaled_back_x = x * orders_of_magnitude

        try:

            result = function(*scaled_back_x)

        except SettingOutOfBounds:

            raise CannotComputeHessian("Cannot compute Hessian, parameters out of bounds at %s" % scaled_back_x)

        else:

            return result

    # Decide a delta for the finite differentiation
    # The algorithm implemented in numdifftools is robust with respect to the choice
    # of delta, as long as we are not going beyond the boundaries (which would cause
    # the procedure to fail)

    scaled_deltas = np.zeros_like(scaled_point)

    for i in range(n_dim):

        scaled_value = scaled_point[i]

        scaled_min_value, scaled_max_value = (scaled_minima[i], scaled_maxima[i])

        if scaled_value == scaled_min_value or scaled_value == scaled_max_value:

            raise ParameterOnBoundary("Value for parameter number %s is on the boundary" % i)

        if scaled_min_value is not None:

            # Parameter with low bound

            distance_to_min = scaled_value - scaled_min_value

        else:

            # No defined minimum

            distance_to_min = np.inf

        if scaled_max_value is not None:

            # Parameter with hi bound

            distance_to_max = scaled_max_value - scaled_value

        else:

            # No defined maximum

            distance_to_max = np.inf

        # Delta is the minimum between 3% of the value, and half of the minimum
        # distance to either boundary

        scaled_deltas[i] = min([0.03 * abs(scaled_point[i]), distance_to_max / 2.0, distance_to_min / 2.0])

    # Compute the Hessian matrix at best_fit_values

    hessian_matrix_ = nd.Hessian(wrapper, scaled_deltas)(scaled_point)

    # Transform it to numpy matrix

    hessian_matrix = np.array(hessian_matrix_)

    # Now correct back the Hessian for the scales
    for i in range(n_dim):

        for j in range(n_dim):

            hessian_matrix[i,j] /= orders_of_magnitude[i] * orders_of_magnitude[j]

    return hessian_matrix