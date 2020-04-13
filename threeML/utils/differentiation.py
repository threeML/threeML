import numdifftools as nd
import numpy as np
from astromodels import SettingOutOfBounds


class ParameterOnBoundary(RuntimeError):
    pass


class CannotComputeHessian(RuntimeError):
    pass


def _get_wrapper(function, point, minima, maxima):

    point = np.array(point, ndmin=1, dtype=float)
    minima = np.array(minima, ndmin=1, dtype=float)
    maxima = np.array(maxima, ndmin=1, dtype=float)

    n_dim = point.shape[0]

    # Find order of magnitude of each coordinate. If one of the coordinates is exactly zero we need
    # to treat it differently
    idx = point == 0.0

    orders_of_magnitude = np.zeros_like(point)
    orders_of_magnitude[idx] = 1.0
    orders_of_magnitude[~idx] = 10 ** np.ceil(
        np.log10(np.abs(point[~idx]))
    )  # type: np.ndarray

    scaled_point = point / orders_of_magnitude
    scaled_minima = minima / orders_of_magnitude
    scaled_maxima = maxima / orders_of_magnitude

    # Decide a delta for the finite differentiation
    # The algorithm implemented in numdifftools is robust with respect to the choice
    # of delta, as long as we are not going beyond the boundaries (which would cause
    # the procedure to fail)

    scaled_deltas = np.zeros_like(scaled_point)

    for i in range(n_dim):

        scaled_value = scaled_point[i]

        scaled_min_value, scaled_max_value = (scaled_minima[i], scaled_maxima[i])

        if scaled_value == scaled_min_value or scaled_value == scaled_max_value:

            raise ParameterOnBoundary(
                "Value for parameter number %s is on the boundary" % i
            )

        if not np.isnan(scaled_min_value):

            # Parameter with low bound

            distance_to_min = scaled_value - scaled_min_value

        else:

            # No defined minimum

            distance_to_min = np.inf

        if not np.isnan(scaled_max_value):

            # Parameter with hi bound

            distance_to_max = scaled_max_value - scaled_value

        else:

            # No defined maximum

            distance_to_max = np.inf

        # Delta is the minimum between 0.03% of the value, and 1/2.5 times the minimum
        # distance to either boundary. 1/2 of that factor is due to the fact that numdifftools uses
        # twice the delta to compute the differential, and the 0.5 is due to the fact that we don't want
        # to go exactly equal to the boundary

        if scaled_point[i] == 0.0:

            scaled_deltas[i] = min([1e-5, distance_to_max / 2.5, distance_to_min / 2.5])

        else:

            scaled_deltas[i] = min(
                [
                    0.003 * abs(scaled_point[i]),
                    distance_to_max / 2.5,
                    distance_to_min / 2.5,
                ]
            )

    def wrapper(x):

        scaled_back_x = x * orders_of_magnitude  # type: np.ndarray

        try:

            result = function(*scaled_back_x)

        except SettingOutOfBounds:

            raise CannotComputeHessian(
                "Cannot compute Hessian, parameters out of bounds at %s" % scaled_back_x
            )

        else:

            return result

    return wrapper, scaled_deltas, scaled_point, orders_of_magnitude, n_dim


def get_jacobian(function, point, minima, maxima):

    wrapper, scaled_deltas, scaled_point, orders_of_magnitude, n_dim = _get_wrapper(
        function, point, minima, maxima
    )

    # Compute the Jacobian matrix at best_fit_values
    jacobian_vector = nd.Jacobian(wrapper, scaled_deltas, method="central")(
        scaled_point
    )

    # Transform it to numpy matrix

    jacobian_vector = np.array(jacobian_vector)

    # Now correct back the Jacobian for the scales

    jacobian_vector /= orders_of_magnitude

    return jacobian_vector[0]


def get_hessian(function, point, minima, maxima):

    wrapper, scaled_deltas, scaled_point, orders_of_magnitude, n_dim = _get_wrapper(
        function, point, minima, maxima
    )

    # Compute the Hessian matrix at best_fit_values

    hessian_matrix_ = nd.Hessian(wrapper, scaled_deltas)(scaled_point)

    # Transform it to numpy matrix

    hessian_matrix = np.array(hessian_matrix_)

    # Now correct back the Hessian for the scales
    for i in range(n_dim):

        for j in range(n_dim):

            hessian_matrix[i, j] /= orders_of_magnitude[i] * orders_of_magnitude[j]

    return hessian_matrix
