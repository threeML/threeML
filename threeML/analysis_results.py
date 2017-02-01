import numpy as np
import uncertainties
import astromodels
from operator import attrgetter, itemgetter
import pandas as pd
import collections
import math
import inspect
import functools

from threeML.utils.uncertainties_regexpr import get_uncertainty_tokens
from threeML.io.table import NumericMatrix, long_path_formatter
from threeML.io.rich_display import display
from threeML.exceptions.custom_exceptions import custom_warnings

def _order_of_magnitude(value):

    return 10 ** np.floor(np.log10(abs(value)))

def _interval_formatter(value, low_bound, hi_bound):
    """
    Gets a value and its error in input, and returns the value, the uncertainty and the common exponent with the proper
    number of significant digits

    :param value:
    :param error: a *positive* value
    :return: (num, unc, exponent)
    """

    # Get the errors (instead of the boundaries)

    error_m = low_bound - value
    error_p = hi_bound - value

    # Compute the sign of the errors
    # NOTE: sometimes value is not within low_bound - hi_bound, so these sign might not always
    # be -1 and +1 respectively

    sign_m = _sign(low_bound - value)
    sign_p = _sign(hi_bound - value)

    # Scale the values to the order of magnitude of the value

    order_of_magnitude = max([_order_of_magnitude(value), _order_of_magnitude(error_m), _order_of_magnitude(error_p)])

    scaled_value = value / order_of_magnitude
    scaled_error_m = error_m / order_of_magnitude
    scaled_error_p = error_p / order_of_magnitude

    # Get the uncertainties instance of the scaled values/errors

    x = uncertainties.ufloat(scaled_value, abs(scaled_error_m))

    # Split the uncertainty in number, negative error, and exponent (if any)

    num1, unc1, exponent1 = get_uncertainty_tokens(x)

    # Since we scaled to the order of magnitude of value, there shouldn't be any exponent

    assert exponent1 is None

    # Repeat the same for the other error

    y = uncertainties.ufloat(scaled_value, abs(scaled_error_p))

    num2, unc2, exponent2 = get_uncertainty_tokens(y)

    assert exponent2 is None

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


class RandomVariates(np.ndarray):
    """
    A subclass of np.array which is meant to contain samples for one parameter. This class contains methods to easily
    compute properties for the parameter (errors and so on)
    """
    def __new__(cls, input_array, value=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type

        obj = np.asarray(input_array).view(cls)

        # Add the value
        obj._orig_value = value

        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):

        # see InfoArray.__array_finalize__ for comments
        if obj is None: return

        # Add the value

        self._orig_value = getattr(obj, '_orig_value', None)

    def __array_wrap__(self, out_arr):

        # This gets called at the end of any operation, where out_arr is the result of the operation
        # We need to update _orig_value so that the final results will have it

        out_arr._orig_value = out_arr.median

        # then just call the parent
        return np.ndarray.__array_wrap__(self, out_arr)

    @property
    def median(self):
        """Returns median value"""

        # the np.asarray casting avoids the calls to __new__ and __array_finalize_ of this class

        return float(np.median(np.asarray(self)))

    @property
    def average(self):
        """Returns average value"""

        return float(np.asarray(self).mean())

    @property
    def value(self):

        return float(self._orig_value)

    @property
    def samples(self):

        return np.asarray(self)

    def highest_posterior_density_interval(self, cl=0.68):
        """
        Returns the Highest Posterior Density interval (HPD) for the parameter, for the given credibility level.

        NOTE: the returned interval is the HPD only if the posterior is not multimodal. If it is multimodal, you should
        probably report the full posterior, not only an interval.

        :param cl: credibility level (0 < cl < 1)
        :return: (low_bound, hi_bound)
        """

        assert 0 < cl < 1, "The credibility level should be 0 < cl < 1"

        # NOTE: we cannot sort the array, because we would destroy the covariance with other physical quantities,
        # so we get a copy instead. This copy will live only for the duration of this method (but of course will be
        # collected only whenevery the garbage collector decides to).

        ordered = np.sort(np.array(self))

        n = ordered.size

        # This is the probability that the interval should span
        interval_integral = 1.0 - cl

        # If all values have the same probability, then the hpd is degenerate, but its length is from 0 to
        # the value corresponding to the (interval_integral * n)-th sample.
        # This is the index of the rightermost element which can be part of the interval

        index_of_rightmost_possibility = int(np.floor(interval_integral * n))

        # Compute the index of the last element that is eligible to be the left bound of the interval

        index_of_leftmost_possibility = n - index_of_rightmost_possibility

        # Now compute the width of all intervals that might be the one we are looking for

        interval_width = ordered[index_of_rightmost_possibility:] - ordered[:index_of_leftmost_possibility]

        # This might happen if there are too few values
        if len(interval_width) == 0:

            raise RuntimeError('Too few elements for interval calculation')

        # Find the index of the shortest interval

        idx_of_minimum = np.argmin(interval_width)

        # Find the extremes of the shortest interval

        hpd_left_bound = ordered[idx_of_minimum]
        hpd_right_bound = ordered[idx_of_minimum + index_of_rightmost_possibility]

        return hpd_left_bound, hpd_right_bound

    def equal_tail_confidence_interval(self, cl=0.68):
        """
        Returns the equal tail confidence interval, i.e., an interval centered on the median of the distribution with
        the same probability on the right and on the left of the mean.

        If the distribution of the parameter is Gaussian and cl=0.68, this is equivalent to the 1 sigma confidence
        interval.

        :param cl: confidence level (0 < cl < 1)
        :return: (low_bound, hi_bound)
        """

        assert 0 < cl < 1, "Confidence level must be 0 < cl < 1"

        half_cl = cl / 2.0 * 100.0

        low_bound, hi_bound = np.percentile(np.asarray(self), [50.0 - half_cl, 50.0 + half_cl])

        return float(low_bound), float(hi_bound)

    # np.ndarray already has a mean() and a std() methods

    def __repr__(self):

        # Get representation for the HPD

        min_bound, max_bound = self.highest_posterior_density_interval(0.68)

        hpd_string = _interval_formatter(self.median, min_bound, max_bound)

        # Get representation for the equal-tail interval

        min_bound, max_bound = self.equal_tail_confidence_interval(0.68)

        eqt_string = _interval_formatter(self.median, min_bound, max_bound)

        # Put them together

        representation = "equal-tail: %s, hpd: %s" % (eqt_string, hpd_string)

        return representation

    def __str__(self):

        return self.__repr__()


class _AnalysisResults(object):

    """
    A unified class to store results from a maximum likelihood or a Bayesian analysis, which provides a unique interface
    and allows for "error propagation" (which means different things in the two contexts) in arbitrary expressions.

    This class is not intended for public consumption. Use either the MLEResults or the BayesianResults subclasses.

    :param optimized_model: a Model instance with the optimized values of the parameters. A clone will be stored within
    the class, so there is no need to clone it before hand
    :type optimized_model: astromodels.Model
    :param samples: the samples for the parameters
    :type samples: np.ndarray
    :param statistic_values: a dictionary containing the statistic (likelihood or posterior) values for the different
    datasets
    :type statistic_values: dict
    """

    def __init__(self, optimized_model, samples, statistic_values):

        # Safety checks

        self._n_free_parameters = len(optimized_model.free_parameters)

        assert samples.shape[1] == self._n_free_parameters, "Number of free parameters (%s) and set of samples (%s) " \
                                                            "do not agree." % (samples.shape[1],
                                                                               self._n_free_parameters)

        # NOTE: we clone the model so that whatever happens outside or after, this copy of the model will not be
        # changed

        self._optimized_model = astromodels.clone_model(optimized_model)

        # Get one instance of PhysicalQuantitySample for each free parameter
        # Put them in an ordered dictionary

        self._parameters_variates = collections.OrderedDict()

        for par_path, this_samples in zip(self._optimized_model.free_parameters.keys(), samples.T):

            this_value = self._optimized_model[par_path].value

            self._parameters_variates[par_path] = RandomVariates(this_samples, this_value)

        # Store likelihood values in a pandas Series

        self._optimal_statistic_values = pd.Series(statistic_values)

    def get_variates(self, param_path):

        assert param_path in self._optimized_model.free_parameters, "Parameter %s is not a " \
                                                                    "free parameters of the model" % param_path

        return self._parameters_variates[param_path]

    def parameters_samples_iter(self, how_many=None):

        if how_many is None:

            how_many = self._parameters_variates.values()[0].size

        for i in range(how_many):

            yield map(itemgetter(i), self._parameters_variates.values())

    @staticmethod
    def propagate(function, **kwargs):
        """
        Allow for propagation of uncertainties on arbitrary functions. It returns a function which is a wrapper around
        the provided input function. Using the wrapper with RandomVariates instances as arguments will return a
        RandomVariates result, with the errors propagated.

        Example:

        def my_function(x, a, b, c):

            return a*x**2 + b*x + c

        > p1 = analysis_results.get_variates("src.spectrum.main.composite.a_1")
        > p2 = analysis_results.get_variates("src.spectrum.main.composite.b_1")
        > wrapped_function = analysis_results.propagate(my_function, a=p1, b=p2)
        > result = wrapped_function(x=1.0, c=2.3)
        > print(result)
        equal-tail: (4.24 -0.16 +0.15) x 10, hpd: (4.24 -0.05 +0.08) x 10

        NOTE: for simple operations, you do not need to use this. This will work:

        > res = p1 + p2
        > print(res)
        equal-tail: (4.11 -0.16 +0.15) x 10, hpd: (4.11 -0.05 +0.08) x 10

        :param function: function to be wrapped
        :param **kwargs: keyword arguments specifying which random variates should substitute which argument in the
        function (see example above)
        :return: a new function, wrapping function, which can be used to propagate errors
        """

        # Get calling sequence of input function
        # arguments will be a list of names, like ['a','b']
        arguments, _, _, _ = inspect.getargspec(function)

        # Get the arguments of function which have not been specified
        # in the calling sequence (the **kwargs dictionary)
        # (they will be excluded from the vectorization)
        to_be_excluded = [item for item in arguments if item not in kwargs.keys()]

        # Vectorize the function
        vectorized = np.vectorize(function, excluded=to_be_excluded)

        # Make a wrapper so we are sure that the arguments are used in the
        # right order, as they will be taken from the kwargs
        wrapper = functools.partial(vectorized, **kwargs)

        return wrapper

    @property
    def optimized_model(self):
        """
        Returns a copy of the optimized model

        :return: a copy of the optimized model
        """

        return astromodels.clone_model(self._optimized_model)

    def estimate_covariance_matrix(self):
        """
        Estimate the covariance matrix from the samples

        :return: a covariance matrix estimated from the samples
        """

        samples = self._parameters_variates.values()

        return np.cov(samples)

    def get_correlation_matrix(self):

        raise NotImplementedError("You need to implement this")

    @property
    def optimal_statistic_values(self):

        return self._optimal_statistic_values

    def _get_correlation_matrix(self, covariance):
        """
        Compute the correlation matrix

        :return: correlation matrix
        """

        # NOTE: we compute this on-the-fly because it is of less frequent use, and contains essentially the same
        # information of the covariance matrix.

        # Compute correlation matrix

        correlation_matrix = np.zeros_like(covariance)

        for i in range(self._n_free_parameters):

            variance_i = covariance[i, i]

            for j in range(self._n_free_parameters):

                variance_j = covariance[j, j]

                if variance_i * variance_j > 0:

                    correlation_matrix[i, j] = covariance[i, j] / (math.sqrt(variance_i * variance_j))

                else:

                    # This should not happen, but it might because a fit failed or the numerical differentiation
                    # failed

                    correlation_matrix[i, j] = np.nan

        return correlation_matrix

    def get_statistic_frame(self, name):

        logl_results = {}

        logl_results[name] = self._optimal_statistic_values

        loglike_dataframe = pd.DataFrame(logl_results)

        return loglike_dataframe

    def get_data_frame(self, error_type="equal tail", cl=0.68):
        """
        Returns a pandas DataFrame with the parameters and their errors, computed as specified in "error_type" and
        with the confidence/credibility level specified in cl.

        Using "equal_tail" and cl=0.68 corresponds to the usual frequentist 1-sigma confidence interval

        :param error_type: "equal tail" or "hpd" (highest posterior density)
        :type error_type: str
        :param cl: confidence/credibility level (0 < cl < 1)
        :return: a pandas DataFrame instance
        """

        # Gather the errors

        if error_type == "equal tail":

            errors_gatherer = RandomVariates.equal_tail_confidence_interval

        elif error_type == "hpd":

            errors_gatherer = RandomVariates.equal_tail_confidence_interval

        else:

            raise ValueError("error_type must be either 'equal_tail' or 'hpd'. Got %s" % error_type)

        # Build the data frame
        values_dict = pd.Series()
        negative_error_dict = pd.Series()
        positive_error_dict = pd.Series()
        average_error_dict = pd.Series()
        units_dict = pd.Series()

        for this_par, this_phys_q in zip(self._optimized_model.free_parameters.values(),
                                         self._parameters_variates.values()):

            this_path = this_par.path

            values_dict[this_path] = this_phys_q.value

            low_bound, hi_bound = errors_gatherer(this_phys_q, cl)

            negative_error_dict[this_path] = low_bound - values_dict[this_path]
            positive_error_dict[this_path] = hi_bound - values_dict[this_path]
            average_error_dict[this_path] = (hi_bound - low_bound) / 2.0
            units_dict[this_path] = this_par.unit

        items = (('value', values_dict),
                 ('negative_error', negative_error_dict),
                 ('positive_error', positive_error_dict),
                 ('error', average_error_dict),
                 ('unit', units_dict))

        data_frame = pd.DataFrame.from_items(items)

        return data_frame

    def _get_best_fit_table(self, error_type, cl):

        fit_results = self.get_data_frame(error_type, cl)

        # Now produce an ad-hoc display. We don't use the pandas display methods because
        # we want to display uncertainties with the right number of significant numbers

        data = (('Value', pd.Series()), ('Unit', pd.Series()))

        for i, parameter_name in enumerate(fit_results.index.values):

            value = fit_results.at[parameter_name, 'value']

            negative_error = fit_results.at[parameter_name, 'negative_error']

            positive_error = fit_results.at[parameter_name, 'positive_error']

            unit = fit_results.at[parameter_name, 'unit']

            # Format the value and the error with sensible significant
            # numbers

            pretty_string = _interval_formatter(value, negative_error + value, positive_error + value)

            # Apply name formatter so long paths are shorten
            this_shortened_name = long_path_formatter(parameter_name, 40)

            data[0][1][this_shortened_name] = pretty_string
            data[1][1][this_shortened_name] = unit

        best_fit_table = pd.DataFrame.from_items(data)

        return best_fit_table


class BayesianResults(_AnalysisResults):
    """
    Store results of a Bayesian analysis (i.e., the samples) and allow for computation with them and "error propagation"

    :param optimized_model: a Model instance with the MAP values of the parameters. A clone will be stored within
    the class, so there is no need to clone it before hand
    :type optimized_model: astromodels.Model
    :param samples: the samples for the parameters
    :type samples: np.ndarray
    :param posterior_values: a dictionary containing the posterior values for the different datasets at the HPD
    :type posterior_values: dict
    """

    def __init__(self, optimized_model, samples, posterior_values):

        super(BayesianResults, self).__init__(optimized_model, samples, posterior_values)

    def get_correlation_matrix(self):
        """
        Compute correlation matrix

        :return: the correlation matrix
        """

        # Here we need to estimate the covariance from the samples, then compute the correlation matrix

        covariance = self.estimate_covariance_matrix()

        return self._get_correlation_matrix(covariance)

    def get_statistic_frame(self, name=None):

        return super(BayesianResults, self).get_statistic_frame(name='-log(posterior)')

    def display(self, display_correlation=False, error_type="equal tail", cl=0.68):

        best_fit_table = self._get_best_fit_table(error_type, cl)

        print("Maximum a posteriori probability (MAP) point:\n")

        display(best_fit_table)

        if display_correlation:

            corr_matrix = NumericMatrix(self.get_correlation_matrix())

            for col in corr_matrix.colnames:

                corr_matrix[col].format = '2.2f'

            print("\nCorrelation matrix:\n")

            display(corr_matrix)

        print("\nValues of -log(posterior) at the minimum:\n")

        display(self.get_statistic_frame())



class MLEResults(_AnalysisResults):

    """
    Build the _AnalysisResults object starting from a covariance matrix.


    :param optimized_model: best fit model
    :type optimized_model:astromodels.Model
    :param covariance_matrix:
    :type covariance_matrix: np.ndarray
    :param likelihood_values:
    :type likelihood_values: dict
    :param n_samples: Number of samples to use
    :type n_samples: int
    :return: an _AnalysisResults instance
    """

    def __init__(self, optimized_model, covariance_matrix, likelihood_values, n_samples=1000):

        # Generate samples for each parameter accounting for their covariance

        # Force covariance into proper type
        covariance_matrix = np.array(covariance_matrix, float, copy=True)

        # Get the best fit value for each parameter
        values = map(attrgetter("value"), optimized_model.free_parameters.values())

        expected_shape = (len(values), len(values))

        assert covariance_matrix.shape == expected_shape, "Covariance matrix has wrong shape. " \
                                                          "Got %s, should be %s" % (covariance_matrix.shape,
                                                                                    expected_shape)

        assert np.all(np.isfinite(covariance_matrix)), "Covariance matrix contains Nan or inf. Cannot continue."

        # Generate samples from the multivariate normal distribution, i.e., accounting for the covariance of the
        # parameters

        samples = np.random.multivariate_normal(np.array(values).T, covariance_matrix, n_samples)

        # Now reject the samples outside of the boundaries. If we reject more than 1% we warn the user

        # Gather boundaries
        # NOTE: every None boundary will become nan thanks to the casting to float
        low_bounds = np.array(map(attrgetter("min_value"), optimized_model.free_parameters.values()), float)
        hi_bounds = np.array(map(attrgetter("max_value"), optimized_model.free_parameters.values()), float)

        # Fix all nans
        low_bounds[np.isnan(low_bounds)] = -np.inf
        hi_bounds[np.isnan(hi_bounds)] = np.inf

        to_be_kept_mask = np.ones(samples.shape[0], bool)

        for i, sample in enumerate(samples):

            if np.any(sample > hi_bounds) or np.any(sample < low_bounds):

                # Remove this sample
                to_be_kept_mask[i] = False

        # Compute how many samples we have removed
        n_removed_samples = samples.shape[0] - np.sum(to_be_kept_mask)

        # Warn the user if more than 1% of the samples have been lost

        if n_removed_samples > samples.shape[0] / 100.0:

            custom_warnings.warn("%s percent of samples have been thrown away because they failed the constraints "
                                 "on the parameters. This results might not be suitable for error propagation. "
                                 "Enlarge the boundaries until you loose less than 1 percent of the samples." %
                                 (float(n_removed_samples) / samples.shape[0] * 100.0))

        # Now remove them
        samples = samples[to_be_kept_mask, :]

        # Finally build the class

        super(MLEResults, self).__init__(optimized_model, samples, likelihood_values)

        # Store the covariance matrix

        self._covariance_matrix = covariance_matrix

    @property
    def covariance_matrix(self):
        """
        Returns the covariance matrix.

        :return: covariance matrix or None (if the class was built from samples.
                 Use estimate_covariance_matrix in that case)
        """

        return self._covariance_matrix

    def get_correlation_matrix(self):
        """
        Compute correlation matrix

        :return: the correlation matrix
        """

        return self._get_correlation_matrix(self._covariance_matrix)

    # We re-implement this because the error in this case is just the sqrt(cov[i][i]) and it
    # is symmetric by contruction. However, when taking samples, the percentage could be different
    def _get_best_fit_table(self, error_type, cl):

        fit_results = self.get_data_frame(error_type, cl)

        # Now produce an ad-hoc display. We don't use the pandas display methods because
        # we want to display uncertainties with the right number of significant numbers

        data = (('Value', pd.Series()), ('Unit', pd.Series()))

        for i, parameter_name in enumerate(fit_results.index.values):

            value = fit_results.at[parameter_name, 'value']

            error = np.sqrt(self.covariance_matrix[i,i])

            unit = fit_results.at[parameter_name, 'unit']

            # Format the value and the error with sensible significant
            # numbers

            pretty_string = _interval_formatter(value,  value - error, value + error)

            # Apply name formatter so long paths are shorten
            this_shortened_name = long_path_formatter(parameter_name, 40)

            data[0][1][this_shortened_name] = pretty_string
            data[1][1][this_shortened_name] = unit

        best_fit_table = pd.DataFrame.from_items(data)

        return best_fit_table

    def get_statistic_frame(self, name=None):

        return super(MLEResults, self).get_statistic_frame(name='-log(likelihood)')

    def display(self, display_correlation=True, error_type="equal tail", cl=0.68):

        best_fit_table = self._get_best_fit_table(error_type, cl)

        print("Best fit values:\n")

        display(best_fit_table)

        if display_correlation:

            corr_matrix = NumericMatrix(self.get_correlation_matrix())

            for col in corr_matrix.colnames:

                corr_matrix[col].format = '2.2f'

            print("\nCorrelation matrix:\n")

            display(corr_matrix)

        print("\nValues of -log(likelihood) at the minimum:\n")

        display(self.get_statistic_frame())