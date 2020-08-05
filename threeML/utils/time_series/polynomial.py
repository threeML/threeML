from __future__ import division
from builtins import zip
from builtins import range
from past.utils import old_div
from builtins import object
import numpy as np
import scipy.optimize as opt
import warnings
from threeML.utils.differentiation import get_hessian, ParameterOnBoundary
from threeML.exceptions.custom_exceptions import custom_warnings


class CannotComputeCovariance(RuntimeWarning):
    pass


from threeML.config.config import threeML_config


class Polynomial(object):
    def __init__(self, coefficients, is_integral=False):

        """

        :param coefficients: array of poly coefficients
        :param is_integral: if this polynomial is an
        """
        self._coefficients = coefficients
        self._degree = len(coefficients) - 1

        self._i_plus_1 = np.array(list(range(1, self._degree + 1 + 1)), dtype=float)

        self._cov_matrix = np.zeros((self._degree + 1, self._degree + 1))

        # we can fix some things for speed
        # we only need to set the coeff for the
        # integral polynomial
        if not is_integral:

            integral_coeff = [0]

            integral_coeff.extend(
                [
                    self._coefficients[i - 1] / float(i)
                    for i in range(1, self._degree + 1 + 1)
                ]
            )

            self._integral_polynomial = Polynomial(integral_coeff, is_integral=True)

    @classmethod
    def from_previous_fit(cls, coefficients, covariance):

        poly = Polynomial(coefficients=coefficients)
        poly._cov_matrix = covariance

        return poly

    @property
    def degree(self):
        """
        the polynomial degree
        :return:
        """
        return self._degree

    @property
    def error(self):
        """
        the error on the polynomial coefficients
        :return:
        """
        return np.sqrt(self._cov_matrix.diagonal())

    def __get_coefficient(self):
        """ gets the coefficients"""

        return self._coefficients

    def ___get_coefficient(self):
        """ Indirect coefficient getter """

        return self.__get_coefficient()

    def __set_coefficient(self, val):
        """ sets the coefficients"""

        self._coefficients = val

        integral_coeff = [0]

        integral_coeff.extend(
            [
                self._coefficients[i - 1] / float(i)
                for i in range(1, self._degree + 1 + 1)
            ]
        )

        self._integral_polynomial = Polynomial(integral_coeff, is_integral=True)

    def ___set_coefficient(self, val):
        """ Indirect coefficient setter """

        return self.__set_coefficient(val)

    coefficients = property(
        ___get_coefficient,
        ___set_coefficient,
        doc="""Gets or sets the coefficients of the polynomial.""",
    )

    def __call__(self, x):

        result = 0
        for coefficient in self._coefficients[::-1]:
            result = result * x + coefficient
        return result

    def compute_covariance_matrix(self, function, best_fit_parameters):
        """
        Compute the covariance matrix of this fit
        :param function: the loglike for the fit
        :param best_fit_parameters: the best fit parameters
        :return:
        """

        minima = np.zeros_like(best_fit_parameters) - 100
        maxima = np.zeros_like(best_fit_parameters) + 100

        try:

            hessian_matrix = get_hessian(function, best_fit_parameters, minima, maxima)

        except ParameterOnBoundary:

            custom_warnings.warn(
                "One or more of the parameters are at their boundaries. Cannot compute covariance and"
                " errors",
                CannotComputeCovariance,
            )

            n_dim = len(best_fit_parameters)

            self._cov_matrix = np.zeros((n_dim, n_dim)) * np.nan

        # Invert it to get the covariance matrix

        try:

            covariance_matrix = np.linalg.inv(hessian_matrix)

            self._cov_matrix = covariance_matrix

        except:

            custom_warnings.warn(
                "Cannot invert Hessian matrix, looks like the matrix is singluar"
            )

            n_dim = len(best_fit_parameters)

            self._cov_matrix = np.zeros((n_dim, n_dim)) * np.nan

    @property
    def covariance_matrix(self):
        return self._cov_matrix

    def integral(self, xmin, xmax):
        """ 
        Evaluate the integral of the polynomial between xmin and xmax

        """

        return self._integral_polynomial(xmax) - self._integral_polynomial(xmin)

    def _eval_basis(self, x):

        return (1.0 / self._i_plus_1) * np.power(x, self._i_plus_1)

    def integral_error(self, xmin, xmax):
        """
        computes the integral error of an interval
        :param xmin: start of the interval
        :param xmax: stop of the interval
        :return: interval error
        """
        c = self._eval_basis(xmax) - self._eval_basis(xmin)
        tmp = c.dot(self._cov_matrix)
        err2 = tmp.dot(c)

        return np.sqrt(err2)


class PolyLogLikelihood(object):
    def __init__(self, model, exposure):

        self._model = model
        self._parameters = model.coefficients
        self._exposure = exposure

        # build the covariance call
        self._build_cov_call()

    def _evaluate_logM(self, M):
        # Evaluate the logarithm with protection for negative or small
        # numbers, using a smooth linear extrapolation (better than just a sharp
        # cutoff)
        tiny = np.float64(np.finfo(M[0]).tiny)

        non_tiny_mask = M > 2.0 * tiny

        tink_mask = np.logical_not(non_tiny_mask)

        if tink_mask.sum() > 0:
            logM = np.zeros(len(M))
            logM[tink_mask] = old_div(np.abs(M[tink_mask]), tiny) + np.log(tiny) - 1
            logM[non_tiny_mask] = np.log(M[non_tiny_mask])

        else:

            logM = np.log(M)

        return logM

    def _fix_precision(self, v):
        """
          Round extremely small number inside v to the smallest usable
          number of the type corresponding to v. This is to avoid warnings
          and errors like underflows or overflows in math operations.
        """
        tiny = np.float64(np.finfo(v[0]).tiny)
        zero_mask = np.abs(v) <= tiny  # type: np.ndarray
        if zero_mask.sum() > 0:
            v[zero_mask] = np.sign(v[zero_mask]) * tiny

        return v, tiny

    def _build_cov_call(self):

        raise NotImplementedError("must be built in subclass")


class PolyBinnedLogLikelihood(PolyLogLikelihood):
    """
    Implements a Poisson likelihood (i.e., the Cash statistic). Mind that this is not
    the Castor statistic (Cstat). The difference between the two is a constant given
    a dataset. I kept Cash instead of Castor to make easier the comparison with ROOT
    during tests, since ROOT implements the Cash statistic.
    """

    def __init__(self, x, y, model, exposure):
        self._bin_centers = x
        self._counts = y

        self._non_zero_mask = self._counts > 0

        super(PolyBinnedLogLikelihood, self).__init__(model, exposure)

    def _build_cov_call(self):
        def cov_call(*parameters):

            self._model.coefficients = parameters

            M = self._model(self._bin_centers) * self._exposure
            M_fixed, tiny = self._fix_precision(M)

            # Replace negative values for the model (impossible in the Poisson context)
            # with zero

            negative_mask = M < 0
            if negative_mask.sum() > 0:
                M[negative_mask] = 0.0

            # Poisson loglikelihood statistic (Cash) is:
            # L = Sum ( M_i - D_i * log(M_i))

            logM = self._evaluate_logM(M)

            # Evaluate v_i = D_i * log(M_i): if D_i = 0 then the product is zero
            # whatever value has log(M_i). Thus, initialize the whole vector v = {v_i}
            # to zero, then overwrite the elements corresponding to D_i > 0

            d_times_logM = np.zeros(len(self._counts))

            d_times_logM[self._non_zero_mask] = (
                self._counts[self._non_zero_mask] * logM[self._non_zero_mask]
            )

            log_likelihood = np.sum(M_fixed - d_times_logM)

            return log_likelihood

        self.cov_call = cov_call

    def __call__(self, parameters):
        """
          Evaluate the Cash statistic for the given set of parameters
        """

        # Compute the values for the model given this set of parameters
        # model is in counts

        self._model.coefficients = parameters
        M = self._model(self._bin_centers) * self._exposure
        M_fixed, tiny = self._fix_precision(M)

        # Replace negative values for the model (impossible in the Poisson context)
        # with zero

        negative_mask = M < 0
        if negative_mask.sum() > 0:
            M[negative_mask] = 0.0

        # Poisson loglikelihood statistic (Cash) is:
        # L = Sum ( M_i - D_i * log(M_i))

        logM = self._evaluate_logM(M)

        # Evaluate v_i = D_i * log(M_i): if D_i = 0 then the product is zero
        # whatever value has log(M_i). Thus, initialize the whole vector v = {v_i}
        # to zero, then overwrite the elements corresponding to D_i > 0

        d_times_logM = np.zeros_like(self._counts)

        d_times_logM[self._non_zero_mask] = (
            self._counts[self._non_zero_mask] * logM[self._non_zero_mask]
        )

        log_likelihood = np.sum(M_fixed - d_times_logM)

        return log_likelihood


class PolyUnbinnedLogLikelihood(PolyLogLikelihood):
    """
    Implements a Poisson likelihood (i.e., the Cash statistic). Mind that this is not
    the Castor statistic (Cstat). The difference between the two is a constant given
    a dataset. I kept Cash instead of Castor to make easier the comparison with ROOT
    during tests, since ROOT implements the Cash statistic.
    """

    def __init__(self, events, model, t_start, t_stop, exposure):

        self._events = events
        self._t_start = t_start  # list of starts
        self._t_stop = t_stop

        super(PolyUnbinnedLogLikelihood, self).__init__(model, exposure)

    def _build_cov_call(self):
        def cov_call(*parameters):

            # Compute the values for the model given this set of parameters
            self._model.coefficients = parameters

            # Integrate the polynomial (or in the future, model) over the given interval

            n_expected_counts = 0.0

            for start, stop in zip(self._t_start, self._t_stop):
                n_expected_counts += self._model.integral(start, stop)

            # Now evaluate the model at the event times and multiply by the exposure

            M = self._model(self._events) * self._exposure

            # Replace negative values for the model (impossible in the Poisson context)
            # with zero
            negative_mask = M < 0

            if negative_mask.sum() > 0:
                M[negative_mask] = 0.0

            # Poisson loglikelihood statistic  is:
            # logL = -Nexp + Sum ( log M_i )

            logM = self._evaluate_logM(M)

            log_likelihood = -n_expected_counts + logM.sum()

            return -log_likelihood

        self.cov_call = cov_call

    def __call__(self, parameters):
        """

        """

        # Compute the values for the model given this set of parameters
        self._model.coefficients = parameters

        # Integrate the polynomial (or in the future, model) over the given interval

        n_expected_counts = 0.0

        for start, stop in zip(self._t_start, self._t_stop):
            n_expected_counts += self._model.integral(start, stop)

        # Now evaluate the model at the event times and multiply by the exposure

        M = self._model(self._events) * self._exposure

        # Replace negative values for the model (impossible in the Poisson context)
        # with zero
        negative_mask = M < 0

        if negative_mask.sum() > 0:
            M[negative_mask] = 0.0

        # Poisson loglikelihood statistic  is:
        # logL = -Nexp + Sum ( log M_i )

        logM = self._evaluate_logM(M)

        log_likelihood = -n_expected_counts + logM.sum()

        return -log_likelihood


def polyfit(x, y, grade, exposure):
    """ function to fit a polynomial to event data. not a member to allow parallel computation """

    # Check that we have enough counts to perform the fit, otherwise
    # return a "zero polynomial"
    non_zero_mask = y > 0
    n_non_zero = non_zero_mask.sum()
    if n_non_zero == 0:
        # No data, nothing to do!
        return Polynomial([0.0]), 0.0

    # Compute an initial guess for the polynomial parameters,
    # with a least-square fit (with weight=1) using SVD (extremely robust):
    # (note that polyfit returns the coefficient starting from the maximum grade,
    # thus we need to reverse the order)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        initial_guess = np.polyfit(x, y, grade)

    initial_guess = initial_guess[::-1]

    polynomial = Polynomial(initial_guess)

    # Check that the solution found is meaningful (i.e., definite positive
    # in the interval of interest)
    M = polynomial(x)

    negative_mask = M < 0

    if negative_mask.sum() > 0:
        # Least square fit failed to converge to a meaningful solution
        # Reset the initialGuess to reasonable value
        initial_guess[0] = np.mean(y)
        meanx = np.mean(x)
        initial_guess = [
            old_div(abs(i[1]), pow(meanx, i[0])) for i in enumerate(initial_guess)
        ]

    # Improve the solution using a logLikelihood statistic (Cash statistic)
    log_likelihood = PolyBinnedLogLikelihood(x, y, polynomial, exposure)

    # Check that we have enough non-empty bins to fit this grade of polynomial,
    # otherwise lower the grade
    dof = n_non_zero - (grade + 1)

    if dof <= 2:
        # Fit is poorly or ill-conditioned, have to reduce the number of parameters
        while dof < 2 and len(initial_guess) > 1:
            initial_guess = initial_guess[:-1]
            polynomial = Polynomial(initial_guess)
            log_likelihood = PolyBinnedLogLikelihood(x, y, polynomial, exposure)

    # Try to improve the fit with the log-likelihood

    final_estimate = opt.minimize(
        log_likelihood,
        initial_guess,
        method=threeML_config["event list"]["binned fit method"],
        options=threeML_config["event list"]["binned fit options"],
    )["x"]
    final_estimate = np.atleast_1d(final_estimate)

    # Get the value for cstat at the minimum

    min_log_likelihood = log_likelihood(final_estimate)

    # Update the polynomial with the fitted parameters,
    # and the relative covariance matrix

    final_polynomial = Polynomial(final_estimate)

    final_polynomial.compute_covariance_matrix(log_likelihood.cov_call, final_estimate)

    return final_polynomial, min_log_likelihood


def unbinned_polyfit(events, grade, t_start, t_stop, exposure, initial_amplitude=1):
    """
    function to fit a polynomial to event data. not a member to allow parallel computation

    """

    # first do a simple amplitude fit
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        search_grid = np.logspace(-2, 4, 10)

        initial_guess = np.zeros(grade + 1)

        polynomial = Polynomial(initial_guess)

        # if there are no events then return nothing

        if len(events) == 0:

            return Polynomial([0]), 0

        log_likelihood = PolyUnbinnedLogLikelihood(
            events, polynomial, t_start, t_stop, exposure
        )

        like_grid = []
        for amp in search_grid:

            initial_guess[0] = amp
            like_grid.append(log_likelihood(initial_guess))

        initial_guess[0] = search_grid[np.argmin(like_grid)]

        # Improve the solution
        dof = len(events) - (grade + 1)

        if dof <= 2:
            # Fit is poorly or ill-conditioned, have to reduce the number of parameters
            while dof < 1 and len(initial_guess) > 1:
                initial_guess = initial_guess[:-1]
                polynomial = Polynomial(initial_guess)
                log_likelihood = PolyUnbinnedLogLikelihood(
                    events, polynomial, t_start, t_stop, exposure
                )

        final_estimate = opt.minimize(
            log_likelihood,
            initial_guess,
            method=threeML_config["event list"]["unbinned fit method"],
            options=threeML_config["event list"]["unbinned fit options"],
        )["x"]

        final_estimate = np.atleast_1d(final_estimate)

        min_log_likelihood = log_likelihood(final_estimate)

    # Update the polynomial with the fitted parameters,
    # and the relative covariance matrix

    final_polynomial = Polynomial(final_estimate)

    final_polynomial.compute_covariance_matrix(log_likelihood.cov_call, final_estimate)

    return final_polynomial, min_log_likelihood
