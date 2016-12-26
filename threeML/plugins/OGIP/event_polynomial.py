import numpy as np
import scipy.optimize as opt
import warnings
import math

from threeML.config.config import threeML_config



class Polynomial(object):
    def __init__(self, coefficients):

        self.coefficients = coefficients
        self._degree = len(coefficients) - 1

        # Build an empty covariance matrix
        self._cov_matrix = np.zeros([self._degree + 1, self._degree + 1])

    @property
    def degree(self):
        return self._degree

    @property
    def error(self):

        return np.sqrt(self._cov_matrix.diagonal())

    def __call__(self, x):

        result = 0
        for coefficient in self.coefficients[::-1]:
            result = result * x + coefficient
        return result

    def get_number_free_parameters(self):
        return self._degree + 1

    def get_free_derivs(self, x):
        n_par = self._degree + 1
        freeDerivs = []

        for i in range(n_par):
            freeDerivs.append(map(lambda xx: pow(xx, i), x))
        pass
        return np.array(freeDerivs)

    pass

    def compute_covariance_matrix(self, statistic_gradient):

        self._cov_matrix = compute_covariance_matrix(statistic_gradient, self.coefficients)

        # Check that the covariance matrix is positive-defined

        negative_elements = (np.matrix.diagonal(self._cov_matrix) < 0)

        if (len(negative_elements.nonzero()[0]) > 0):
            raise RuntimeError(
                    "Negative element in the diagonal of the covariance matrix. Try to reduce the polynomial grade.")

    @property
    def covariance_matrix(self):
        return self._cov_matrix

    def integral(self, xmin, xmax):
        """ 
        Evaluate the integral of the polynomial between xmin and xmax
        """

        integral_coeff = [0]

        integral_coeff.extend(map(lambda i: self.coefficients[i - 1] / float(i), range(1, self._degree + 1 + 1)))

        integral_polynomial = Polynomial(integral_coeff)

        return integral_polynomial(xmax) - integral_polynomial(xmin)

    def integral_error(self, xmin, xmax):
        # Based on http://root.cern.ch/root/html/tutorials/fit/ErrorIntegral.C.html


        # Set the weights
        i_plus_1 = np.array(range(1, self._degree + 1 + 1), 'd')

        def eval_basis(x):
            return (1 / i_plus_1) * pow(x, i_plus_1)

        c = eval_basis(xmax) - eval_basis(xmin)

        # Compute the error on the integral
        err2 = 0.0

        n_par = self._degree + 1

        for i in range(n_par):
            s = 0.0
            for j in range(n_par):
                s += self._cov_matrix[i, j] * c[j]

            err2 += c[i] * s

        return math.sqrt(err2)




##
## The log likelihoods for binned and unbinned fits
##

class PolyLogLikelihood(object):
    """
    Implements a Poisson likelihood (i.e., the Cash statistic). Mind that this is not
    the Castor statistic (Cstat). The difference between the two is a constant given
    a dataset. I kept Cash instead of Castor to make easier the comparison with ROOT
    during tests, since ROOT implements the Cash statistic.
    """

    def __init__(self, x, y, model, exposure):
        self._bin_centers = x
        self._counts = y
        self._model = model
        self._parameters = model.coefficients
        self._exposure = exposure

    def _evaluate_logM(self, M):
        # Evaluate the logarithm with protection for negative or small
        # numbers, using a smooth linear extrapolation (better than just a sharp
        # cutoff)
        tiny = np.float64(np.finfo(M[0]).tiny)

        non_tiny_mask = (M > 2.0 * tiny)

        tink_mask = np.logical_not(non_tiny_mask)

        if (len(tink_mask.nonzero()[0]) > 0):
            logM = np.zeros(len(M))
            logM[tink_mask] = np.abs(M[tink_mask]) / tiny + np.log(tiny) - 1
            logM[non_tiny_mask] = np.log(M[non_tiny_mask])

        else:

            logM = np.log(M)

        return logM

    pass

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

        negative_mask = (M < 0)
        if (len(negative_mask.nonzero()[0]) > 0):
            M[negative_mask] = 0.0

        # Poisson loglikelihood statistic (Cash) is:
        # L = Sum ( M_i - D_i * log(M_i))

        logM = self._evaluate_logM(M)

        # Evaluate v_i = D_i * log(M_i): if D_i = 0 then the product is zero
        # whatever value has log(M_i). Thus, initialize the whole vector v = {v_i}
        # to zero, then overwrite the elements corresponding to D_i > 0

        d_times_logM = np.zeros(len(self._counts))

        non_zero_mask = (self._counts > 0)

        d_times_logM[non_zero_mask] = self._counts[non_zero_mask] * logM[non_zero_mask]

        log_likelihood = np.sum(M_fixed - d_times_logM)

        return log_likelihood

    def _fix_precision(self, v):
        """
          Round extremely small number inside v to the smallest usable
          number of the type corresponding to v. This is to avoid warnings
          and errors like underflows or overflows in math operations.
        """
        tiny = np.float64(np.finfo(v[0]).tiny)
        zero_mask = (np.abs(v) <= tiny)
        if (len(zero_mask.nonzero()[0]) > 0):
            v[zero_mask] = np.sign(v[zero_mask]) * tiny

        return v, tiny

    pass

    def get_free_derivs(self, parameters=None):
        """
        Return the gradient of the logLikelihood for a given set of parameters (or the current
        defined one, if parameters=None)
        """
        # The derivative of the logLikelihood statistic respect to parameter p is:
        # dC / dp = Sum [ (dM/dp)_i - D_i/M_i (dM/dp)_i]

        # Get the number of parameters and initialize the gradient to 0
        n_free = self._model.get_number_free_parameters()

        derivatives = np.zeros(n_free)

        # Set the parameters, if a new set has been provided

        if parameters is not None:
            self._model.coefficients = parameters

        # Get the gradient of the model respect to the parameters

        model_derivatives = self._model.get_free_derivs(self._bin_centers) * self._exposure

        # Get the model

        M = self._model(self._bin_centers) * self._exposure

        M, tiny_M = self._fix_precision(M)

        # Compute y_divided_M = y/M: inizialize y_divided_M to zero
        # and then overwrite the elements for which y > 0. This is to avoid
        # possible underflow and overflow due to the finite precision of the
        # computer

        y_divided_M = np.zeros(len(self._counts))

        non_zero = (self._counts > 0)
        y_divided_M[non_zero] = self._counts[non_zero] / M[non_zero]

        for p in range(n_free):
            this_model_derivatives, tinyMd = self._fix_precision(model_derivatives[p])
            derivatives[p] = np.sum(this_model_derivatives * (1.0 - y_divided_M))

        return derivatives

    pass


class PolyUnbinnedLogLikelihood(object):
    """
    Implements a Poisson likelihood (i.e., the Cash statistic). Mind that this is not
    the Castor statistic (Cstat). The difference between the two is a constant given
    a dataset. I kept Cash instead of Castor to make easier the comparison with ROOT
    during tests, since ROOT implements the Cash statistic.
    """

    def __init__(self, events, model, t_start, t_stop, exposure):
        self._events = events

        self._model = model
        self._parameters = model.coefficients
        self._exposure = exposure
        self._t_start = t_start  # list of starts
        self._t_stop = t_stop

    def _evaluate_logM(self, M):
        # Evaluate the logarithm with protection for negative or small
        # numbers, using a smooth linear extrapolation (better than just a sharp
        # cutoff)
        tiny = np.float64(np.finfo(M[0]).tiny)

        nontinyMask = (M > 2.0 * tiny)
        tinyMask = np.logical_not(nontinyMask)

        if (len(tinyMask.nonzero()[0]) > 0):
            logM = np.zeros(len(M))
            logM[tinyMask] = np.abs(M[tinyMask]) / tiny + np.log(tiny) - 1
            logM[nontinyMask] = np.log(M[nontinyMask])
        else:
            logM = np.log(M)
        return logM

    def __call__(self, parameters):
        """
          Evaluate the unbinned Poisson log likelihood

        Args:
            parameters:

        Returns:

        """

        # Compute the values for the model given this set of parameters
        self._model.coefficients = parameters

        # Integrate the polynomial (or in the future, model) over the given interval

        n_expected_counts = 0.

        for start, stop in zip(self._t_start, self._t_stop):
            n_expected_counts += self._model.integral(start, stop)

        # Now evaluate the model at the event times and multiply by the exposure

        M = self._model(self._events) * self._exposure

        # Replace negative values for the model (impossible in the Poisson context)
        # with zero
        negative_mask = (M < 0)

        if (len(negative_mask.nonzero()[0]) > 0):
            M[negative_mask] = 0.0

        # Poisson loglikelihood statistic  is:
        # logL = -Nexp + Sum ( log M_i )

        logM = self._evaluate_logM(M)

        log_likelihood = -n_expected_counts + logM.sum()

        return -log_likelihood

    def _fix_precision(self, v):
        """
          Round extremely small number inside v to the smallest usable
          number of the type corresponding to v. This is to avoid warnings
          and errors like underflows or overflows in math operations.
        """
        tiny = np.float64(np.finfo(v[0]).tiny)
        zero_mask = (np.abs(v) <= tiny)
        if (len(zero_mask.nonzero()[0]) > 0):
            v[zero_mask] = np.sign(v[zero_mask]) * tiny

        return v, tiny

    def get_free_derivs(self, parameters=None):
        """
        Return the gradient of the logLikelihood for a given set of parameters (or the current
        defined one, if parameters=None)
        """
        # The derivative of the unbinned logLikelihood statistic respect to parameter p is:
        # d logL / d theta_j = -(1/j+1) (t_f^(j+1) - t_0^(j+1)) + Sum( P(t_i, theta_k)^(-1) *t_i^j  )


        # Set the parameters, if a new set has been provided
        if (parameters is not None):
            self._model.coefficients = parameters
        pass

        M = self._model(self._events)  # * self._exposure

        M, tiny_m = self._fix_precision(M)

        degrees = np.arange(self._model.degree + 1)

        def derivative_per_degree(degree):

            d_1 = degree + 1

            pre_factor = 0
            for start, stop in zip(self._t_start, self._t_stop):
                pre_factor += (stop ** d_1 - start ** d_1)

            raised_events, _ = self._fix_precision(np.power(self._events, degree))

            return -(raised_events / M).sum() + pre_factor / float(d_1)

        derivs = np.array([derivative_per_degree(degree) for degree in degrees])

        return derivs


def polyfit(x, y, grade, exposure):
    """ funtion to fit a polynomial to event data. not a member to allow parallel computation """
    test = False

    # Check that we have enough counts to perform the fit, otherwise
    # return a "zero polynomial"
    non_zero_mask = (y > 0)
    n_non_zero = len(non_zero_mask.nonzero()[0])
    if n_non_zero == 0:
        # No data, nothing to do!
        return Polynomial([0.0]), 0.0

    # Compute an initial guess for the polynomial parameters,
    # with a least-square fit (with weight=1) using SVD (extremely robust):
    # (note that polyfit returns the coefficient starting from the maximum grade,
    # thus we need to reverse the order)
    if test:
        print("  Initial estimate with SVD..."),
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        initial_guess = np.polyfit(x, y, grade)

    initial_guess = initial_guess[::-1]

    if (test):
        print("  done -> %s" % (initial_guess))

    polynomial = Polynomial(initial_guess)

    # Check that the solution found is meaningful (i.e., definite positive
    # in the interval of interest)
    M = polynomial(x)

    negative_mask = (M < 0)

    if len(negative_mask.nonzero()[0]) > 0:
        # Least square fit failed to converge to a meaningful solution
        # Reset the initialGuess to reasonable value
        initial_guess[0] = np.mean(y)
        meanx = np.mean(x)
        initial_guess = map(lambda x: abs(x[1]) / pow(meanx, x[0]), enumerate(initial_guess))

    # Improve the solution using a logLikelihood statistic (Cash statistic)
    log_likelihood = PolyLogLikelihood(x, y, polynomial, exposure)

    # Check that we have enough non-empty bins to fit this grade of polynomial,
    # otherwise lower the grade
    dof = n_non_zero - (grade + 1)

    if dof <= 2:
        # Fit is poorly or ill-conditioned, have to reduce the number of parameters
        while (dof < 2 and len(initial_guess) > 1):
            initial_guess = initial_guess[:-1]
            polynomial = Polynomial(initial_guess)
            log_likelihood = PolyLogLikelihood(x, y, polynomial, exposure)

    # Try to improve the fit with the log-likelihood



    final_estimate = \
    opt.minimize(log_likelihood, initial_guess, method=threeML_config['event list']['binned fit method'],
                 options=threeML_config['event list']['binned fit options'])['x']
    final_estimate = np.atleast_1d(final_estimate)

    # Get the value for cstat at the minimum

    min_log_likelihood = log_likelihood(final_estimate)

    # Update the polynomial with the fitted parameters,
    # and the relative covariance matrix

    final_polynomial = Polynomial(final_estimate)

    try:
        final_polynomial.compute_covariance_matrix(log_likelihood.get_free_derivs)
    except Exception:
        raise
    # if test is defined, compare the results with those obtained with ROOT


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

        log_likelihood = PolyUnbinnedLogLikelihood(events,
                                                   polynomial,
                                                   t_start,
                                                   t_stop,
                                                   exposure)

        like_grid = []
        for amp in search_grid:

            initial_guess[0] = amp
            like_grid.append(log_likelihood(initial_guess))

        initial_guess[0] = search_grid[np.argmin(like_grid)]

        # Improve the solution
        dof = len(events) - (grade + 1)

        if dof <= 2:
            # Fit is poorly or ill-conditioned, have to reduce the number of parameters
            while (dof < 1 and len(initial_guess) > 1):
                initial_guess = initial_guess[:-1]
                polynomial = Polynomial(initial_guess)
                log_likelihood = PolyUnbinnedLogLikelihood(events,
                                                           polynomial,
                                                           t_start,
                                                           t_stop,
                                                           exposure)

        final_estimate = \
        opt.minimize(log_likelihood, initial_guess, method=threeML_config['event list']['unbinned fit method'],
                     options=threeML_config['event list']['unbinned fit options'])['x']

        final_estimate = np.atleast_1d(final_estimate)
        # print
        # print final_estimate

        # Get the value for cstat at the minimum

        min_log_likelihood = log_likelihood(final_estimate)

    # Update the polynomial with the fitted parameters,
    # and the relative covariance matrix

    final_polynomial = Polynomial(final_estimate)

    try:
        final_polynomial.compute_covariance_matrix(log_likelihood.get_free_derivs)
    except Exception:
        raise
    # if test is defined, compare the results with those obtained with ROOT


    return final_polynomial, min_log_likelihood
