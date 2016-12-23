import numpy as np
import scipy
import warnings
import math


class Polynomial(object):
    def __init__(self, coefficients, is_integral=False):

        """

        :param coefficients: array of poly coefficients
        :param is_integral: if this polynomial is an
        """
        self._coefficients = coefficients
        self._degree = len(coefficients) - 1

        self._i_plus_1 = np.array(range(1, self._degree + 1 + 1), 'd')

        # Build an empty covariance matrix
        self._cov_matrix = np.zeros([self._degree + 1, self._degree + 1])

        # we can fix some things for speed
        # we only need to set the coeff for the
        # integral polynomial
        if not is_integral:
            integral_coeff = [0]

            integral_coeff.extend(map(lambda i: self._coefficients[i - 1] / float(i), range(1, self._degree + 1 + 1)))

            self._integral_polynomial = Polynomial(integral_coeff, is_integral=True)






    @property
    def degree(self):
        return self._degree

    @property
    def error(self):

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

        integral_coeff.extend(map(lambda i: self._coefficients[i - 1] / float(i), range(1, self._degree + 1 + 1)))

        self._integral_polynomial = Polynomial(integral_coeff, is_integral=True)

    def ___set_coefficient(self, val):
        """ Indirect coefficient setter """

        return self.__set_coefficient(val)

    coefficients = property(___get_coefficient, ___set_coefficient,
                            doc="""Gets or sets the coefficients of the polynomial.""")

    def __call__(self, x):

        result = 0
        for coefficient in self._coefficients[::-1]:
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

        self._cov_matrix = compute_covariance_matrix(statistic_gradient, self._coefficients)

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

        return self._integral_polynomial(xmax) - self._integral_polynomial(xmin)

    def _eval_basis(self, x):

        return (1. / self._i_plus_1) * np.power(x, self._i_plus_1)


    def integral_error(self, xmin, xmax):
        # Based on http://root.cern.ch/root/html/tutorials/fit/ErrorIntegral.C.html


        c = self._eval_basis(xmax) - self._eval_basis(xmin)
        tmp = c.dot(self._cov_matrix)
        err2 = tmp.dot(c)

        return math.sqrt(err2)


def polyfit(x, y, grade, exposure):
    """
    function to fit a polynomial to event data. not a member to allow parallel computation
    """

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

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        initial_guess = np.polyfit(x, y, grade)

    initial_guess = initial_guess[::-1]

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
    if test:
        print("Effective dof: %s" % (dof))

    if dof <= 2:
        # Fit is poorly or ill-conditioned, have to reduce the number of parameters
        while (dof < 2 and len(initial_guess) > 1):
            initial_guess = initial_guess[:-1]
            polynomial = Polynomial(initial_guess)
            log_likelihood = PolyLogLikelihood(x, y, polynomial, exposure)

    # Try to improve the fit with the log-likelihood

    final_estimate = scipy.optimize.fmin(log_likelihood, initial_guess,
                                         ftol=1E-5, xtol=1E-5,
                                         maxiter=1e6, maxfun=1E6,
                                         disp=False)

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

    search_grid = np.logspace(-2, 4, 10)

    initial_guess = np.zeros(grade + 1)

    polynomial = Polynomial(initial_guess)

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



    final_estimate = scipy.optimize.fmin(log_likelihood, initial_guess,
                                         ftol=1E-5, xtol=1E-5,
                                         maxiter=1e6, maxfun=1E6,
                                         disp=False)

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


def compute_covariance_matrix(grad, par, full_output=False,
                              init_step=0.01, min_step=1e-12, max_step=1, max_iters=50,
                              target=0.1, min_func=1e-7, max_func=4):
    """Perform finite differences on the _analytic_ gradient provided by user to calculate hessian/covariance matrix.

    Positional args:
        grad                : a function to return a gradient
        par                 : vector of parameters (should be function minimum for covariance matrix calculation)

    Keyword args:

        full_output [False] : if True, return information about convergence, else just the covariance matrix
        init_step   [1e-3]  : initial step size (0.04 ~ 10% in log10 space); can be a scalar or vector
        min_step    [1e-6]  : the minimum step size to take in parameter space
        max_step    [1]     : the maximum step size to take in parameter sapce
        max_iters   [5]     : maximum number of iterations to attempt to converge on a good step size
        target      [0.5]   : the target change in the function value for step size
        min_func    [1e-4]  : the minimum allowable change in (abs) function value to accept for convergence
        max_func    [4]     : the maximum allowable change in (abs) function value to accept for convergence
    """

    nparams = len(par)
    step_size = np.ones(nparams) * init_step
    step_size = np.maximum(step_size, min_step * 1.1)
    step_size = np.minimum(step_size, max_step * 0.9)
    hess = np.zeros([nparams, nparams])
    min_flags = np.asarray([False] * nparams)
    max_flags = np.asarray([False] * nparams)

    def revised_step(delta_f, current_step, index):
        if (current_step == max_step):
            max_flags[i] = True
            return True, 0

        elif (current_step == min_step):
            min_flags[i] = True
            return True, 0

        else:
            adf = abs(delta_f)
            if adf < 1e-8:
                # need to address a step size that results in a likelihood change that's too
                # small compared to precision
                pass

            if (adf < min_func) or (adf > max_func):
                new_step = current_step / (adf / target)
                new_step = min(new_step, max_step)
                new_step = max(new_step, min_step)
                return False, new_step
            else:
                return True, 0

    iters = np.zeros(nparams)
    for i in xrange(nparams):
        converged = False

        for j in xrange(max_iters):
            iters[i] += 1

            di = step_size[i]
            par[i] += di
            g_up = grad(par)

            par[i] -= 2 * di
            g_dn = grad(par)

            par[i] += di

            delta_f = (g_up - g_dn)[i]

            converged, new_step = revised_step(delta_f, di, i)
            # print 'Parameter %d -- Iteration %d -- Step size: %.2e -- delta: %.2e'%(i,j,di,delta_f)

            if converged:
                break
            else:
                step_size[i] = new_step

        hess[i, :] = (g_up - g_dn) / (2 * di)  # central difference

        if not converged:
            print 'Warning: step size for parameter %d (%.2g) did not result in convergence.' % (i, di)
    try:
        cov = np.linalg.inv(hess)
    except:
        print 'Error inverting hessian.'
        raise Exception('Error inverting hessian')
    if full_output:
        return cov, hess, step_size, iters, min_flags, max_flags
    else:
        return cov
