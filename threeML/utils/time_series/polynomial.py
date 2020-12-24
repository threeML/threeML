from __future__ import division

import warnings
from builtins import object, range, zip

import numpy as np
import scipy.optimize as opt
from astromodels import (Constant, Cubic, Gaussian, Line, Log_normal, Model,
                         PointSource, Quadratic)
from past.utils import old_div

from threeML.bayesian.bayesian_analysis import BayesianAnalysis
from threeML.classicMLE.joint_likelihood import FitFailed, JointLikelihood
from threeML.data_list import DataList
from threeML.exceptions.custom_exceptions import custom_warnings
from threeML.plugins.UnbinnedPoissonLike import (EventObservation,
                                                 UnbinnedPoissonLike)
from threeML.plugins.XYLike import XYLike
from threeML.utils.differentiation import ParameterOnBoundary, get_hessian

_grade_model_lookup = (Constant, Line, Quadratic, Cubic, Quadratic)


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

    def set_covariace_matrix(self, matrix):

        self._cov_matrix = matrix

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


def polyfit(x, y, grade, exposure, bayes=False):
    """ function to fit a polynomial to event data. not a member to allow parallel computation """

    # Check that we have enough counts to perform the fit, otherwise
    # return a "zero polynomial"
    non_zero_mask = y > 0
    n_non_zero = non_zero_mask.sum()
    if n_non_zero == 0:
        # No data, nothing to do!
        return Polynomial([0.0]), 0.0

    # create 3ML plugins and fit them with 3ML!
    # should eventuallly allow better config

    # seelct the model based on the grade

    shape = _grade_model_lookup[grade]()

    ps = PointSource("_dummy", 0, 0, spectral_shape=shape)

    model = Model(ps)

    xy = XYLike("series", x=x, y=y, exposure=exposure, poisson_data=True, quiet=True)

    if not bayes:

        # make sure the model is positive

        for i, (k, v) in enumerate(model.free_parameters.items()):

            if i == 0:

                v.bounds = (0, None)

                v.value = 1

            else:

                v.value = 0.1

        jl: JointLikelihood = JointLikelihood(model, DataList(xy))

        jl.set_minimizer("minuit")

        # if the fit falis, retry and then just accept

        try:

            jl.fit(quiet=True)

        except:

            try:

                jl.fit(quiet=True)

            except:

                pass

        coeff = [v.value for _, v in model.free_parameters.items()]

        final_polynomial = Polynomial(coeff)

        final_polynomial.set_covariace_matrix(jl.results.covariance_matrix)

        min_log_likelihood = xy.get_log_like()

    else:

        # set smart priors

        for i, (k, v) in enumerate(model.free_parameters.items()):

            if i == 0:

                v.bounds = (0, None)
                v.prior = Log_normal(mu=np.log(1), sigma=2)
                v.value = 1

            else:

                v.prior = Gaussian(mu=0, sigma=1)
                v.value = 0.1

        ba: BayesianAnalysis = BayesianAnalysis(model, DataList(xy))

        ba.set_sampler("emcee")

        ba.sampler.setup(n_iterations=500, n_burn_in=200, n_walkers=20)

        ba.sample(quiet=True)

        ba.restore_median_fit()

        coeff = [v.value for _, v in model.free_parameters.items()]

        final_polynomial = Polynomial(coeff)

        final_polynomial.set_covariace_matrix(ba.results.estimate_covariance_matrix())

        min_log_likelihood = xy.get_log_like()

    return final_polynomial, min_log_likelihood


def unbinned_polyfit(events, grade, t_start, t_stop, exposure, bayes=False):
    """
    function to fit a polynomial to event data. not a member to allow parallel computation

    """

    # Check that we have enough counts to perform the fit, otherwise
    # return a "zero polynomial"
    
    # n_non_zero = non_zero_mask.sum()
    # if n_non_zero == 0:
    #     # No data, nothing to do!
    #     return Polynomial([0.0]), 0.0

    # create 3ML plugins and fit them with 3ML!
    # should eventuallly allow better config

    # seelct the model based on the grade

    shape = _grade_model_lookup[grade]()

    ps = PointSource("_dummy", 0, 0, spectral_shape=shape)

    model = Model(ps)

    observation = EventObservation(events, exposure, t_start, t_stop)

    xy = UnbinnedPoissonLike("series")

    if not bayes:

        # make sure the model is positive

        for i, (k, v) in enumerate(model.free_parameters.items()):

            if i == 0:

                v.bounds = (0, None)

                v.value = 1

            else:

                v.value = 0.1

        jl: JointLikelihood = JointLikelihood(model, DataList(xy))

        jl.set_minimizer("minuit")

        # if the fit falis, retry and then just accept

        try:

            jl.fit(quiet=True)

        except:

            try:

                jl.fit(quiet=True)

            except:

                pass

        coeff = [v.value for _, v in model.free_parameters.items()]

        final_polynomial = Polynomial(coeff)

        final_polynomial.set_covariace_matrix(jl.results.covariance_matrix)

        min_log_likelihood = xy.get_log_like()

    else:

        # set smart priors

        for i, (k, v) in enumerate(model.free_parameters.items()):

            if i == 0:

                v.bounds = (0, None)
                v.prior = Log_normal(mu=np.log(1), sigma=2)
                v.value = 1

            else:

                v.prior = Gaussian(mu=0, sigma=1)
                v.value = 0.1

        ba: BayesianAnalysis = BayesianAnalysis(model, DataList(xy))

        ba.set_sampler("emcee")

        ba.sampler.setup(n_iterations=500, n_burn_in=200, n_walkers=20)

        ba.sample(quiet=True)

        ba.restore_median_fit()

        coeff = [v.value for _, v in model.free_parameters.items()]

        final_polynomial = Polynomial(coeff)

        final_polynomial.set_covariace_matrix(ba.results.estimate_covariance_matrix())

        min_log_likelihood = xy.get_log_like()

    return final_polynomial, min_log_likelihood
