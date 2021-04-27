from __future__ import division

# Provides some universal statistical utilities and stats comparison tools

from past.utils import old_div
from builtins import object
from math import sqrt

import numpy as np
import pandas as pd
import scipy.interpolate
import scipy.stats
import warnings
from scipy.special import erfinv

from threeML.io.rich_display import display


def aic(log_like, n_parameters, n_data_points):
    """
    The Aikake information criterion.
    A model comparison tool based of infomormation theory. It assumes that N is large i.e.,
    that the model is approaching the CLT.


    """
    try:
        val = -2.0 * log_like + 2 * n_parameters
        val += (
            2 * n_parameters * (n_parameters + 1) / float(n_data_points - n_parameters - 1)
        )

    except:

        val = 0

        
    if not np.isfinite(val):
        val = 0

        warnings.warn("AIC was NAN. Recording zero, but you should examine your fit.")

    return val


def bic(log_like, n_parameters, n_data_points):
    """
    The Bayesian information criterion.
    """
    val = -2.0 * log_like + n_parameters * np.log(n_data_points)

    if not np.isfinite(val):
        val = 0

        warnings.warn("BIC was NAN. Recording zero, but you should examine your fit.")

    return val


def waic(bayesian_trace):
    raise NotImplementedError("Coming soon to a theater near you.")


def dic(bayes_analysis):
    """
    elpd_DIC = log p(y|mean(parameters)) - p_DIC

    the first term is the deviance at the mean of the posterior
    and p_DIC is the effective number of free parameters:

    p_DIC = 2(log p(y|mean(parameters)) - 1/N sum(log p(y|parameters_s), 1,N) )

    DIC = -2*elpd_DIC

    the effective number of free parameters can be negative if the mean is
    the mean is far from the mode

    :param bayes_analysis: a bayesian analysis object
    :return dic, effective number of free parameters:

    """

    mean_of_free_parameters = np.mean(bayes_analysis.raw_samples, axis=0)

    deviance_at_mean = bayes_analysis.get_posterior(mean_of_free_parameters)

    mean_deviance = np.mean(bayes_analysis.log_probability_values)

    pdic = 2 * (deviance_at_mean - mean_deviance)

    elpd_dic = deviance_at_mean - pdic

    if not np.isfinite(pdic) or not np.isfinite(elpd_dic):

        elpd_dic = 0
        pdic = 0

        warnings.warn("DIC was NAN. Recording zero, but you should examine your fit.")

    return -2 * elpd_dic, pdic


def sqrt_sum_of_squares(arg):
    """
    :param arg: and array of number to be squared and summed
    :return: the sqrt of the sum of the squares
    """

    return np.sqrt(np.square(arg).sum())


class PoissonResiduals(object):
    """
    This class implements a way to compute residuals for a Poisson distribution mapping them to residuals of a standard
    normal distribution. The probability of obtaining the observed counts given the expected one is computed, and then
    transformed "in unit of sigma", i.e., the sigma value corresponding to that probability is computed.

    The algorithm implemented here uses different branches so that it is fairly accurate between -36 and +36 sigma.

    NOTE: if the expected number of counts is not very high, then the Poisson distribution is skewed and so the
    probability of obtaining a downward fluctuation at a given sigma level is not the same as obtaining the same
    fluctuation in the upward direction. Therefore, the distribution of residuals is *not* expected to be symmetric
    in that case. The sigma level at which this effect is visible depends strongly on the expected number of counts.
    Under normal circumstances residuals are expected to be a few sigma at most, in which case the effect becomes
    important for expected number of counts <~ 15-20.

    """

    # Putting these here make them part of the *class*, not the instance, i.e., they are created
    # only once when the module is imported, and then are referred to by any instance of the class

    # These are lookup tables for the significance from a Poisson distribution when the
    # probability is very low so that the normal computation is not possible due to
    # the finite numerical precision of the computer

    _x = np.logspace(np.log10(5), np.log10(36), 1000)
    _logy = np.log10(scipy.stats.norm.sf(_x))

    # Make the interpolator here so we do it only once. Also use ext=3 so that the interpolation
    # will return the maximum value instead of extrapolating

    _interpolator = scipy.interpolate.InterpolatedUnivariateSpline(
        _logy[::-1], _x[::-1], k=1, ext=3
    )

    def __init__(self, Non, Noff, alpha=1.0):

        assert alpha > 0 and alpha <= 1, "alpha was %f" % alpha

        self.Non = np.array(Non, dtype=float, ndmin=1)

        self.Noff = np.array(Noff, dtype=float, ndmin=1)

        self.alpha = float(alpha)

        self.expected = self.alpha * self.Noff

        self.net = self.Non - self.expected

        # This is the minimum difference between 1 and the next representable floating point number
        self._epsilon = np.finfo(float).eps

    def significance_one_side(self):

        # For the points where Non > expected, we need to use the survival function
        # sf(x) = 1 - cdf, which can go do very low numbers
        # Instead, for points where Non < expected, we need to use the cdf which allows
        # to go to very low numbers in that directions

        idx = self.Non >= self.expected

        out = np.zeros_like(self.Non)

        if np.sum(idx) > 0:
            out[idx] = self._using_sf(self.Non[idx], self.expected[idx])

        if np.sum(~idx) > 0:
            out[~idx] = self._using_cdf(self.Non[~idx], self.expected[~idx])

        return out

    def _using_sf(self, x, exp):

        sf = scipy.stats.poisson.sf(x, exp)

        # print(sf)

        # return erfinv(2 * sf) * sqrt(2)

        return scipy.stats.norm.isf(sf)

    def _using_cdf(self, x, exp):

        # Get the value of the cumulative probability function, instead of the survival function (1 - cdf),
        # because for extreme values sf(x) = 1 - cdf(x) = 1 due to numerical precision problems

        cdf = scipy.stats.poisson.cdf(x, exp)

        # print(cdf)

        out = np.zeros_like(x)

        idx = cdf >= 2 * self._epsilon

        # We can do a direct computation, because the numerical precision is sufficient
        # for this computation, as -sf = cdf - 1 is a representable number

        out[idx] = erfinv(2 * cdf[idx] - 1) * sqrt(2)

        # We use a lookup table with interpolation because the numerical precision would not
        # be sufficient to make the computation

        out[~idx] = -1 * self._interpolator(np.log10(cdf[~idx]))

        return out


class Significance(object):
    """
    Implements equations in Li&Ma 1983

    """

    def __init__(self, Non, Noff, alpha=1):

        assert alpha > 0 and alpha <= 1, "alpha was %f" % alpha

        self.Non = np.array(Non, dtype=float, ndmin=1)

        self.Noff = np.array(Noff, dtype=float, ndmin=1)

        self.alpha = float(alpha)

        self.expected = self.alpha * self.Noff

        self.net = self.Non - self.expected

    def known_background(self):
        """
        Compute the significance under the hypothesis that there is no uncertainty in the background. In other words,
        compute the probability of obtaining the observed counts given the expected counts from the background, then
        transform it in sigma.

        NOTE: this is reliable for expected counts >~10-15 if the significance is not very high. The higher the
        expected counts, the more reliable the significance estimation. As rule of thumb, you need at least 25 counts
        to have reliable estimates up to 5 sigma.

        NOTE 2: if you use to compute residuals in units of sigma, you should not expected them to be symmetrically
        distributed around 0 unless the expected number of counts is high enough for all bins (>~15). This is due to
        the fact that the Poisson distribution is very skewed at low counts.

        :return: significance vector
        """

        # Poisson probability of obtaining Non given Noff * alpha, in sigma units

        poisson_probability = PoissonResiduals(
            self.Non, self.Noff, self.alpha
        ).significance_one_side()

        return poisson_probability

    def li_and_ma(self, assign_sign=True):
        """
        Compute the significance using the formula from Li & Ma 1983, which is appropriate when both background and
        observed signal are counts coming from a Poisson distribution.

        :param assign_sign: whether to assign a sign to the significance, according to the sign of the net counts
        Non - alpha * Noff, so that excesses will have positive significances and defects negative significances
        :return:
        """

        one = np.zeros_like(self.Non, dtype=float)

        idx = self.Non > 0

        one[idx] = self.Non[idx] * np.log(
            old_div((1 + self.alpha), self.alpha)
            * (old_div(self.Non[idx], (self.Non[idx] + self.Noff[idx])))
        )

        two = np.zeros_like(self.Noff, dtype=float)

        two[idx] = self.Noff[idx] * np.log(
            (1 + self.alpha)
            * (old_div(self.Noff[idx], (self.Non[idx] + self.Noff[idx])))
        )

        if assign_sign:

            sign = np.where(self.net > 0, 1, -1)

        else:

            sign = 1

        return sign * np.sqrt(2 * (one + two))

    def li_and_ma_equivalent_for_gaussian_background(self, sigma_b):
        """
        Compute the significance using the formula from Vianello 2018
        (https://iopscience.iop.org/article/10.3847/1538-4365/aab780/meta),
        which is appropriate when the observation is Poisson distributed but
        the background has been modeled and thus has Gaussian distributed errors.

        :param sigma_b: The gaussian 1 sigma errors on the background  
        :return:

        """

        # This is a computation I need to publish (G. Vianello)

        # Actually, you did (and beat J. Michael!) For details on this computation

        b = self.expected
        o = self.Non

        b0 = 0.5 * (
            np.sqrt(b ** 2 - 2 * sigma_b ** 2 * (b - 2 * o) + sigma_b ** 4)
            + b
            - sigma_b ** 2
        )

        S = sqrt(2) * np.sqrt(
            o * np.log(old_div(o, b0))
            + old_div((b0 - b) ** 2, (2 * sigma_b ** 2))
            + b0
            - o
        )

        sign = np.where(o > b, 1, -1)

        return sign * S
