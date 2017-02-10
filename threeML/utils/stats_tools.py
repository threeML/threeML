# Provides some universal statistical utilities and stats comparison tools

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

    val = -2. * log_like + 2 * n_parameters
    val += 2 * n_parameters * (n_parameters + 1) / (n_data_points - n_parameters - 1)

    return val


def bic(log_like, n_parameters, n_data_points):
    """
    The Bayesian information criterion.
    """
    val = -2. * log_like + n_parameters * np.log(n_data_points)

    return val


def waic(bayesian_trace):
    raise NotImplementedError("Coming soon to a theater near you.")


def effective_number_of_parameters(bayesian_trace):
    raise NotImplementedError("Coming soon to a theater near you.")


def dic(bayesian_trace):
    """
    The Deviance information criteria derived from MCMC traces
    Read more:  dx.doi.org/10.1111/1467-9868.00353


    :param bayesian_trace: an instance of Bayesian Analysis

    :return: deviance information criteria

    """

    mean_deviance = -2. * np.mean(bayesian_trace.log_probability_values)

    mean_of_free_parameters = np.mean(bayesian_trace.raw_samples, axis=0)

    deviance_at_mean = -2. * bayesian_trace.get_posterior(mean_of_free_parameters)

    return 2 * mean_deviance - deviance_at_mean


def sqrt_sum_of_squares(arg):
    """
    :param arg: and array of number to be squared and summed
    :return: the sqrt of the sum of the squares
    """

    return np.sqrt( np.square(arg).sum() )



class ModelComparison(object):
    def __init__(self, *analyses):
        self._analysis_container = analyses

        # First make sure that it is all bayesian or all MLE
        assert (np.unique([a.analysis_type for a in analyses])).shape[
                   0] == 1, "Only all Bayesian or all MLE analyses are allowed. Not a mixture!"


        self._analysis_type = analyses[0].analysis_type

        if self._analysis_type == 'mle':

            self._stat_df = self._compute_mle_statistics()



        elif self._analysis_type == 'bayesian':

            self._stat_df = self._compute_bayes_statistics()

    def report(self, sort=None, normalized=True, precision=1):
        """
        Create a statistical report for multiple fits
        :param sort: (optional) which statistic to sort (str)
        :param normalized: (optional) normalize the stats to the best fit
        :param precision: (optional) precision of the table
        :return: stats DataFrame
        """

        pd.options.display.float_format = ('{:.%df}' % (precision)).format

        this_df = self._stat_df.copy()

        if self._analysis_type == 'bayesian':

            # Remember to add WAIC in later

            display_order = ['Model',
                             '-2 ln(like)',
                             'AIC',
                             'BIC',
                             'DIC',
                             'log10 (Z)',
                             'N. Free Parameters',
                             #                  'Eff. N. Free Parameters',
                             'dof']

            min_stat = ['-2 ln(like)',
                        'AIC',
                        'BIC',
                        'DIC'
                        ]

            max_stat = ['log10 (Z)']


        else:

            display_order = ['Model',
                             '-2 ln(like)',
                             'AIC',
                             'BIC',
                             'N. Free Parameters',
                             'dof']

            min_stat = ['-2 ln(like)',
                        'AIC',
                        'BIC'
                        ]

            max_stat = []

        if normalized:

            # Normalize the statistics to the 'worst' fit.
            # MLE type stats have lowest value as best
            # while Bayesian evidence has highest as best

            for key in min_stat:
                this_df[key] = this_df[key] - this_df[key].max()

            for key in max_stat:
                this_df[key] = this_df[key] - this_df[key].max()

        if sort is not None:

            if sort in min_stat:

                ascend = True

            elif sort in max_stat:

                ascend = False

            else:

                warnings.warn('%s is not a valid statistic' % sort)

                display(this_df[display_order])

                return self._stat_df

            this_df = this_df.sort_values(by=sort, ascending=ascend, inplace=False)

            # for col in format_columns:

            #    self._float_format(this_df[col])


            display(this_df[display_order])

            return self._stat_df

        else:

            #   for col in format_columns:
            #      self._float_format(self._stat_df[col])

            display(this_df[display_order])

            return self._stat_df

        pd.reset_option('float_format')

    @property
    def statistical_dataframe(self):

        return self._stat_df

    def _compute_mle_statistics(self):

        stat_table = {}
        stat_table['AIC'] = []
        stat_table['BIC'] = []
        stat_table['-2 ln(like)'] = []
        stat_table['dof'] = []
        stat_table['N. Free Parameters'] = []
        stat_table['Model'] = []

        for analysis in self._analysis_container:
            n_data_points = np.sum([data.n_data_points for data in analysis.data_list.values()])
            n_free_params = len(analysis._free_parameters.values())  # should add a property
            dof = n_data_points - n_free_params
            model_name = \
                [parameter_name.split('.')[-2] for (parameter_name, parameter) in
                 analysis._free_parameters.iteritems()][0]
            loglike = - analysis.current_minimum

            this_aic = aic(loglike, n_free_params, n_data_points)
            this_bic = bic(loglike, n_free_params, n_data_points)

            stat_table['AIC'].append(this_aic)
            stat_table['BIC'].append(this_bic)
            stat_table['-2 ln(like)'].append(-2. * loglike)
            stat_table['dof'].append(dof)
            stat_table['N. Free Parameters'].append(n_free_params)
            stat_table['Model'].append(model_name)

        stat_df = pd.DataFrame(stat_table)

        return stat_df

    def _compute_bayes_statistics(self):

        stat_table = {}
        stat_table['AIC'] = []
        stat_table['BIC'] = []
        stat_table['DIC'] = []
        stat_table['log10 (Z)'] = []
        # stat_table['WAIC'] = []
        stat_table['-2 ln(like)'] = []
        stat_table['dof'] = []
        stat_table['N. Free Parameters'] = []
        # stat_table['Eff. N. Free Parameters'] = []
        stat_table['Model'] = []

        for analysis in self._analysis_container:
            n_data_points = np.sum([data.n_data_points for data in analysis.data_list.values()])
            n_free_params = len(analysis._free_parameters.values())  # should add a property
            dof = n_data_points - n_free_params

            # eff_n_params = analysis.get_effective_free_parameters()  # change this to local function later

            model_name = \
                [parameter_name.split('.')[-2] for (parameter_name, parameter) in
                 analysis._free_parameters.iteritems()][0]

            if analysis.log_like_values is None:
                stat_table['AIC'].append(None)
                stat_table['BIC'].append(None)
                stat_table['DIC'].append(None)
                stat_table['log10 (Z)'].append(analysis.log_marginal_likelihood)
                # stat_table['WAIC'].append(this_waic)
                stat_table['-2 ln(like)'].append(None)
                stat_table['N. Free Parameters'].append(n_free_params)
                # stat_table['Eff. N. Free Parameters'].append(eff_n_params)
                stat_table['dof'].append(dof)
                stat_table['Model'].append(model_name)

            else:

                if analysis.log_probability_values is not None:

                    this_dic = dic(analysis)

                else:

                    this_dic = None

                # this_waic = waic(analysis)

                # We will now compute the AIC/BIC/ etc. at the max of the posterior likelihood

                loglike = analysis.log_like_values.max()

                this_aic = aic(loglike, n_free_params, n_data_points)
                this_bic = bic(loglike, n_free_params, n_data_points)

                # Create the dataframe

                stat_table['AIC'].append(this_aic)
                stat_table['BIC'].append(this_bic)
                stat_table['DIC'].append(this_dic)
                stat_table['log10 (Z)'].append(analysis.log_marginal_likelihood)
                # stat_table['WAIC'].append(this_waic)
                stat_table['-2 ln(like)'].append(-2. * loglike)
                stat_table['N. Free Parameters'].append(n_free_params)
                # stat_table['Eff. N. Free Parameters'].append(eff_n_params)
                stat_table['dof'].append(dof)
                stat_table['Model'].append(model_name)

        stat_df = pd.DataFrame(stat_table)

        return stat_df


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


    _interpolator = scipy.interpolate.InterpolatedUnivariateSpline(_logy[::-1], _x[::-1], k=1, ext=3)

    def __init__(self, Non, Noff, alpha=1.0):

        assert alpha > 0 and alpha <= 1

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

        idx = (cdf >= 2 * self._epsilon)

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

        assert alpha > 0 and alpha <= 1

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

        poisson_probability = PoissonResiduals(self.Non, self.Noff, self.alpha).significance_one_side()

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

        one[idx] = self.Non[idx] * np.log((1 + self.alpha) / self.alpha *
                                          (self.Non[idx] / (self.Non[idx] + self.Noff[idx])))

        two = np.zeros_like(self.Noff, dtype=float)

        two[idx] = self.Noff[idx] * np.log((1 + self.alpha) * (self.Noff[idx] / (self.Non[idx] + self.Noff[idx])))

        if assign_sign:

            sign = np.where(self.net > 0, 1, -1)

        else:

            sign = 1

        return sign * np.sqrt(2 * (one + two))

    def li_and_ma_equivalent_for_gaussian_background(self, sigma_b):

        # This is a computation I need to publish (G. Vianello)

        b = self.expected
        o = self.Non

        b0 = 0.5 * (np.sqrt(b ** 2 - 2 * sigma_b ** 2 * (b - 2 * o) + sigma_b ** 4) + b - sigma_b ** 2)

        S = sqrt(2) * np.sqrt(o * np.log(o / b0) + (b0 - b) ** 2 / (2 * sigma_b ** 2) + b0 - o)

        sign = np.where(o > b, 1, -1)

        return sign * S
