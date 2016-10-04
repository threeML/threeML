# Author: J. Michael Burgess

# Provides some universal statistical utilities and stats comparison tools

import numpy as np
import pandas as pd
from threeML.io.rich_display import display
import copy

import warnings


def li_and_ma(total, background, alpha=1.):
    """"Li and Ma (1983) signal to noise significance"""

    a = total / (total + background)
    b = background / (total + background)
    S = np.sqrt(2) * np.sqrt((total * np.log(((1. + alpha) / alpha) * a) + background * np.log((1 + alpha) * b)))

    return S


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

    deviance_at_mean = -2. * bayesian_trace.log_probability(mean_of_free_parameters)[0]

    return 2 * mean_deviance - deviance_at_mean


class ModelComparison(object):
    def __init__(self, *analyses):
        self._analysis_container = analyses

        # First make sure that it is all bayesian or all MLE
        if (np.unique([a._analysis_type for a in analyses])).shape[0] > 1:
            raise RuntimeError("Only all Bayesian or all MLE analyses are allowed. Not a mixture!")

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
                stat_table['log10 (Z)'].append(analysis.log10_evidence)
                # stat_table['WAIC'].append(this_waic)
                stat_table['-2 ln(like)'].append(None)
                stat_table['N. Free Parameters'].append(n_free_params)
                # stat_table['Eff. N. Free Parameters'].append(eff_n_params)
                stat_table['dof'].append(dof)
                stat_table['Model'].append(model_name)

            else:

                this_dic = dic(analysis)
                # this_waic = waic(analysis)

                # We will now compute the AIC/BIC/ etc. at the max of the posterior likelihood

                loglike = analysis.log_like_values.max()

                this_aic = aic(loglike, n_free_params, n_data_points)
                this_bic = bic(loglike, n_free_params, n_data_points)

                # Create the dataframe

                stat_table['AIC'].append(this_aic)
                stat_table['BIC'].append(this_bic)
                stat_table['DIC'].append(this_dic)
                stat_table['log10 (Z)'].append(analysis.log10_evidence)
                # stat_table['WAIC'].append(this_waic)
                stat_table['-2 ln(like)'].append(-2. * loglike)
                stat_table['N. Free Parameters'].append(n_free_params)
                # stat_table['Eff. N. Free Parameters'].append(eff_n_params)
                stat_table['dof'].append(dof)
                stat_table['Model'].append(model_name)

        stat_df = pd.DataFrame(stat_table)

        return stat_df
