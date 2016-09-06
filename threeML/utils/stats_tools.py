# Author: J. Michael Burgess

# Provides some universal statistical utilities and stats comparison tools

import numpy as np


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


    Returns:

    """
    val = -2. * log_like + n_parameters * np.log(n_data_points)


def waic(bayesian_trace):
    log_prob = bayesian_trace.log_probability_values

    np.sum()


def dic(bayesian_trace):
    """
    The Deviance information criteria derived from MCMC traces
    Read more:  dx.doi.org/10.1111/1467-9868.00353

    Args:
        bayesian_trace: an instance of Bayesian Analysis

    Returns:

    """

    mean_deviance = -2. * np.mean(bayesian_trace.log_probability_values)

    mean_of_free_parameters = np.mean(bayesian_trace.raw_samples, axis=0)

    deviance_at_mean = -2. * bayesian_trace.log_probability(mean_of_free_parameters)[0]

    return 2 * mean_deviance - deviance_at_mean
