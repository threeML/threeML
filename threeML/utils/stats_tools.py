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


def waic():
    pass


def dic():
    mean_deviance = -2. * np.mean(log_prob_model)

    deviance_at_mean
