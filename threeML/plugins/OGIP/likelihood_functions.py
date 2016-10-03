import numpy as np
from threeML.plugins.gammaln import logfactorial
from math import log


def regularized_log(vector):
    """
    A function which is log(vector) where vector > 0, and zero otherwise.

    :param vector:
    :return:
    """

    return np.where(vector > 0, np.log(vector), 0)


def poisson_log_likelihood_ideal_bkg(observed_counts, expected_bkg_counts, expected_model_counts):
    """
    Poisson log-likelihood for the case where the background has no uncertainties:

    L = \sum_{i=0}^{N}~o_i~\log{(m_i + b_i)} - (m_i + b_i) - \log{o_i!}

    :param observed_counts:
    :param expected_bkg_counts:
    :param expected_model_counts:
    :return:
    """

    # Model predicted counts
    # In this likelihood the background becomes part of the model, which means that
    # the uncertainty in the background is completely neglected

    predicted_counts = expected_bkg_counts + expected_model_counts

    log_likes = observed_counts * np.log(predicted_counts + 1e-100) - predicted_counts - \
                logfactorial(observed_counts)

    return np.sum(log_likes)


def poisson_observed_poisson_background(observed_counts, background_counts, exposure_ratio, expected_model_counts):
    """
    Profile log-likelihood for the case when the observed counts are Poisson distributed, and the background counts
    are Poisson distributed as well (typical for X-ray analysis with aperture photometry). This has been derived
    by Keith Arnaud (see the Xspec manual, Wstat statistic)
    """

    # We follow Arnaud et al. (Xspec manual) in the computation, which means that at the end we need to multiply by
    # (-1) as he computes the -log(L), while we need log(L). Also, he multiplies -log(L) by 2 at the end to make it
    # converge to chisq^2. We don't do that to keep it a proper (profile) likelihood.

    # Compute the nuisance background parameter

    first_term = exposure_ratio * (observed_counts + background_counts) - (1 + exposure_ratio) * expected_model_counts
    second_term = np.sqrt(first_term ** 2 + 4 * exposure_ratio * (exposure_ratio + 1)
                          * background_counts * expected_model_counts)

    background_nuisance_parameter = (first_term + second_term) / (2 * exposure_ratio * (exposure_ratio + 1))

    first_term = expected_model_counts + (1 + exposure_ratio) * background_nuisance_parameter

    # we regularize the log so it will not give NaN if expected_model_counts and background_nuisance_parameter are both
    # zero. For any good model this should also mean observed_counts = 0, btw.

    second_term = - observed_counts * np.log(expected_model_counts + exposure_ratio * background_nuisance_parameter +
                                             1e-100)

    third_term = - background_counts * regularized_log(background_nuisance_parameter)

    ppstat = 2 * (first_term + second_term + third_term)

    ppstat += 2 * (- observed_counts * (1 - regularized_log(observed_counts))
                   - background_counts * (1 - regularized_log(background_counts)))

    assert np.isfinite(ppstat).all()

    return np.sum(ppstat) * (-1)


def poisson_observed_gaussian_background(observed_counts, background_counts, background_error, expected_model_counts):

    # This loglike assume Gaussian errors on the background and Poisson uncertainties on the
    # observed counts. It is a profile likelihood.

    MB = background_counts + expected_model_counts
    s2 = background_error ** 2

    b = 0.5 * (np.sqrt(MB ** 2 - 2 * s2 * (MB - 2 * observed_counts) + background_error ** 4)
               + background_counts - expected_model_counts - s2)

    # Now there are two branches: when the background is 0 we are in the normal situation of a pure
    # Poisson likelihood, while when the background is not zero we use the profile likelihood

    # NOTE: bkgErr can be 0 only when also bkgCounts = 0
    # Also it is evident from the expression above that when bkgCounts = 0 and bkgErr=0 also b=0

    # Let's do the branch with background > 0 first

    idx = background_counts > 0

    log_likes = np.empty_like(expected_model_counts)

    log_likes[idx] = (-(b[idx] - background_counts[idx]) ** 2 / (2 * s2[idx])
                      + observed_counts[idx] * np.log(b[idx] + expected_model_counts[idx])
                      - b[idx] - expected_model_counts[idx] - logfactorial(observed_counts[idx])
                      - 0.5 * log(2 * np.pi) - np.log(background_error[idx]))

    # Let's do the other branch

    nidx = ~idx

    # the 1e-100 in the log is to avoid zero divisions
    # This is the Poisson likelihood with no background
    log_likes[nidx] = observed_counts[nidx] * np.log(expected_model_counts[nidx] + 1e-100) - \
                      expected_model_counts[nidx] - logfactorial(observed_counts[nidx])

    return np.sum(log_likes)


# For some statistical functions (WAIC) it is desirable to look at the likelihood of each individual data point.
# These functions return the vector of the observation for that purpose

def poisson_log_likelihood_ideal_bkg_vector(observed_counts, expected_bkg_counts, expected_model_counts):
    """
    Poisson log-likelihood for the case where the background has no uncertainties:

    L = \sum_{i=0}^{N}~o_i~\log{(m_i + b_i)} - (m_i + b_i) - \log{o_i!}

    :param observed_counts:
    :param expected_bkg_counts:
    :param expected_model_counts:
    :return:
    """

    # Model predicted counts
    # In this likelihood the background becomes part of the model, which means that
    # the uncertainty in the background is completely neglected

    predicted_counts = expected_bkg_counts + expected_model_counts

    log_likes = observed_counts * np.log(predicted_counts + 1e-100) - predicted_counts - \
                logfactorial(observed_counts)

    return log_likes


def poisson_observed_poisson_background_vector(observed_counts, background_counts, exposure_ratio,
                                               expected_model_counts):
    """
    Profile log-likelihood for the case when the observed counts are Poisson distributed, and the background counts
    are Poisson distributed as well (typical for X-ray analysis with aperture photometry). This has been derived
    by Keith Arnaud (see the Xspec manual, Wstat statistic)
    """

    # We follow Arnaud et al. (Xspec manual) in the computation, which means that at the end we need to multiply by
    # (-1) as he computes the -log(L), while we need log(L). Also, he multiplies -log(L) by 2 at the end to make it
    # converge to chisq^2. We don't do that to keep it a proper (profile) likelihood.

    # Compute the nuisance background parameter

    first_term = exposure_ratio * (observed_counts + background_counts) - (1 + exposure_ratio) * expected_model_counts
    second_term = np.sqrt(first_term ** 2 + 4 * exposure_ratio * (exposure_ratio + 1)
                          * background_counts * expected_model_counts)

    background_nuisance_parameter = (first_term + second_term) / (2 * exposure_ratio * (exposure_ratio + 1))

    first_term = expected_model_counts + (1 + exposure_ratio) * background_nuisance_parameter

    # we regularize the log so it will not give NaN if expected_model_counts and background_nuisance_parameter are both
    # zero. For any good model this should also mean observed_counts = 0, btw.

    second_term = - observed_counts * np.log(expected_model_counts + exposure_ratio * background_nuisance_parameter +
                                             1e-100)

    third_term = - background_counts * regularized_log(background_nuisance_parameter)

    ppstat = 2 * (first_term + second_term + third_term)

    ppstat += 2 * (- observed_counts * (1 - regularized_log(observed_counts))
                   - background_counts * (1 - regularized_log(background_counts)))

    assert np.isfinite(ppstat).all()

    return ppstat * (-1)


def poisson_observed_gaussian_background_vector(observed_counts, background_counts, background_error,
                                                expected_model_counts):
    # This loglike assume Gaussian errors on the background and Poisson uncertainties on the
    # observed counts. It is a profile likelihood.

    MB = background_counts + expected_model_counts
    s2 = background_error ** 2

    b = 0.5 * (np.sqrt(MB ** 2 - 2 * s2 * (MB - 2 * observed_counts) + background_error ** 4)
               + background_counts - expected_model_counts - s2)

    # Now there are two branches: when the background is 0 we are in the normal situation of a pure
    # Poisson likelihood, while when the background is not zero we use the profile likelihood

    # NOTE: bkgErr can be 0 only when also bkgCounts = 0
    # Also it is evident from the expression above that when bkgCounts = 0 and bkgErr=0 also b=0

    # Let's do the branch with background > 0 first

    idx = background_counts > 0

    log_likes = np.empty_like(expected_model_counts)

    log_likes[idx] = (-(b[idx] - background_counts[idx]) ** 2 / (2 * s2[idx])
                      + observed_counts[idx] * np.log(b[idx] + expected_model_counts[idx])
                      - b[idx] - expected_model_counts[idx] - logfactorial(observed_counts[idx])
                      - 0.5 * log(2 * np.pi) - np.log(background_error[idx]))

    # Let's do the other branch

    nidx = ~idx

    # the 1e-100 in the log is to avoid zero divisions
    # This is the Poisson likelihood with no background
    log_likes[nidx] = observed_counts[nidx] * np.log(expected_model_counts[nidx] + 1e-100) - \
                      expected_model_counts[nidx] - logfactorial(observed_counts[nidx])

    return log_likes
