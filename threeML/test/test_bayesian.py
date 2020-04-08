from threeML import BayesianAnalysis, Uniform_prior, Log_uniform_prior
import numpy as np
import pytest


def remove_priors(model):

    for parameter in model:

        parameter.prior = None


def set_priors(model):

    powerlaw = model.bn090217206.spectrum.main.Powerlaw

    powerlaw.index.prior = Uniform_prior(lower_bound=-5.0, upper_bound=5.0)
    powerlaw.K.prior = Log_uniform_prior(lower_bound=1.0, upper_bound=10)


def check_results(fit_results):

    expected_results = [2.531028, -1.1831566000728451]

    assert np.isclose(fit_results['value']['bn090217206.spectrum.main.Powerlaw.K'],
                      expected_results[0], rtol=0.1)

    assert np.isclose(fit_results['value']['bn090217206.spectrum.main.Powerlaw.index'],
                      expected_results[1], rtol=0.1)



def test_bayes_constructor(fitted_joint_likelihood_bn090217206_nai):

    jl, fit_results, like_frame = fitted_joint_likelihood_bn090217206_nai
    datalist = jl.data_list
    model = jl.likelihood_model

    jl.restore_best_fit()

    # Priors might have been set by other tests, let's make sure they are
    # removed so we can test the error
    remove_priors(model)
    with pytest.raises(RuntimeError):

        _ = BayesianAnalysis(model, datalist)

    set_priors(model)

    bayes = BayesianAnalysis(model, datalist)


def test_emcee():

    # This has been already tested in the fixtures (see conftest.py)

    pass


def test_multinest(completed_bn090217206_bayesian_analysis):

    bayes, _ = completed_bn090217206_bayesian_analysis

    bayes.sample_multinest(n_live_points=400)

    res = bayes.results.get_data_frame()

    bayes.restore_median_fit()
    
    check_results(res)

# def test_parallel_temp():
#
#     powerlaw.index.prior = Uniform_prior(lower_bound=-5.0, upper_bound=5.0)
#     powerlaw.K.prior = Log_uniform_prior(lower_bound=1.0, upper_bound=10)
#
#     bayes = BayesianAnalysis(model, data_list)
#
#     # test parallel temp
#     bayes.sample_parallel_tempering(n_temps=2, n_walkers=10, burn_in=2, n_samples=500, quiet=False)
#
#

def test_bayes_plots(completed_bn090217206_bayesian_analysis):

    bayes, samples = completed_bn090217206_bayesian_analysis



    with pytest.raises(AssertionError):
        bayes.convergence_plots(n_samples_in_each_subset=100,n_subsets=2000)

    bayes.convergence_plots(n_samples_in_each_subset=10, n_subsets=5)

    bayes.plot_chains()

    bayes.restore_median_fit()
