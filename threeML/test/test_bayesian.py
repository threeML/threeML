from threeML import BayesianAnalysis, Uniform_prior, Log_uniform_prior
import numpy as np
import pytest

_


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



def test_bayes_constructor(bayes_fitter):

    assert bayes_fitter.sampler is None
    assert bayes_fitter.results is None
    assert bayes_fitter.samples is None
    assert bayes_fitter.log_like_values is None
    assert bayes_fitter.log_probability_values is None
    
def test_emcee(bayes_fitter):

    # This has been already tested in the fixtures (see conftest.py)

    


def test_multinest(bayes_fitter, completed_bn090217206_bayesian_analysis):

    bayes, _ = completed_bn090217206_bayesian_analysis

    bayes.set_sampler('multinest')
    
    bayes.sampler.setup(n_live_points=400)

    bayes.sample()
    
    res = bayes.results.get_data_frame()

    check_results(res)

def test_ultranest(bayes_fitter, completed_bn090217206_bayesian_analysis):

    bayes, _ = completed_bn090217206_bayesian_analysis

    bayes.set_sampler('ultranest')
    
    bayes.sampler.setup()

    bayes.sample()
    
    res = bayes.results.get_data_frame()

    check_results(res)



    
def test_zeus(bayes_fitter, completed_bn090217206_bayesian_analysis):

    bayes, _ = completed_bn090217206_bayesian_analysis

    bayes.set_sampler('zeus')
    
    bayes.sampler.setup(n_interations=200,n_walkers=20)

    bayes.sample()
    
    res = bayes.results.get_data_frame()

    bayes.restore_median_fit()
    
    check_results(res)


def test_bayes_plots(completed_bn090217206_bayesian_analysis):

    bayes, samples = completed_bn090217206_bayesian_analysis



    with pytest.raises(AssertionError):
        bayes.convergence_plots(n_samples_in_each_subset=100,n_subsets=2000)

    bayes.convergence_plots(n_samples_in_each_subset=10, n_subsets=5)

    bayes.plot_chains()

    bayes.restore_median_fit()
