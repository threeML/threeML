from threeML import BayesianAnalysis, Uniform_prior, Log_uniform_prior
import numpy as np
import pytest

try:
    import ultranest
except:
    has_ultranest = False
else:
    has_ultranest = True
skip_if_ultranest_is_not_available = pytest.mark.skipif(
    not has_ultranest, reason="No ultranest available"
)

try:
    import dynesty
except:
    has_dynesty = False
else:
    has_dynesty = True
skip_if_dynesty_is_not_available = pytest.mark.skipif(
    not has_dynesty, reason="No dynesty available"
)


try:
    import pymultinest
except:
    has_pymultinest = False
else:
    has_pymultinest = True
skip_if_pymultinest_is_not_available = pytest.mark.skipif(
    not has_pymultinest, reason="No pymultinest available"
)

try:
    import zeus
except:
    has_zeus = False
else:
    has_zeus = True
skip_if_zeus_is_not_available = pytest.mark.skipif(
    not has_zeus, reason="No zeus available"
)


def remove_priors(model):

    for parameter in model:

        parameter.prior = None


def set_priors(model):

    powerlaw = model.bn090217206.spectrum.main.Powerlaw

    powerlaw.index.prior = Uniform_prior(lower_bound=-5.0, upper_bound=5.0)
    powerlaw.K.prior = Log_uniform_prior(lower_bound=1.0, upper_bound=10)


def check_results(fit_results):

    expected_results = [2.531028, -1.1831566000728451]

    assert np.isclose(
        fit_results["value"]["bn090217206.spectrum.main.Powerlaw.K"],
        expected_results[0],
        rtol=0.1,
    )

    assert np.isclose(
        fit_results["value"]["bn090217206.spectrum.main.Powerlaw.index"],
        expected_results[1],
        rtol=0.1,
    )


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

    bayes.set_sampler("emcee")

    assert bayes.results is None
    assert bayes.samples is None
    assert bayes.log_like_values is None
    assert bayes.log_probability_values is None


def test_emcee(bayes_fitter):

    pass
    # This has been already tested in the fixtures (see conftest.py)


@skip_if_pymultinest_is_not_available
def test_multinest(bayes_fitter, completed_bn090217206_bayesian_analysis):

    bayes, _ = completed_bn090217206_bayesian_analysis

    bayes.set_sampler("multinest")

    bayes.sampler.setup(n_live_points=400)

    bayes.sample()

    res = bayes.results.get_data_frame()

    check_results(res)


@skip_if_ultranest_is_not_available
def test_ultranest(bayes_fitter, completed_bn090217206_bayesian_analysis):

    bayes, _ = completed_bn090217206_bayesian_analysis

    bayes.set_sampler("ultranest")

    bayes.sampler.setup()

    bayes.sample()

    res = bayes.results.get_data_frame()

    check_results(res)


@skip_if_dynesty_is_not_available
def test_dynesty_nested(bayes_fitter, completed_bn090217206_bayesian_analysis):

    bayes, _ = completed_bn090217206_bayesian_analysis

    bayes.set_sampler("dynesty_nested")

    bayes.sampler.setup(n_live_points=100, n_effective=10)

    bayes.sample()

    res = bayes.results.get_data_frame()

    check_results(res)




@skip_if_dynesty_is_not_available
def test_dynesty_dynamic(bayes_fitter, completed_bn090217206_bayesian_analysis):

    bayes, _ = completed_bn090217206_bayesian_analysis

    bayes.set_sampler("dynesty_dynamic")

    bayes.sampler.setup(nlive_init=100, maxbatch=2, n_effective=10)

    bayes.sample()

    res = bayes.results.get_data_frame()

    check_results(res)


    

@skip_if_zeus_is_not_available
def test_zeus(bayes_fitter, completed_bn090217206_bayesian_analysis):

    bayes, _ = completed_bn090217206_bayesian_analysis

    bayes.set_sampler("zeus")

    bayes.sampler.setup(n_iterations=200, n_walkers=20)

    bayes.sample()

    res = bayes.results.get_data_frame()

    bayes.restore_median_fit()

    check_results(res)


def test_bayes_plots(completed_bn090217206_bayesian_analysis):

    bayes, samples = completed_bn090217206_bayesian_analysis

    with pytest.raises(AssertionError):
        bayes.convergence_plots(n_samples_in_each_subset=100, n_subsets=2000)

    bayes.convergence_plots(n_samples_in_each_subset=10, n_subsets=5)

    bayes.plot_chains()

    bayes.restore_median_fit()

def test_bayes_shared(fitted_joint_likelihood_bn090217206_nai6_nai9_bgo1):

    jl, _, _ = fitted_joint_likelihood_bn090217206_nai6_nai9_bgo1

    jl.restore_best_fit()

    model = jl.likelihood_model
    data_list = jl.data_list
    powerlaw = jl.likelihood_model.bn090217206.spectrum.main.Powerlaw

    powerlaw.index.prior = Uniform_prior(lower_bound=-5.0, upper_bound=5.0)
    powerlaw.K.prior = Log_uniform_prior(lower_bound=1.0, upper_bound=10)

    bayes = BayesianAnalysis(model, data_list)

    bayes.set_sampler("emcee", share_spectrum=True)
    bayes.sampler.setup(n_walkers=50, n_burn_in=50, n_iterations=100, seed=1234)
    samples = bayes.sample()

    res_shared = bayes.results.get_data_frame()

    bayes = BayesianAnalysis(model, data_list)

    bayes.set_sampler("emcee", share_spectrum=False)
    bayes.sampler.setup(n_walkers=50, n_burn_in=50, n_iterations=100, seed=1234)
    samples = bayes.sample()

    res_not_shared = bayes.results.get_data_frame()
    
    assert np.isclose(
        res_shared["value"]["bn090217206.spectrum.main.Powerlaw.K"],
        res_not_shared["value"]["bn090217206.spectrum.main.Powerlaw.K"],
        rtol=0.1,
    )

    assert np.isclose(
        res_shared["value"]["bn090217206.spectrum.main.Powerlaw.index"],
        res_not_shared["value"]["bn090217206.spectrum.main.Powerlaw.index"],
        rtol=0.1,
    )
