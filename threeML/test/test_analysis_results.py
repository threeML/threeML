import pytest
import os
import numpy as np

from threeML.plugins.XYLike import XYLike
from threeML import Model, DataList, JointLikelihood, PointSource
from threeML import BayesianAnalysis, Uniform_prior, Log_uniform_prior
from threeML.analysis_results import MLEResults, load_analysis_results, AnalysisResultsSet
from astromodels import Line, Gaussian


_cache = {}

# These are the same simulated dataset we use in the test of the XY plugin

x = np.linspace(0, 10, 50)

poiss_sig = [44, 43, 38, 25, 51, 37, 46, 47, 55, 36, 40, 32, 46, 37, 44, 42, 50, 48, 52, 47, 39, 55, 80, 93, 123, 135,
             96, 74, 43, 49, 43, 51, 27, 32, 35, 42, 43, 49, 38, 43, 59, 54, 50, 40, 50, 57, 55, 47, 38, 64]


def _get_mle_analysis_results():

    global _cache

    if 'ar' in _cache:

        return _cache['ar']

    y = np.array(poiss_sig)

    xy = XYLike("test", x, y, poisson_data=True)

    fitfun = Line() + Gaussian()

    fitfun.a_1.bounds = (-10, 10.0)
    fitfun.b_1.bounds = (-100, 100.0)
    fitfun.F_2 = 60.0
    fitfun.F_2.bounds = (1e-3, 200.0)
    fitfun.mu_2 = 5.0
    fitfun.mu_2.bounds = (0.0, 100.0)
    fitfun.sigma_2.bounds = (1e-3, 10.0)

    model = Model(PointSource('fake',0.0, 0.0, fitfun))

    data = DataList(xy)

    jl = JointLikelihood(model, data)
    _ = jl.fit()

    ar = jl.results

    # Cache it so we don't continue doing it
    _cache['ar'] = ar

    return ar


def _get_bayes_analysis_results():

    global _cache

    if 'arb' in _cache:

        return _cache['arb']

    y = np.array(poiss_sig)

    xy = XYLike("test", x, y, poisson_data=True)

    fitfun = Line() + Gaussian()

    fitfun.a_1.bounds = (-10, 10.0)
    fitfun.b_1.bounds = (-100, 100.0)
    fitfun.F_2 = 60.0
    fitfun.F_2.bounds = (1e-3, 200.0)
    fitfun.mu_2 = 5.0
    fitfun.mu_2.bounds = (0.0, 100.0)
    fitfun.sigma_2.bounds = (1e-3, 10.0)

    model = Model(PointSource('fake',0.0, 0.0, fitfun))

    data = DataList(xy)

    # Exactly the same can be done for a Bayesian analysis
    # Let's run it first

    ar = _get_mle_analysis_results()

    for parameter in ar.optimized_model:

        model[parameter.path].value = parameter.value

    model.fake.spectrum.main.composite.a_1.set_uninformative_prior(Uniform_prior)
    model.fake.spectrum.main.composite.b_1.set_uninformative_prior(Uniform_prior)
    model.fake.spectrum.main.composite.F_2.set_uninformative_prior(Log_uniform_prior)
    model.fake.spectrum.main.composite.mu_2.set_uninformative_prior(Uniform_prior)
    model.fake.spectrum.main.composite.sigma_2.set_uninformative_prior(Log_uniform_prior)

    bs = BayesianAnalysis(model, data)

    _ = bs.sample(20, 100, 1000)

    arb = bs.results

    _cache['arb'] = arb

    return arb


def _results_are_same(res1, res2, bayes=False):

    # Check that they are the same

    if not bayes:

        # Check covariance

        assert np.allclose(res1.covariance_matrix, res2.covariance_matrix)

    else:

        # Check samples
        np.allclose(res1.samples, res2.samples)

    frame1 = res1.get_data_frame()
    frame2 = res2.get_data_frame()

    # Remove the units (which cannot be checked with np.allclose)
    unit1 = frame1.pop('unit')
    unit2 = frame2.pop('unit')

    assert np.allclose(frame1.values, frame2.values, rtol=0.15)
    assert np.all(unit1 == unit2)

    # Now check the values for the statistics

    s1 = res1.optimal_statistic_values
    s2 = res2.optimal_statistic_values

    assert np.allclose(s1.values, s2.values)




def test_analysis_results_input_output():

    ar = _get_mle_analysis_results()  # type: MLEResults

    temp_file = "__test_mle.fits"

    ar.write_to(temp_file, overwrite=True)

    ar_reloaded = load_analysis_results(temp_file)

    os.remove(temp_file)

    _results_are_same(ar, ar_reloaded)


def test_analysis_set_input_output():

    ar = _get_mle_analysis_results()
    ar2 = _get_mle_analysis_results()

    analysis_set = AnalysisResultsSet([ar, ar2])

    analysis_set.set_bins("testing", [-1, 1], [3, 5], unit = 's')

    temp_file = "_analysis_set_test"

    analysis_set.write_to(temp_file, overwrite=True)

    analysis_set_reloaded = load_analysis_results(temp_file)

    os.remove(temp_file)

    # Test they are the same
    assert len(analysis_set_reloaded) == len(analysis_set)

    for res1, res2 in zip(analysis_set, analysis_set_reloaded):

        _results_are_same(res1, res2)


def test_error_propagation():

    ar = _get_mle_analysis_results()

    # You can use the results for propagating errors non-linearly for analytical functions
    p1 = ar.get_variates("fake.spectrum.main.composite.a_1")
    p2 = ar.get_variates("fake.spectrum.main.composite.b_1")

    print(p1)
    print(p2)

    res = p1 + p2

    assert abs(res.value - (p1.value + p2.value)) / (p1.value + p2.value) < 0.01

    # Make ratio with error 0
    res = p1 / p1

    low_b, hi_b = res.equal_tail_confidence_interval()

    assert low_b == 1
    assert hi_b == 1

    # Now with a function
    fitfun = ar.optimized_model.fake.spectrum.main.shape

    arguments = {}

    for par in fitfun.parameters.values():

        if par.free:

            this_name = par.name

            this_variate = ar.get_variates(par.path)

            # Do not use more than 1000 values (would make computation too slow for nothing)

            if len(this_variate) > 1000:

                this_variate = np.random.choice(this_variate, size=1000)

            arguments[this_name] = this_variate

    # Prepare the error propagator function

    pp = ar.propagate(ar.optimized_model.fake.spectrum.main.shape.evaluate_at, **arguments)

    new_variate = pp(5.0)

    assert abs(new_variate.median - 130.0) < 20

    low_b, hi_b = new_variate.equal_tail_confidence_interval()

    assert abs(low_b - 120) < 20
    assert abs(hi_b - 140) < 20


def test_bayesian_input_output():

    rb1 = _get_bayes_analysis_results()

    temp_file = "_test_bayes.fits"

    rb1.write_to(temp_file, overwrite=True)

    rb2 = load_analysis_results(temp_file)

    os.remove(temp_file)

    _results_are_same(rb1, rb2, bayes=True)