from __future__ import division
from __future__ import print_function
from builtins import zip
from past.utils import old_div
import pytest
import os
import numpy as np
import astropy.units as u

from threeML.plugins.XYLike import XYLike
from threeML import Model, DataList, JointLikelihood, PointSource
from threeML import BayesianAnalysis, Uniform_prior, Log_uniform_prior
from threeML.analysis_results import (
    MLEResults,
    load_analysis_results,
    load_analysis_results_hdf,
    convert_fits_analysis_result_to_hdf,
    AnalysisResultsSet,
)
from astromodels import Line, Gaussian, Powerlaw


_cache = {}

# These are the same simulated dataset we use in the test of the XY plugin

x = np.linspace(0, 10, 50)

poiss_sig = [
    44,
    43,
    38,
    25,
    51,
    37,
    46,
    47,
    55,
    36,
    40,
    32,
    46,
    37,
    44,
    42,
    50,
    48,
    52,
    47,
    39,
    55,
    80,
    93,
    123,
    135,
    96,
    74,
    43,
    49,
    43,
    51,
    27,
    32,
    35,
    42,
    43,
    49,
    38,
    43,
    59,
    54,
    50,
    40,
    50,
    57,
    55,
    47,
    38,
    64,
]


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
    unit1 = frame1.pop("unit")
    unit2 = frame2.pop("unit")

    assert np.allclose(frame1.values, frame2.values, rtol=0.15)
    assert np.all(unit1 == unit2)

    # Now check the values for the statistics

    s1 = res1.optimal_statistic_values
    s2 = res2.optimal_statistic_values

    assert np.allclose(s1.values, s2.values)


def test_analysis_results_input_output(xy_fitted_joint_likelihood):

    jl, _, _ = xy_fitted_joint_likelihood  # type: JointLikelihood, None, None

    jl.restore_best_fit()

    ar = jl.results  # type: MLEResults

    temp_file = "__test_mle.fits"

    ar.write_to(temp_file, overwrite=True)

    ar_reloaded = load_analysis_results(temp_file)

    os.remove(temp_file)

    _results_are_same(ar, ar_reloaded)

def test_analysis_results_input_output_hdf(xy_fitted_joint_likelihood):

    jl, _, _ = xy_fitted_joint_likelihood  # type: JointLikelihood, None, None

    jl.restore_best_fit()

    ar = jl.results  # type: MLEResults

    temp_file = "__test_mle.h5"

    ar.write_to(temp_file, overwrite=True, as_hdf=True)

    ar_reloaded = load_analysis_results_hdf(temp_file)

    os.remove(temp_file)

    _results_are_same(ar, ar_reloaded)

    

def test_analysis_set_input_output(xy_fitted_joint_likelihood):

    # Collect twice the same analysis results just to see if we can
    # save them in a file as set of results

    jl, _, _ = xy_fitted_joint_likelihood  # type: JointLikelihood, None, None

    jl.restore_best_fit()

    ar = jl.results  # type: MLEResults

    ar2 = jl.results

    analysis_set = AnalysisResultsSet([ar, ar2])

    analysis_set.set_bins("testing", [-1, 1], [3, 5], unit="s")

    temp_file = "_analysis_set_test"

    analysis_set.write_to(temp_file, overwrite=True)

    analysis_set_reloaded = load_analysis_results(temp_file)

    os.remove(temp_file)

    # Test they are the same
    assert len(analysis_set_reloaded) == len(analysis_set)

    for res1, res2 in zip(analysis_set, analysis_set_reloaded):

        _results_are_same(res1, res2)


def test_conversion_fits2hdf(xy_fitted_joint_likelihood):

    jl, _, _ = xy_fitted_joint_likelihood  # type: JointLikelihood, None, None

    jl.restore_best_fit()

    ar = jl.results  # type: MLEResults

    ar2 = jl.results

    analysis_set = AnalysisResultsSet([ar, ar2])

    analysis_set.set_bins("testing", [-1, 1], [3, 5], unit="s")

    temp_file = "_analysis_set_test.fits"

    analysis_set.write_to(temp_file, overwrite=True)

    convert_fits_analysis_result_to_hdf(temp_file)

    analysis_set_reloaded = load_analysis_results_hdf("_analysis_set_test.h5")

        # Test they are the same
    assert len(analysis_set_reloaded) == len(analysis_set)

    for res1, res2 in zip(analysis_set, analysis_set_reloaded):

        _results_are_same(res1, res2)

    
        
def test_analysis_set_input_output_hdf(xy_fitted_joint_likelihood):

    # Collect twice the same analysis results just to see if we can
    # save them in a file as set of results

    jl, _, _ = xy_fitted_joint_likelihood  # type: JointLikelihood, None, None

    jl.restore_best_fit()

    ar = jl.results  # type: MLEResults

    ar2 = jl.results

    analysis_set = AnalysisResultsSet([ar, ar2])

    analysis_set.set_bins("testing", [-1, 1], [3, 5], unit="s")

    temp_file = "_analysis_set_test_hdf"

    analysis_set.write_to(temp_file, overwrite=True, as_hdf=True)

    analysis_set_reloaded = load_analysis_results_hdf(temp_file)

    os.remove(temp_file)

    # Test they are the same
    assert len(analysis_set_reloaded) == len(analysis_set)

    for res1, res2 in zip(analysis_set, analysis_set_reloaded):

        _results_are_same(res1, res2)


def test_error_propagation(xy_fitted_joint_likelihood):

    jl, _, _ = xy_fitted_joint_likelihood  # type: JointLikelihood, None, None

    jl.restore_best_fit()

    ar = jl.results  # type: MLEResults

    # You can use the results for propagating errors non-linearly for analytical functions
    p1 = ar.get_variates("fake.spectrum.main.composite.b_1")
    p2 = ar.get_variates("fake.spectrum.main.composite.a_1")

    # Test the printing
    print(p1)
    print(p2)

    res = p1 + p2

    assert old_div(abs(res.value - (p1.value + p2.value)), (p1.value + p2.value)) < 0.01

    # Make ratio with error 0
    res = old_div(p1, p1)

    low_b, hi_b = res.equal_tail_interval()

    assert low_b == 1
    assert hi_b == 1

    # Now with a function
    fitfun = ar.optimized_model.fake.spectrum.main.shape

    arguments = {}

    for par in list(fitfun.parameters.values()):

        if par.free:

            this_name = par.name

            this_variate = ar.get_variates(par.path)

            # Do not use more than 1000 values (would make computation too slow for nothing)

            if len(this_variate) > 1000:

                this_variate = np.random.choice(this_variate, size=1000)

            arguments[this_name] = this_variate

    # Prepare the error propagator function

    pp = ar.propagate(
        ar.optimized_model.fake.spectrum.main.shape.evaluate_at, **arguments
    )

    new_variate = pp(5.0)

    assert abs(new_variate.median - 130.0) < 20

    low_b, hi_b = new_variate.equal_tail_interval()

    assert abs(low_b - 120) < 20
    assert abs(hi_b - 140) < 20


def test_bayesian_input_output(xy_completed_bayesian_analysis):

    bs, _ = xy_completed_bayesian_analysis

    rb1 = bs.results

    temp_file = "_test_bayes.fits"

    rb1.write_to(temp_file, overwrite=True)

    rb2 = load_analysis_results(temp_file)

    os.remove(temp_file)

    _results_are_same(rb1, rb2, bayes=True)


def test_corner_plotting(xy_completed_bayesian_analysis):

    bs, _ = xy_completed_bayesian_analysis

    ar = bs.results

    ar.corner_plot()


def test_one_free_parameter_input_output():

    fluxUnit = 1.0 / (u.TeV * u.cm ** 2 * u.s)

    temp_file = "__test_mle.fits"

    spectrum = Powerlaw()
    source = PointSource("tst", ra=100, dec=20, spectral_shape=spectrum)
    model = Model(source)

    spectrum.piv = 7 * u.TeV
    spectrum.index = -2.3
    spectrum.K = 1e-15 * fluxUnit

    spectrum.piv.fix = True

    # two free parameters (one with units)
    spectrum.index.fix = False
    spectrum.K.fix = False
    cov_matrix = np.diag([0.001] * 2)
    ar = MLEResults(model, cov_matrix, {})

    ar.write_to(temp_file, overwrite=True)
    ar_reloaded = load_analysis_results(temp_file)
    os.remove(temp_file)
    _results_are_same(ar, ar_reloaded)

    # one free parameter with units
    spectrum.index.fix = True
    spectrum.K.fix = False
    cov_matrix = np.diag([0.001] * 1)
    ar = MLEResults(model, cov_matrix, {})

    ar.write_to(temp_file, overwrite=True)
    ar_reloaded = load_analysis_results(temp_file)
    os.remove(temp_file)
    _results_are_same(ar, ar_reloaded)

    # one free parameter without units
    spectrum.index.fix = False
    spectrum.K.fix = True
    cov_matrix = np.diag([0.001] * 1)
    ar = MLEResults(model, cov_matrix, {})

    ar.write_to(temp_file, overwrite=True)
    ar_reloaded = load_analysis_results(temp_file)
    os.remove(temp_file)
    _results_are_same(ar, ar_reloaded)
