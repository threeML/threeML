import pytest
import numpy as np

from threeML import LocalMinimization, GlobalMinimization
from threeML import parallel_computation


try:

    import ROOT

except:

    has_root = False

else:

    has_root = True

skip_if_ROOT_is_not_available = pytest.mark.skipif(
    not has_root, reason="No ROOT available"
)


try:

    import pygmo

except:

    has_pygmo = False

else:

    has_pygmo = True

skip_if_pygmo_is_not_available = pytest.mark.skipif(
    not has_pygmo, reason="No pygmo available"
)

skip_if_ROOT_is_available = pytest.mark.skipif(
    (not has_pygmo) or has_root, reason="ROOT is available. Skipping incompatible tests."
)


def check_results(fit_results):

    assert abs(fit_results['value']['bn090217206.spectrum.main.Powerlaw.K'] - 2.531028) < 5e-2
    assert abs(fit_results['value']['bn090217206.spectrum.main.Powerlaw.index'] + 1.1831566000728451) < 1e-2


def do_analysis(jl, minimizer):

    jl.set_minimizer(minimizer)

    fit_results, like_frame = jl.fit()

    check_results(fit_results)

    fit_results = jl.get_errors()

    check_results(fit_results)
    

def test_minuit_simple(joint_likelihood_bn090217206_nai):

    do_analysis(joint_likelihood_bn090217206_nai, "minuit")


def test_minuit_complete(joint_likelihood_bn090217206_nai):

    minuit = LocalMinimization("minuit")
    minuit.setup(ftol=1e-3)

    do_analysis(joint_likelihood_bn090217206_nai, minuit)


@skip_if_ROOT_is_not_available
def test_ROOT_simple(joint_likelihood_bn090217206_nai):

    do_analysis(joint_likelihood_bn090217206_nai, "ROOT")


@skip_if_ROOT_is_not_available
def test_ROOT_complete(joint_likelihood_bn090217206_nai):

    root = LocalMinimization("ROOT")
    root.setup(ftol=1e-3, max_function_calls=10000, strategy=2)

    do_analysis(joint_likelihood_bn090217206_nai, root)


def test_grid(joint_likelihood_bn090217206_nai):

    grid = GlobalMinimization("GRID")
    minuit = LocalMinimization("minuit")

    grid.setup(
        grid={
            joint_likelihood_bn090217206_nai.likelihood_model.bn090217206.spectrum.main.Powerlaw.K: np.linspace(
                0.1, 10, 10
            )
        },
        second_minimization=minuit,
    )

    do_analysis(joint_likelihood_bn090217206_nai, grid)


@skip_if_pygmo_is_not_available
def test_pagmo(joint_likelihood_bn090217206_nai):

    pagmo = GlobalMinimization("PAGMO")
    minuit = LocalMinimization("minuit")

    algo = pygmo.algorithm(pygmo.bee_colony(gen=100))

    pagmo.setup(
        islands=4,
        population_size=20,
        evolution_cycles=1,
        second_minimization=minuit,
        algorithm=algo,
    )

    do_analysis(joint_likelihood_bn090217206_nai, pagmo)


@skip_if_ROOT_is_available
def test_parallel_pagmo(joint_likelihood_bn090217206_nai):

    with parallel_computation(start_cluster=False):

        test_pagmo(joint_likelihood_bn090217206_nai)


def test_scipy(joint_likelihood_bn090217206_nai):

    minim = LocalMinimization("scipy")

    do_analysis(joint_likelihood_bn090217206_nai, minim)

    joint_likelihood_bn090217206_nai.likelihood_model.bn090217206.spectrum.main.Powerlaw.K = (
        1.25
    )

    do_analysis(joint_likelihood_bn090217206_nai, minim)
