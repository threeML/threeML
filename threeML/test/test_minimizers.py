import pytest
import os
import numpy as np

from astromodels import Powerlaw, PointSource, Model
from threeML import DataList, JointLikelihood
from threeML.plugins.OGIPLike import OGIPLike
from threeML import LocalMinimization, GlobalMinimization
from threeML import parallel_computation

from threeML.utils.initalize_testing import initialize_testing

initialize_testing()

try:

    import ROOT

except:

    has_root = False

else:

    has_root = True

skip_if_ROOT_is_not_available = pytest.mark.skipif(not has_root, reason="No ROOT available")


try:

    import pygmo

except:

    has_pygmo = False

else:

    has_pygmo = True

skip_if_pygmo_is_not_available = pytest.mark.skipif(not has_pygmo, reason="No pygmo available")


def get_joint_likelihood():

    triggerName = 'bn090217206'
    ra = 204.9
    dec = -8.4

    # Data are in the current directory

    datadir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../examples'))

    # Create an instance of the GBM plugin for each detector
    # Data files
    obsSpectrum = os.path.join(datadir, "bn090217206_n6_srcspectra.pha{1}")
    bakSpectrum = os.path.join(datadir, "bn090217206_n6_bkgspectra.bak{1}")
    rspFile = os.path.join(datadir, "bn090217206_n6_weightedrsp.rsp{1}")

    # Plugin instance
    NaI6 = OGIPLike("NaI6", obsSpectrum, bakSpectrum, rspFile)

    # Choose energies to use (in this case, I exclude the energy
    # range from 30 to 40 keV to avoid the k-edge, as well as anything above
    # 950 keV, where the calibration is uncertain)
    NaI6.set_active_measurements("10.0-30.0", "40.0-950.0")

    NaI6.display_rsp()

    # This declares which data we want to use. In our case, all that we have already created.

    data_list = DataList(NaI6)

    powerlaw = Powerlaw()

    GRB = PointSource(triggerName, ra, dec, spectral_shape=powerlaw)

    model = Model(GRB)

    jl = JointLikelihood(model, data_list, verbose=False)

    return jl


def check_results(fit_results):

    assert abs(fit_results['value']['bn090217206.spectrum.main.Powerlaw.K'] - 2.531028) < 1e-2
    assert abs(fit_results['value']['bn090217206.spectrum.main.Powerlaw.index'] + 1.1831566000728451) < 1e-2


def do_analysis(minimizer):

    jl = get_joint_likelihood()

    jl.set_minimizer(minimizer)

    fit_results, like_frame = jl.fit()

    check_results(fit_results)


def test_minuit_simple():

    do_analysis("minuit")


def test_minuit_complete():

    minuit = LocalMinimization("minuit")
    minuit.setup(ftol=1e-3)

    do_analysis(minuit)


@skip_if_ROOT_is_not_available
def test_ROOT_simple():

    do_analysis("ROOT")


@skip_if_ROOT_is_not_available
def test_ROOT_complete():

    root = LocalMinimization("ROOT")
    root.setup(ftol=1e-3, max_function_calls=10000, strategy=2)

    do_analysis(root)


def test_grid():

    jl = get_joint_likelihood()

    grid = GlobalMinimization("GRID")
    minuit = LocalMinimization("minuit")

    grid.setup(grid={jl.likelihood_model.bn090217206.spectrum.main.Powerlaw.K: np.linspace(0.1, 10, 10)},
               second_minimization=minuit)

    jl.set_minimizer(grid)

    fit_results, like_frame = jl.fit()

    check_results(fit_results)


@skip_if_pygmo_is_not_available
def test_pagmo():

    jl = get_joint_likelihood()

    pagmo = GlobalMinimization("PAGMO")
    minuit = LocalMinimization("minuit")

    algo = pygmo.algorithm(pygmo.bee_colony(gen=100))

    pagmo.setup(islands=4, population_size=20, evolution_cycles=1, second_minimization=minuit, algorithm=algo)

    jl.set_minimizer(pagmo)

    fit_results, like_frame = jl.fit()

    check_results(fit_results)


@skip_if_pygmo_is_not_available
def test_parallel_pagmo():

    with parallel_computation():

        test_pagmo()
