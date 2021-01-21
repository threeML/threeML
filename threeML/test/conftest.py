import os
import signal
import subprocess
import time
from pathlib import Path

import numpy as np
import numba as nb
import pytest
from astromodels import *
from astromodels import (Blackbody, Gaussian, Line, Log_uniform_prior, Model,
                         PointSource, Powerlaw, Uniform_prior)

from threeML.bayesian.bayesian_analysis import BayesianAnalysis
from threeML.classicMLE.joint_likelihood import JointLikelihood
from threeML.data_list import DataList
from threeML.io.package_data import get_path_of_data_dir
from threeML.plugins.OGIPLike import OGIPLike
from threeML.plugins.PhotometryLike import PhotometryLike
from threeML.plugins.XYLike import XYLike
from threeML.utils.photometry import get_photometric_filter_library, PhotometericObservation
from threeML.plugins.UnbinnedPoissonLike import EventObservation
from threeML.plugins.XYLike import XYLike
from threeML.utils.numba_utils import VectorFloat64

# Set up an ipyparallel cluster for the tests to use


@pytest.fixture(scope="session", autouse=True)
def setup_ipcluster():

    ipycluster_process = subprocess.Popen(["ipcluster", "start", "-n", "2"])

    time.sleep(10.0)

    yield ipycluster_process

    ipycluster_process.send_signal(signal.SIGINT)

    time.sleep(10.0)

    ipycluster_process.kill()


# This is run automatically before *every* test (autouse=True)
@pytest.fixture(scope="function", autouse=True)
def reset_random_seed():

    # Reset the random seed so the results of the tests using
    # random numbers are actually predictable
    np.random.seed(1234)

    # Suppress numpy warnings
    np.seterr(over="ignore", under="ignore", divide="ignore", invalid="ignore")


def get_grb_model(spectrum):

    triggerName = "bn090217206"
    ra = 204.9
    dec = -8.4

    GRB = PointSource(triggerName, ra, dec, spectral_shape=spectrum)

    model = Model(GRB)

    return model


def get_test_datasets_directory():

    return Path(get_path_of_data_dir(), "datasets").absolute()


def get_dataset():

    data_dir = Path(get_test_datasets_directory(), "bn090217206")

    obs_spectrum = Path(data_dir, "bn090217206_n6_srcspectra.pha{1}")
    bak_spectrum = Path(data_dir, "bn090217206_n6_bkgspectra.bak{1}")
    rsp_file = Path(data_dir, "bn090217206_n6_weightedrsp.rsp{1}")
    NaI6 = OGIPLike("NaI6", str(obs_spectrum),
                    str(bak_spectrum), str(rsp_file))
    NaI6.set_active_measurements("10.0-30.0", "40.0-950.0")

    return NaI6


def get_dataset_det(det):

    data_dir = Path(get_test_datasets_directory(), "bn090217206")

    obs_spectrum = Path(data_dir, f"bn090217206_{det}_srcspectra.pha{{1}}")
    bak_spectrum = Path(data_dir, f"bn090217206_{det}_bkgspectra.bak{{1}}")
    rsp_file = Path(data_dir, f"bn090217206_{det}_weightedrsp.rsp{{1}}")
    p = OGIPLike(det, str(obs_spectrum),
                 str(bak_spectrum), str(rsp_file))
    if det[0] == "b":
        p.set_active_measurements("250-25000")
    else:
        p.set_active_measurements("10.0-30.0", "40.0-950.0")

    return p


@pytest.fixture(scope="session")
def data_list_bn090217206_nai6():

    NaI6 = get_dataset()

    data_list = DataList(NaI6)

    return data_list


@pytest.fixture(scope="session")
def data_list_bn090217206_nai6_nai9_bgo1():

    p_list = []
    p_list.append(get_dataset_det("n6"))
    p_list.append(get_dataset_det("n9"))
    p_list.append(get_dataset_det("b1"))

    data_list = DataList(*p_list)

    return data_list

# This is going to be run every time a test need it, so the jl object
# is always "fresh"


@pytest.fixture(scope="function")
def joint_likelihood_bn090217206_nai(data_list_bn090217206_nai6):

    powerlaw = Powerlaw()

    model = get_grb_model(powerlaw)

    jl = JointLikelihood(model, data_list_bn090217206_nai6, verbose=False)

    return jl

# This is going to be run every time a test need it, so the jl object
# is always "fresh"


@pytest.fixture(scope="function")
def joint_likelihood_bn090217206_nai6_nai9_bgo1(data_list_bn090217206_nai6_nai9_bgo1):

    powerlaw = Powerlaw()

    model = get_grb_model(powerlaw)

    jl = JointLikelihood(
        model, data_list_bn090217206_nai6_nai9_bgo1, verbose=False)

    return jl


# No need to keep refitting, so we fit once (scope=session)
@pytest.fixture(scope="function")
def fitted_joint_likelihood_bn090217206_nai(joint_likelihood_bn090217206_nai):

    jl = joint_likelihood_bn090217206_nai

    fit_results, like_frame = jl.fit()

    return jl, fit_results, like_frame

# No need to keep refitting, so we fit once (scope=session)


@pytest.fixture(scope="function")
def fitted_joint_likelihood_bn090217206_nai6_nai9_bgo1(joint_likelihood_bn090217206_nai6_nai9_bgo1):

    jl = joint_likelihood_bn090217206_nai6_nai9_bgo1

    fit_results, like_frame = jl.fit()

    return jl, fit_results, like_frame


def set_priors(model):

    powerlaw = model.bn090217206.spectrum.main.Powerlaw

    powerlaw.index.prior = Uniform_prior(lower_bound=-5.0, upper_bound=5.0)
    powerlaw.K.prior = Log_uniform_prior(lower_bound=1.0, upper_bound=10)


def remove_priors(model):

    for parameter in model:

        parameter.prior = None


@pytest.fixture(scope="function")
def bayes_fitter(fitted_joint_likelihood_bn090217206_nai):
    jl, fit_results, like_frame = fitted_joint_likelihood_bn090217206_nai
    datalist = jl.data_list
    model = jl.likelihood_model

    jl.restore_best_fit()

    set_priors(model)

    bayes = BayesianAnalysis(model, datalist)

    return bayes


@pytest.fixture(scope="function")
def completed_bn090217206_bayesian_analysis(fitted_joint_likelihood_bn090217206_nai):

    jl, _, _ = fitted_joint_likelihood_bn090217206_nai

    jl.restore_best_fit()

    model = jl.likelihood_model
    data_list = jl.data_list
    powerlaw = jl.likelihood_model.bn090217206.spectrum.main.Powerlaw

    powerlaw.index.prior = Uniform_prior(lower_bound=-5.0, upper_bound=5.0)
    powerlaw.K.prior = Log_uniform_prior(lower_bound=1.0, upper_bound=10)

    bayes = BayesianAnalysis(model, data_list)

    bayes.set_sampler("emcee")
    bayes.sampler.setup(n_walkers=50, n_burn_in=50,
                        n_iterations=100, seed=1234)
    samples = bayes.sample()

    return bayes, samples


2


@pytest.fixture(scope="session")
def joint_likelihood_bn090217206_nai_multicomp(data_list_bn090217206_nai6):

    composite = Powerlaw() + Blackbody()

    model = get_grb_model(composite)

    jl = JointLikelihood(model, data_list_bn090217206_nai6, verbose=False)

    return jl


# No need to keep refitting, so we fit once (scope=session)
@pytest.fixture(scope="session")
def fitted_joint_likelihood_bn090217206_nai_multicomp(
    joint_likelihood_bn090217206_nai_multicomp,
):

    jl = joint_likelihood_bn090217206_nai_multicomp

    fit_results, like_frame = jl.fit()

    return jl, fit_results, like_frame


@pytest.fixture(scope="session")
def completed_bn090217206_bayesian_analysis_multicomp(
    fitted_joint_likelihood_bn090217206_nai_multicomp,
):

    jl, _, _ = fitted_joint_likelihood_bn090217206_nai_multicomp

    # This is necessary because other tests/functions might have messed up with the
    # model stored within
    jl.restore_best_fit()

    model = jl.likelihood_model
    data_list = jl.data_list
    spectrum = jl.likelihood_model.bn090217206.spectrum.main.shape

    spectrum.index_1.prior = Uniform_prior(lower_bound=-5.0, upper_bound=5.0)
    spectrum.K_1.prior = Log_uniform_prior(lower_bound=1.0, upper_bound=10)
    spectrum.K_2.prior = Log_uniform_prior(lower_bound=1e-20, upper_bound=10)
    spectrum.kT_2.prior = Log_uniform_prior(lower_bound=1e0, upper_bound=1e3)

    bayes = BayesianAnalysis(model, data_list)

    bayes.set_sampler("emcee")

    bayes.sampler.setup(n_walkers=50, n_burn_in=50,
                        n_iterations=100, seed=1234)

    samples = bayes.sample()

    return bayes, bayes.samples


x = np.linspace(0, 10, 50)

poiss_sig = np.array([44, 43, 38, 25, 51, 37, 46, 47, 55, 36, 40, 32, 46, 37,
                      44, 42, 50, 48, 52, 47, 39, 55, 80, 93, 123, 135, 96, 74,
                      43, 49, 43, 51, 27, 32, 35, 42, 43, 49, 38, 43, 59, 54,
                      50, 40, 50, 57, 55, 47, 38, 64])


@pytest.fixture(scope="session")
def xy_model_and_datalist():

    y = np.array(poiss_sig)

    xy = XYLike("test", x, y, poisson_data=True)

    fitfun = Line() + Gaussian()

    fitfun.b_1.bounds = (-10, 10.0)
    fitfun.a_1.bounds = (-100, 100.0)
    fitfun.F_2 = 60.0
    fitfun.F_2.bounds = (1e-3, 200.0)
    fitfun.mu_2 = 5.0
    fitfun.mu_2.bounds = (0.0, 100.0)
    fitfun.sigma_2.bounds = (1e-3, 10.0)

    model = Model(PointSource("fake", 0.0, 0.0, fitfun))

    data = DataList(xy)

    return model, data


@pytest.fixture(scope="session")
def xy_fitted_joint_likelihood(xy_model_and_datalist):

    model, data = xy_model_and_datalist

    jl = JointLikelihood(model, data)
    res_frame, like_frame = jl.fit()

    return jl, res_frame, like_frame


@pytest.fixture(scope="session")
def xy_completed_bayesian_analysis(xy_fitted_joint_likelihood):

    jl, _, _ = xy_fitted_joint_likelihood

    jl.restore_best_fit()

    model = jl.likelihood_model
    data = jl.data_list

    model.fake.spectrum.main.composite.a_1.set_uninformative_prior(
        Uniform_prior)
    model.fake.spectrum.main.composite.b_1.set_uninformative_prior(
        Uniform_prior)
    model.fake.spectrum.main.composite.F_2.set_uninformative_prior(
        Log_uniform_prior)
    model.fake.spectrum.main.composite.mu_2.set_uninformative_prior(
        Uniform_prior)
    model.fake.spectrum.main.composite.sigma_2.set_uninformative_prior(
        Log_uniform_prior
    )

    bs = BayesianAnalysis(model, data)

    bs.set_sampler("emcee")

    bs.sampler.setup(n_burn_in=100, n_iterations=100, n_walkers=20)

    samples = bs.sample()

    return bs, samples


@pytest.fixture(scope="function")
def test_directory():

    test_directory = Path("dummy_dir")

    test_directory.mkdir(parents=True, exist_ok=True)

    yield test_directory

    test_directory.rmdir()


@pytest.fixture(scope="function")
def test_file():

    test_file = Path("dummy_file")

    test_file.touch(exist_ok=True)

    yield test_file

    test_file.unlink()


@pytest.fixture(scope="session")
def threeML_filter_library():

    threeML_filter_library = get_photometric_filter_library()

    yield threeML_filter_library



@pytest.fixture(scope="session")
def photo_obs():

    photo_obs = PhotometericObservation.from_kwargs(
        g=(19.92, 0.1),
        r=(19.75, 0.1),
        i=(19.65, 0.1),
        z=(19.56, 0.1),
        J=(19.38, 0.1),
        H=(19.22, 0.1),
        K=(19.07, 0.1),
)

    fn = Path("grond_observation.h5")

    photo_obs.to_hdf5(fn, overwrite=True)

    restored = PhotometericObservation.from_hdf5(fn)

    yield restored

    fn.unlink()
    
    
@pytest.fixture(scope="function")
def grond_plugin(threeML_filter_library, photo_obs):

    grond = PhotometryLike(
        "GROND",
        filters=threeML_filter_library.LaSilla.GROND,
        observation=photo_obs
    )

    yield grond


@pytest.fixture(scope="function")
def photometry_data_model(grond_plugin):

    spec = Powerlaw()  # * XS_zdust() * XS_zdust()

    datalist = DataList(grond_plugin)

    model = Model(PointSource("grb", 0, 0, spectral_shape=spec))

    yield model, datalist

    
@nb.njit(fastmath=True, cache=True)
def poisson_generator(tstart, tstop, slope, intercept, seed=1234):
    """
    Non-homogeneous poisson process generator
    for a given max rate and time range, this function
    generates time tags sampled from the energy integrated
    lightcurve.
    """

    np.random.seed(seed)

    num_time_steps = 1000

    time_grid = np.linspace(tstart, tstop + 1.0, num_time_steps)

    tmp = intercept + slope * time_grid

    fmax = tmp.max()

    time = tstart
    arrival_times = VectorFloat64(0)
    arrival_times.append(tstart)

    while time < tstop:

        time = time - (1.0 / fmax) * np.log(np.random.rand())
        test = np.random.rand()

        p_test = (intercept + slope * time) / fmax

        if test <= p_test:
            arrival_times.append(time)

    return arrival_times.arr


@pytest.fixture(scope="session")
def event_observation_contiguous():

    events = poisson_generator(
        tstart=0, tstop=10, slope=1., intercept=10, seed=1234)

    obs = EventObservation(events, exposure=10, start=0., stop=10.)

    yield obs


@pytest.fixture(scope="session")
def event_observation_split():

    events = poisson_generator(
        tstart=0, tstop=2, slope=.2, intercept=1, seed=1234)
    events = np.append(events, poisson_generator(
        tstart=30, tstop=40, slope=.2, intercept=1, seed=1234))

    obs = EventObservation(events, exposure=12, start=[0., 30.], stop=[2., 40.])

    yield obs
