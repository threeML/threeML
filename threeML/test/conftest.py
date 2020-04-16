import pytest
import numpy as np
import os
import subprocess
import time
import signal

from astromodels import *
from threeML.classicMLE.joint_likelihood import JointLikelihood
from threeML.bayesian.bayesian_analysis import BayesianAnalysis
from threeML.io.package_data import get_path_of_data_dir
from threeML.data_list import DataList
from threeML.plugins.OGIPLike import OGIPLike
from threeML.plugins.XYLike import XYLike
from astromodels import PointSource, Model, Uniform_prior, Log_uniform_prior
from astromodels import Line, Gaussian, Blackbody, Powerlaw

# Set up an ipyparallel cluster for the tests to use
@pytest.fixture(scope="session", autouse=True)
def setup_ipcluster():

    ipycluster_process = subprocess.Popen(["ipcluster", "start", "-n", "2"])

    time.sleep(5.0)

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

    return os.path.abspath(os.path.join(get_path_of_data_dir(), "datasets"))


def get_dataset():

    datadir = os.path.join(get_test_datasets_directory(), "bn090217206")

    obsSpectrum = os.path.join(datadir, "bn090217206_n6_srcspectra.pha{1}")
    bakSpectrum = os.path.join(datadir, "bn090217206_n6_bkgspectra.bak{1}")
    rspFile = os.path.join(datadir, "bn090217206_n6_weightedrsp.rsp{1}")
    NaI6 = OGIPLike("NaI6", obsSpectrum, bakSpectrum, rspFile)
    NaI6.set_active_measurements("10.0-30.0", "40.0-950.0")

    return NaI6


@pytest.fixture(scope="session")
def data_list_bn090217206_nai6():

    NaI6 = get_dataset()

    data_list = DataList(NaI6)

    return data_list


# This is going to be run every time a test need it, so the jl object
# is always "fresh"
@pytest.fixture(scope="function")
def joint_likelihood_bn090217206_nai(data_list_bn090217206_nai6):

    powerlaw = Powerlaw()

    model = get_grb_model(powerlaw)

    jl = JointLikelihood(model, data_list_bn090217206_nai6, verbose=False)

    return jl


# No need to keep refitting, so we fit once (scope=session)
@pytest.fixture(scope="function")
def fitted_joint_likelihood_bn090217206_nai(joint_likelihood_bn090217206_nai):

    jl = joint_likelihood_bn090217206_nai

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
    bayes.sampler.setup(n_walkers=50, n_burn_in=50, n_iterations=100, seed=1234)
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

    bayes.sampler.setup(n_walkers=50, n_burn_in=50, n_iterations=100, seed=1234)

    samples = bayes.sample()

    return bayes, bayes.samples


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


@pytest.fixture(scope="session")
def xy_model_and_datalist():

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

    model.fake.spectrum.main.composite.a_1.set_uninformative_prior(Uniform_prior)
    model.fake.spectrum.main.composite.b_1.set_uninformative_prior(Uniform_prior)
    model.fake.spectrum.main.composite.F_2.set_uninformative_prior(Log_uniform_prior)
    model.fake.spectrum.main.composite.mu_2.set_uninformative_prior(Uniform_prior)
    model.fake.spectrum.main.composite.sigma_2.set_uninformative_prior(
        Log_uniform_prior
    )

    bs = BayesianAnalysis(model, data)

    bs.set_sampler("emcee")

    bs.sampler.setup(n_burn_in=100, n_iterations=100, n_walkers=20)

    samples = bs.sample()

    return bs, samples
