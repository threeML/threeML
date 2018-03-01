from threeML import DataList, BayesianAnalysis, Uniform_prior, Log_uniform_prior
from threeML import Model, Powerlaw, PointSource
from threeML.plugins.OGIPLike import OGIPLike
import os
import pytest




def get_model_and_datalist(with_prior=True):

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

    datalist = DataList(NaI6)

    powerlaw = Powerlaw()

    model = Model(PointSource(triggerName, ra, dec, spectral_shape=powerlaw))

    if with_prior:

        powerlaw.index.prior = Uniform_prior(lower_bound=-5.0, upper_bound=5.0)
        powerlaw.K.prior = Log_uniform_prior(lower_bound=1.0, upper_bound=10)

    return model, datalist




def get_bayesian_analysis():

    model, datalist = get_model_and_datalist()


    bayes = BayesianAnalysis(model, datalist)


    return bayes


def check_results(fit_results):


    assert abs(fit_results['value']['bn090217206.spectrum.main.Powerlaw.K'] - 2.531028) < 1e-1
    assert abs(fit_results['value']['bn090217206.spectrum.main.Powerlaw.index'] + 1.1831566000728451) < 1e-1


def test_bayes_constructor():

    # before setting priors, cannot create bayes

    model, datalist = get_model_and_datalist(with_prior=False)

    with pytest.raises(RuntimeError):

        bayes = BayesianAnalysis(model, datalist)

    model, datalist = get_model_and_datalist(with_prior=True)



    bayes = BayesianAnalysis(model, datalist)

    with pytest.raises(RuntimeError):

        bayes.corner_plot()


def test_emcee():

    bayes = get_bayesian_analysis()

    n_walkers = 50
    burn_in = 300
    n_samples = 500

    samples = bayes.sample(n_walkers=n_walkers, burn_in=burn_in, n_samples=n_samples)

    assert bayes.raw_samples.shape == (n_walkers * n_samples, 2)


    res = bayes.results.get_data_frame()

    check_results(res)



def test_multinest():


    bayes = get_bayesian_analysis()


    bayes.sample_multinest(n_live_points=400)


    res = bayes.results.get_data_frame()

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

def test_bayes_plots():


    bayes = get_bayesian_analysis()

    n_walkers = 50
    burn_in = 10
    n_samples=100


    samples = bayes.sample(n_walkers=n_walkers, burn_in=burn_in, n_samples=n_samples)

    bayes.corner_plot()

    renamed_parameters = {'K':'norm'}

    bayes.corner_plot(renamed_parameters)

    with pytest.raises(AssertionError):
        bayes.convergence_plots(n_samples_in_each_subset=100,n_subsets=2000)

    bayes.convergence_plots(n_samples_in_each_subset=10, n_subsets=5)


    bayes.plot_chains()
