from threeML import *
from threeML.plugins.OGIPLike import OGIPLike


def test_a_basic_analysis_from_start_to_finish():
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

    fit_results, like_frame = jl.fit()

    assert abs(fit_results['value']['bn090217206.spectrum.main.Powerlaw.K'] - 2.531028) < 1e-2
    assert abs(fit_results['value']['bn090217206.spectrum.main.Powerlaw.index'] + 1.1831566000728451) < 1e-2

    res = jl.get_errors()

    res = jl.get_contours(powerlaw.index, -1.3, -1.1, 20)

    res = jl.get_contours(powerlaw.index, -1.25, -1.1, 60, powerlaw.K, 1.8, 3.4, 60)


    powerlaw.index.prior = Uniform_prior(lower_bound=-5.0, upper_bound=5.0)
    powerlaw.K.prior = Log_uniform_prior(lower_bound=1.0, upper_bound=10)

    bayes = BayesianAnalysis(model, data_list)

    samples = bayes.sample(n_walkers=50, burn_in=10, n_samples=10)

    fig = bayes.corner_plot()




def test_a_basic_multicomp_analysis_from_start_to_finish():
    triggerName = 'bn090217206'
    ra = 204.9
    dec = -8.4

    datadir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../examples'))

    obsSpectrum = os.path.join(datadir, "bn090217206_n6_srcspectra.pha{1}")
    bakSpectrum = os.path.join(datadir, "bn090217206_n6_bkgspectra.bak{1}")
    rspFile = os.path.join(datadir, "bn090217206_n6_weightedrsp.rsp{1}")

    # Plugin instance
    NaI6 = OGIPLike("NaI6", obsSpectrum, bakSpectrum, rspFile)

    NaI6.set_active_measurements("10.0-30.0", "40.0-950.0")

    data_list = DataList(NaI6)

    powerlaw = Powerlaw() + Blackbody()

    GRB = PointSource(triggerName, ra, dec, spectral_shape=powerlaw)

    model = Model(GRB)

    jl = JointLikelihood(model, data_list, verbose=False)

    fit_results, like_frame = jl.fit()




    powerlaw.index_1.prior = Uniform_prior(lower_bound=-5.0, upper_bound=5.0)
    powerlaw.K_1.prior = Log_uniform_prior(lower_bound=1.0, upper_bound=10)
    powerlaw.K_2.prior = Log_uniform_prior(lower_bound=1E-20, upper_bound=10)
    powerlaw.kT_2.prior = Log_uniform_prior(lower_bound=1E0, upper_bound=1E3)

    bayes = BayesianAnalysis(model, data_list)

    samples = bayes.sample(n_walkers=50, burn_in=10, n_samples=10)

    fig = bayes.corner_plot()

