import pytest
from threeML import *
from threeML.plugins.OGIPLike import OGIPLike
from threeML.io.flux_calculator import InvalidUnitError


# TODO: add tests for area corrections

def test_mle_flux_calculations():
    # In[2]:

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

    # In[3]:

    # This declares which data we want to use. In our case, all that we have already created.

    data_list = DataList(NaI6)

    # In[4]:

    powerlaw = Powerlaw()

    # In[5]:

    GRB = PointSource(triggerName, ra, dec, spectral_shape=powerlaw)

    # In[6]:

    model = Model(GRB)

    # In[7]:

    jl = JointLikelihood(model, data_list, verbose=False)

    fit_results, like_frame = jl.fit()

    flux = SpectralFlux(jl)

    res = flux.model_flux(flux_unit='erg/(s cm2)', energy_unit='keV', ene_min=10, ene_max=40000)

    res = flux.model_flux(flux_unit='1/(s cm2)', energy_unit='keV', ene_min=10, ene_max=40000)

    res = flux.model_flux(flux_unit='erg2/(s cm2)', energy_unit='MeV', ene_min=10, ene_max=40000)

    with pytest.raises(InvalidUnitError):
        res = flux.model_flux(flux_unit='erg2/(cm2)', energy_unit='keV', ene_min=10, ene_max=40000)


def test_bayes_flux_calculations():
    # In[2]:

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

    # In[3]:

    # This declares which data we want to use. In our case, all that we have already created.

    data_list = DataList(NaI6)

    # In[4]:

    powerlaw = Powerlaw()

    # In[5]:

    GRB = PointSource(triggerName, ra, dec, spectral_shape=powerlaw)

    # In[6]:

    model = Model(GRB)

    powerlaw.index.prior = Uniform_prior(lower_bound=-5.0, upper_bound=5.0)
    powerlaw.K.prior = Log_uniform_prior(lower_bound=1.0, upper_bound=10)

    bayes = BayesianAnalysis(model, data_list)

    # In[12]:

    samples = bayes.sample(n_walkers=50, burn_in=10, n_samples=10)

    flux = SpectralFlux(bayes)

    res = flux.model_flux(flux_unit='erg/(s cm2)', energy_unit='keV', ene_min=10, ene_max=40000)

    res = flux.model_flux(flux_unit='1/(s cm2)', energy_unit='keV', ene_min=10, ene_max=40000)

    res = flux.model_flux(flux_unit='erg2/(s cm2)', energy_unit='keV', ene_min=10, ene_max=40000)

    with pytest.raises(InvalidUnitError):
        res = flux.model_flux(flux_unit='erg2/(cm2)', energy_unit='keV', ene_min=10, ene_max=40000)
