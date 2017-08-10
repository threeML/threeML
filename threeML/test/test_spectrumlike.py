from threeML.plugins.SpectrumLike import SpectrumLike
from threeML.plugins.DispersionSpectrumLike import DispersionSpectrumLike
from threeML.io.package_data import get_path_of_data_file
from threeML.plugins.OGIP.response import OGIPResponse
from threeML import JointLikelihood, DataList

from astromodels import Blackbody, Powerlaw, Model, PointSource

import numpy as np




def test_spectrumlike_fit():

    energies = np.logspace(1, 3, 51)

    low_edge = energies[:-1]
    high_edge = energies[1:]

    sim_K = 1E-1
    sim_kT = 20.

    # get a blackbody source function
    source_function = Blackbody(K=sim_K, kT=sim_kT)

    # power law background function
    background_function = Powerlaw(K=1, index=-1.5, piv=100.)

    spectrum_generator = SpectrumLike.from_function('fake',
                                                    source_function=source_function,
                                                    background_function=background_function,
                                                    energy_min=low_edge,
                                                    energy_max=high_edge)

    bb = Blackbody()

    pts = PointSource('mysource', 0, 0, spectral_shape=bb)

    model = Model(pts)

    # MLE fitting

    jl = JointLikelihood(model, DataList(spectrum_generator))

    result = jl.fit()

    K_variates = jl.results.get_variates('mysource.spectrum.main.Blackbody.K')

    kT_variates = jl.results.get_variates('mysource.spectrum.main.Blackbody.kT')

    assert np.all(np.isclose([K_variates.mean(), kT_variates.mean()], [sim_K, sim_kT], atol=1 ))


def test_dispersionspectrumlike_fit():



    response = OGIPResponse(get_path_of_data_file('datasets/ogip_powerlaw.rsp'))

    sim_K = 1E-1
    sim_kT = 20.

    # get a blackbody source function
    source_function = Blackbody(K=sim_K, kT=sim_kT)

    # power law background function
    background_function = Powerlaw(K=1, index=-1.5, piv=100.)

    spectrum_generator = DispersionSpectrumLike.from_function('test', source_function=source_function,
                                                                    response=response,
                                                                         background_function=background_function)


    bb = Blackbody()

    pts = PointSource('mysource', 0, 0, spectral_shape=bb)

    model = Model(pts)

    # MLE fitting

    jl = JointLikelihood(model, DataList(spectrum_generator))

    result = jl.fit()

    K_variates = jl.results.get_variates('mysource.spectrum.main.Blackbody.K')

    kT_variates = jl.results.get_variates('mysource.spectrum.main.Blackbody.kT')

    assert np.all(np.isclose([K_variates.mean(), kT_variates.mean()], [sim_K, sim_kT], atol=1))




def test_spectrum_like_with_background_model():
    energies = np.logspace(1, 3, 51)

    low_edge = energies[:-1]
    high_edge = energies[1:]

    sim_K = 1E-1
    sim_kT = 20.

    # get a blackbody source function
    source_function = Blackbody(K=sim_K, kT=sim_kT)

    # power law background function
    background_function = Powerlaw(K=5, index=-1.5, piv=100.)

    spectrum_generator = SpectrumLike.from_function('fake',
                                                    source_function=source_function,
                                                    background_function=background_function,
                                                    energy_min=low_edge,
                                                    energy_max=high_edge)


    background_plugin = SpectrumLike.from_background('background',spectrum_generator)


    bb = Blackbody()


    pl = Powerlaw()
    pl.piv = 100

    bkg_ps = PointSource('bkg',0,0,spectral_shape=pl)

    bkg_model = Model(bkg_ps)

    jl_bkg = JointLikelihood(bkg_model,DataList(background_plugin))

    _ = jl_bkg.fit()




    plugin_bkg_model = SpectrumLike('full',spectrum_generator.observed_spectrum,background=background_plugin)

    pts = PointSource('mysource', 0, 0, spectral_shape=bb)

    model = Model(pts)

    # MLE fitting

    jl = JointLikelihood(model, DataList(plugin_bkg_model))

    result = jl.fit()

    K_variates = jl.results.get_variates('mysource.spectrum.main.Blackbody.K')

    kT_variates = jl.results.get_variates('mysource.spectrum.main.Blackbody.kT')

    assert np.all(np.isclose([K_variates.mean(), kT_variates.mean()], [sim_K, sim_kT], rtol=0.5))


