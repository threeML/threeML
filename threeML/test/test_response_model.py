import pytest
from threeML.plugins.spectrum.binned_spectrum import BinnedSpectrum, BinnedSpectrumWithDispersion, ChannelSet
from threeML.plugins.SpectrumLike import SpectrumLike
from threeML.plugins.DispersionSpectrumLike import DispersionSpectrumLike
from threeML.plugins.OGIP.response import OGIPResponse
from astromodels import Powerlaw, PointSource, Model

import numpy as np

import os

__this_dir__ = os.path.join(os.path.abspath(os.path.dirname(__file__)))

__example_dir = os.path.join(__this_dir__, '../../examples')

def test_spectrum_constructor():

    ebounds = ChannelSet.from_list_of_edges(np.array([0,1,2,3,4,5]))

    pl = Powerlaw()

    ps = PointSource('fake',0,0,spectral_shape=pl)

    model = Model(ps)

    obs_spectrum = BinnedSpectrum(counts=np.ones(len(ebounds)),exposure=1,ebounds=ebounds, is_poisson=True)
    bkg_spectrum = BinnedSpectrum(counts=np.ones(len(ebounds)),exposure=1,ebounds=ebounds, is_poisson=True)

    assert np.all(obs_spectrum.counts == obs_spectrum.rates)
    assert np.all(bkg_spectrum.counts == bkg_spectrum.rates)


    specLike = SpectrumLike('fake', observation=obs_spectrum, background=bkg_spectrum)
    specLike.set_model(model)
    specLike.get_model()

    specLike.get_simulated_dataset()

    specLike.rebin_on_background(min_number_of_counts=1E-1)
    specLike.remove_rebinning()


    specLike.significance
    specLike.significance_per_channel

    obs_spectrum = BinnedSpectrum(counts=np.ones(len(ebounds)), count_errors=np.ones(len(ebounds)),exposure=1, ebounds=ebounds, is_poisson=False)
    bkg_spectrum = BinnedSpectrum(counts=np.ones(len(ebounds)), exposure=1, ebounds=ebounds, is_poisson=True)


    with pytest.raises(NotImplementedError):

        specLike = SpectrumLike('fake', observation=obs_spectrum, background=bkg_spectrum)

    # gaussian source only

    obs_spectrum = BinnedSpectrum(counts=np.ones(len(ebounds)), count_errors=np.ones(len(ebounds)), exposure=1,
                                  ebounds=ebounds)

    specLike = SpectrumLike('fake', observation=obs_spectrum, background=None)
    specLike.set_model(model)
    specLike.get_model()

    specLike.get_simulated_dataset()

    with pytest.raises(AssertionError):
        specLike.rebin_on_background(min_number_of_counts=1E-1)

def test_spectrum_constructor_no_background():

    ebounds = ChannelSet.from_list_of_edges(np.array([0,1,2,3,4,5]))

    obs_spectrum = BinnedSpectrum(counts=np.ones(len(ebounds)),exposure=1,ebounds=ebounds, is_poisson=True)

    assert np.all(obs_spectrum.counts == obs_spectrum.rates)

    specLike = SpectrumLike('fake', observation=obs_spectrum, background=None)

    specLike.__repr__()

def addition_proof_simple(x,y,z):
    assert x.counts[3] + y.counts[3] == z.counts[3]

def addition_proof_weighted(x,y,z):
    assert (x.rates[3]/x.rate_errors[3]**2 + y.rates[3]/y.rate_errors[3]**2) / \
           (1/x.rate_errors[3]**2 + 1/y.rate_errors[3]**2) \
           == z.rates[3]/z.exposure 

def spectrum_addition(obs_spectrum_1,obs_spectrum_2,obs_spectrum_incompatible,addition,addition_proof):
    obs_spectrum = addition(obs_spectrum_1, obs_spectrum_2)
    
    addition_proof(obs_spectrum_1, obs_spectrum_2, obs_spectrum)

    assert obs_spectrum_1.exposure + obs_spectrum_2.exposure == obs_spectrum.exposure

    assert np.all(obs_spectrum.counts == obs_spectrum.rates * obs_spectrum.exposure)

    specLike = SpectrumLike('fake', observation=obs_spectrum, background=None)

    assert obs_spectrum.count_errors is None or obs_spectrum.count_errors.__class__ == np.ndarray

    specLike.__repr__()


def test_spectrum_addition():
    ebounds = ChannelSet.from_list_of_edges(np.array([0,1,2,3,4,5]))
    ebounds_different = ChannelSet.from_list_of_edges(np.array([0,1,2,3,4,5]))

    obs_spectrum_1 = BinnedSpectrum(counts=np.ones(len(ebounds)),count_errors=np.ones(len(ebounds)),exposure=1,ebounds=ebounds, is_poisson=False)
    obs_spectrum_2 = BinnedSpectrum(counts=np.ones(len(ebounds)),count_errors=np.ones(len(ebounds)),exposure=2,ebounds=ebounds, is_poisson=False)
    obs_spectrum_incompatible = BinnedSpectrum(counts=np.ones(len(ebounds)),count_errors=np.ones(len(ebounds)),exposure=2,ebounds=ebounds_different, is_poisson=False)

    spectrum_addition(obs_spectrum_1,obs_spectrum_2,obs_spectrum_incompatible,lambda x,y:x+y,addition_proof_simple)
    spectrum_addition(obs_spectrum_1,obs_spectrum_2,obs_spectrum_incompatible,lambda x,y:x.add_inverse_variance_weighted(y),addition_proof_weighted)

def test_spectrum_addition_poisson():
    ebounds = ChannelSet.from_list_of_edges(np.array([0,1,2,3,4,5]))
    ebounds_different = ChannelSet.from_list_of_edges(np.array([0,1,2,3,4,5]))

    obs_spectrum_1 = BinnedSpectrum(counts=np.ones(len(ebounds)),exposure=1,ebounds=ebounds, is_poisson=True)
    obs_spectrum_2 = BinnedSpectrum(counts=np.ones(len(ebounds)),exposure=2,ebounds=ebounds, is_poisson=True)
    obs_spectrum_incompatible = BinnedSpectrum(counts=np.ones(len(ebounds_different)),exposure=2,ebounds=ebounds, is_poisson=True)
    
    spectrum_addition(obs_spectrum_1,obs_spectrum_2,obs_spectrum_incompatible,lambda x,y:x+y,addition_proof_simple)
    #spectrum_addition(obs_spectrum_1,obs_spectrum_2,obs_spectrum_incompatible,lambda x,y:x.add_inverse_variance_weighted(y))


def test_spectrum_clone():
    ebounds = ChannelSet.from_list_of_edges(np.array([0,1,2,3,4,5]))

    obs_spectrum = BinnedSpectrum(counts=np.ones(len(ebounds)),count_errors=np.ones(len(ebounds)),exposure=1,ebounds=ebounds, is_poisson=False)
    obs_spectrum.clone(new_counts=np.zeros_like(obs_spectrum.counts), new_count_errors=np.zeros_like(obs_spectrum.counts))
    obs_spectrum.clone()


def test_dispersion_spectrum_constructor():
    rsp = OGIPResponse(os.path.join(__example_dir, 'bn090217206_n6_weightedrsp.rsp'))

    pl = Powerlaw()

    ps = PointSource('fake', 0, 0, spectral_shape=pl)

    model = Model(ps)

    obs_spectrum = BinnedSpectrumWithDispersion(counts=np.ones(128), exposure=1, response=rsp, is_poisson=True)
    bkg_spectrum = BinnedSpectrumWithDispersion(counts=np.ones(128), exposure=1, response=rsp, is_poisson=True)

    specLike = DispersionSpectrumLike('fake', observation=obs_spectrum, background=bkg_spectrum)
    specLike.set_model(model)
    specLike.get_model()

    specLike.write_pha('test_from_dispersion', overwrite=True)

def test_dispersion_spectrum_addition_poisson():
    rsp = OGIPResponse(os.path.join(__example_dir, 'bn090217206_n6_weightedrsp.rsp'))
    ebounds = ChannelSet.from_instrument_response(rsp)

    obs_spectrum_1 = BinnedSpectrumWithDispersion(counts=np.ones(len(ebounds)),exposure=1, response=rsp, is_poisson=True)
    obs_spectrum_2 = BinnedSpectrumWithDispersion(counts=np.ones(len(ebounds)),exposure=2, response=rsp, is_poisson=True)
    obs_spectrum_incompatible = None
    
    spectrum_addition(obs_spectrum_1,obs_spectrum_2,obs_spectrum_incompatible,lambda x,y:x+y,addition_proof_simple)
    #spectrum_addition(obs_spectrum_1,obs_spectrum_2,obs_spectrum_incompatible,lambda x,y:x.add_inverse_variance_weighted(y),addition_proof_weighted)

def test_dispersion_spectrum_addition():
    rsp = OGIPResponse(os.path.join(__example_dir, 'bn090217206_n6_weightedrsp.rsp'))
    ebounds = ChannelSet.from_instrument_response(rsp)

    obs_spectrum_1 = BinnedSpectrumWithDispersion(counts=np.ones(len(ebounds)),count_errors=np.ones(len(ebounds)),exposure=1, response=rsp, is_poisson=False)
    obs_spectrum_2 = BinnedSpectrumWithDispersion(counts=np.ones(len(ebounds)),count_errors=np.ones(len(ebounds)),exposure=2, response=rsp, is_poisson=False)
    obs_spectrum_incompatible = None

    spectrum_addition(obs_spectrum_1,obs_spectrum_2,obs_spectrum_incompatible,lambda x,y:x+y,addition_proof_simple)
    spectrum_addition(obs_spectrum_1,obs_spectrum_2,obs_spectrum_incompatible,lambda x,y:x.add_inverse_variance_weighted(y),addition_proof_weighted)


def test_dispersion_spectrum_clone():
    rsp = OGIPResponse(os.path.join(__example_dir, 'bn090217206_n6_weightedrsp.rsp'))

    obs_spectrum = BinnedSpectrumWithDispersion(counts=np.ones(128), exposure=1, response=rsp, is_poisson=True)

    obs_spectrum.clone(new_counts=np.zeros_like(obs_spectrum.counts), new_count_errors=None)

    obs_spectrum.clone()
