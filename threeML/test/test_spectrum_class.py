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
