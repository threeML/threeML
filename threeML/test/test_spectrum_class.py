import pytest
from threeML.plugins.spectrum.binned_spectrum import BinnedSpectrum, ChannelSet
from threeML.plugins.SpectrumLike import SpectrumLike
from astromodels import Powerlaw, PointSource, Model

import numpy as np

import os

__this_dir__ = os.path.join(os.path.abspath(os.path.dirname(__file__)))


def test_spectrum_constructor():

    ebounds = ChannelSet.from_list_of_edges(np.array([0,1,2,3,4,5]))

    pl = Powerlaw()

    ps = PointSource('fake',0,0,spectral_shape=pl)

    model = Model(ps)

    obs_spectrum = BinnedSpectrum(counts=np.ones(len(ebounds)),exposure=1,ebounds=ebounds, is_poisson=True)
    bkg_spectrum = BinnedSpectrum(counts=np.ones(len(ebounds)),exposure=1,ebounds=ebounds, is_poisson=True)

    assert np.all(obs_spectrum.counts == obs_spectrum.rates)
    assert np.all(bkg_spectrum.counts == bkg_spectrum.rates)



    specLike = SpectrumLike('fake', observed_spectrum=obs_spectrum, background_spectrum=bkg_spectrum)
    specLike.set_model(model)
    specLike.get_model()

    specLike.get_simulated_dataset()


    specLike.significance
    specLike.significance_per_channel

    obs_spectrum = BinnedSpectrum(counts=np.ones(len(ebounds)), count_errors=np.ones(len(ebounds)),exposure=1, ebounds=ebounds, is_poisson=False)
    bkg_spectrum = BinnedSpectrum(counts=np.ones(len(ebounds)), exposure=1, ebounds=ebounds, is_poisson=True)


    with pytest.raises(NotImplementedError):

        specLike = SpectrumLike('fake', observed_spectrum=obs_spectrum, background_spectrum=bkg_spectrum)






