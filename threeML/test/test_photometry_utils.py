import pytest
import speclite.filters as spec_filters
import numpy as np

from threeML.classicMLE.joint_likelihood import JointLikelihood
from threeML.io.plotting.post_process_data_plots import \
    display_photometry_model_magnitudes

from threeML.utils.photometry.filter_set import FilterSet, NotASpeclikeFilter




def test_filter_set():

    sf = spec_filters.load_filters("bessell-*")

    fs1 = FilterSet(sf)

    # sf = spec_filters.load_filter('bessell-r')

    # fs2 = FilterSet(sf)

    with pytest.raises(NotASpeclikeFilter):

        fs2 = FilterSet("a")


def test_constructor(grond_plugin):

    assert not grond_plugin.is_poisson

    grond_plugin.display_filters()

    assert grond_plugin._mask.sum() == 7

    grond_plugin.band_g.on = False

    assert grond_plugin._mask.sum() == 6


    grond_plugin.band_g.on = True

    assert grond_plugin._mask.sum() == 7


    grond_plugin.band_g.off = True

    assert grond_plugin._mask.sum() == 6


    grond_plugin.band_g.off = False

    assert grond_plugin._mask.sum() == 7


    

def test_fit(photometry_data_model):

    model, datalist = photometry_data_model

    jl = JointLikelihood(model, datalist)

    jl.fit()

    _ = display_photometry_model_magnitudes(jl)

    np.testing.assert_allclose([model.grb.spectrum.main.Powerlaw.K.value,model.grb.spectrum.main.Powerlaw.index.value], [0.00296,-1.505936], rtol=1e-3)
    
