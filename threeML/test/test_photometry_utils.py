import pytest
from astromodels import *
from threeML.classicMLE.joint_likelihood import JointLikelihood
from threeML.data_list import DataList
from threeML.io.plotting.post_process_data_plots import display_photometry_model_magnitudes
from threeML.plugins.photometry.filter_set import FilterSet, NotASpeclikeFilter
from threeML.plugins.photometry.filter_library import threeML_filter_library
import speclite.filters as spec_filters
from threeML.plugins.PhotometryLike import PhotometryLike

def test_filter_set():

    sf = spec_filters.load_filters('bessell-*')

    fs1 = FilterSet(sf)

    #sf = spec_filters.load_filter('bessell-r')

    #fs2 = FilterSet(sf)

    with pytest.raises(NotASpeclikeFilter):

        fs2 = FilterSet('a')


    threeML_filter_library.instruments



def test_photo_plugin():


    grond = PhotometryLike('GROND',
                           filters=threeML_filter_library.ESO.GROND,
                           g=(19.92, .1),
                           r=(19.75, .1),
                           i=(19.65, .1),
                           z=(19.56, .1),
                           J=(19.38, .1),
                           H=(19.22, .1),
                           K=(19.07, .1))

    grond.display_filters()

    spec = Powerlaw()  # * XS_zdust() * XS_zdust()

    data_list = DataList(grond)

    model = Model(PointSource('grb', 0, 0, spectral_shape=spec))

    jl = JointLikelihood(model, data_list)

    spec.piv = 1E0
    spec.K.min_value = 0.

    #jl.set_minimizer('ROOT')

    _ = jl.fit()

    _ = display_photometry_model_magnitudes(jl)


















