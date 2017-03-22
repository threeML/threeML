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
                           g=(20.93, .23),
                           r=(19.96, 0.12),
                           i=(18.8, .07),
                           z=(17.82, .04),
                           J=(16.29, .03),
                           H=(15.28, .03),
                           K=(14.68, .04))

    grond.display_filters()

    spec = Powerlaw()  # * XS_zdust() * XS_zdust()

    data_list = DataList(grond)

    model = Model(PointSource('grb', 0, 0, spectral_shape=spec))

    jl = JointLikelihood(model, data_list)

    spec.piv = 1E-2

    jl.set_minimizer('ROOT')

    _ = jl.fit()

    _ = display_photometry_model_magnitudes(jl)


















