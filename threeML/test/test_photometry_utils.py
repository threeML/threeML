import pytest
import speclite.filters as spec_filters
from astromodels import *
from threeML.utils.photometry.filter_set import FilterSet, NotASpeclikeFilter

from threeML.classicMLE.joint_likelihood import JointLikelihood
from threeML.data_list import DataList
from threeML.io.plotting.post_process_data_plots import (
    display_photometry_model_magnitudes,
)
from threeML.plugins.PhotometryLike import PhotometryLike
from threeML.utils.photometry.filter_library import threeML_filter_library


def get_plugin():

    grond = PhotometryLike(
        "GROND",
        filters=threeML_filter_library.ESO.GROND,
        g=(19.92, 0.1),
        r=(19.75, 0.1),
        i=(19.65, 0.1),
        z=(19.56, 0.1),
        J=(19.38, 0.1),
        H=(19.22, 0.1),
        K=(19.07, 0.1),
    )

    return grond


def get_model_and_datalist():

    grond = get_plugin()

    spec = Powerlaw()  # * XS_zdust() * XS_zdust()

    datalist = DataList(grond)

    model = Model(PointSource("grb", 0, 0, spectral_shape=spec))

    return model, datalist


def test_filter_set():

    sf = spec_filters.load_filters("bessell-*")

    fs1 = FilterSet(sf)

    # sf = spec_filters.load_filter('bessell-r')

    # fs2 = FilterSet(sf)

    with pytest.raises(NotASpeclikeFilter):

        fs2 = FilterSet("a")


def test_constructor():

    grond = PhotometryLike(
        "GROND",
        filters=threeML_filter_library.ESO.GROND,
        g=(19.92, 0.1),
        r=(19.75, 0.1),
        i=(19.65, 0.1),
        z=(19.56, 0.1),
        J=(19.38, 0.1),
        H=(19.22, 0.1),
        K=(19.07, 0.1),
    )

    assert not grond.is_poisson

    grond.display_filters()


def test_fit():

    model, datalist = get_model_and_datalist()

    jl = JointLikelihood(model, datalist)

    jl.fit()

    _ = display_photometry_model_magnitudes(jl)


# def test_filter_selection():
#
#     pi = get_plugin()
#
#     n_filters_original = sum(pi._mask)
#
#     original_fnames = pi._filter_set.filter_names
#
#     pi.set_inactive_filters(*original_fnames)
#
#     assert sum(pi._mask) == 0
#
#     pi.set_active_filters(*original_fnames)
#
#     assert sum(pi._mask) == n_filters_original
#
