import pytest
from threeML import *
from threeML.utils.binner import NotEnoughData


def test_OGIP_plotting(fitted_joint_likelihood_bn090217206_nai):

    jl, _, _ = fitted_joint_likelihood_bn090217206_nai

    NaI6 = jl.data_list["NaI6"]

    # OGIP channel plotting

    NaI6.view_count_spectrum(plot_errors=True, show_bad_channels=True)

    NaI6.view_count_spectrum(plot_errors=False, show_bad_channels=True)

    NaI6.view_count_spectrum(plot_errors=True, show_bad_channels=False)

    NaI6.view_count_spectrum(plot_errors=False, show_bad_channels=False)

    _ = display_spectrum_model_counts(jl)

    _ = display_spectrum_model_counts(jl, data=("NaI6"))

    _ = display_spectrum_model_counts(jl, data=("wrong"))

    _ = display_spectrum_model_counts(jl, min_rate=1e-8)

    with pytest.raises(NotEnoughData):

        _ = display_spectrum_model_counts(jl, min_rate=1e8)
