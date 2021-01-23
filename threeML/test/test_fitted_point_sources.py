from builtins import zip
import pytest
from threeML import *
from threeML.plugins.OGIPLike import OGIPLike
from threeML.utils.fitted_objects.fitted_point_sources import InvalidUnitError
from threeML.io.calculate_flux import _calculate_point_source_flux
import astropy.units as u
import matplotlib.pyplot as plt
import os

from threeML.io.package_data import get_path_of_data_dir


# Init some globals

datadir = os.path.abspath(
    os.path.join(get_path_of_data_dir(), "datasets", "bn090217206")
)

good_d_flux_units = ["1/(cm2 s keV)", "erg/(cm2 s keV)", "erg2/(cm2 s keV)"]

good_i_flux_units = ["1/(cm2 s )", "erg/(cm2 s )", "erg2/(cm2 s )"]


good_energy_units = ["keV", "Hz", "nm"]


bad_flux_units = ["g"]


def make_simple_model():

    triggerName = "bn090217206"
    ra = 204.9
    dec = -8.4

    powerlaw = Powerlaw()
    GRB = PointSource(triggerName, ra, dec, spectral_shape=powerlaw)
    model = Model(GRB)

    powerlaw.index.prior = Uniform_prior(lower_bound=-5.0, upper_bound=5.0)
    powerlaw.K.prior = Log_uniform_prior(lower_bound=1.0, upper_bound=10)

    return model


def make_components_model():

    triggerName = "bn090217206"
    ra = 204.9
    dec = -8.4

    powerlaw = Powerlaw() + Blackbody()
    GRB = PointSource(triggerName, ra, dec, spectral_shape=powerlaw)
    model = Model(GRB)

    powerlaw.index_1.prior = Uniform_prior(lower_bound=-5.0, upper_bound=5.0)
    powerlaw.K_1.prior = Log_uniform_prior(lower_bound=1.0, upper_bound=10)

    powerlaw.K_2.prior = Uniform_prior(lower_bound=-5.0, upper_bound=5.0)
    powerlaw.kT_2.prior = Log_uniform_prior(lower_bound=1.0, upper_bound=10)

    return model


def make_dless_components_model():

    triggerName = "bn090217206"
    ra = 204.9
    dec = -8.4

    powerlaw = Powerlaw() * Constant()
    GRB = PointSource(triggerName, ra, dec, spectral_shape=powerlaw)
    model = Model(GRB)

    powerlaw.index_1.prior = Uniform_prior(lower_bound=-5.0, upper_bound=5.0)
    powerlaw.K_1.prior = Log_uniform_prior(lower_bound=1.0, upper_bound=10)
    powerlaw.k_2 = 1.0
    powerlaw.k_2.fix = True

    return model


@pytest.fixture
def analysis_to_test(data_list_bn090217206_nai6):

    simple_model = make_simple_model()

    complex_model = make_components_model()
    # prepare mle

    dless_model = make_dless_components_model()

    jl_simple = JointLikelihood(simple_model, data_list_bn090217206_nai6)

    jl_simple.fit()

    jl_complex = JointLikelihood(complex_model, data_list_bn090217206_nai6)

    jl_complex.fit()

    jl_dless = JointLikelihood(dless_model, data_list_bn090217206_nai6)

    jl_dless.fit()

    bayes_simple = BayesianAnalysis(simple_model, data_list_bn090217206_nai6)

    bayes_simple.set_sampler("emcee")
    bayes_simple.sampler.setup(n_iterations=10, n_burn_in=10, n_walkers=20)
    bayes_simple.sample()

    bayes_complex = BayesianAnalysis(complex_model, data_list_bn090217206_nai6)

    bayes_complex.set_sampler("emcee")

    bayes_complex.sampler.setup(n_iterations=10, n_burn_in=10, n_walkers=20)

    bayes_complex.sample()

    bayes_dless = BayesianAnalysis(dless_model, data_list_bn090217206_nai6)

    bayes_dless.set_sampler("emcee")

    bayes_dless.sampler.setup(n_iterations=10, n_burn_in=10, n_walkers=20)

    bayes_dless.sample()

    analysis_to_test = [
        jl_simple.results,
        jl_complex.results,
        jl_dless.results,
        bayes_simple.results,
        bayes_complex.results,
        bayes_dless.results,
    ]

    return analysis_to_test


def test_fitted_point_source_plotting(analysis_to_test):

    plot_keywords = {
        "use_components": True,
        "components_to_use": ["Powerlaw", "total"],
        "sources_to_use": ["bn090217206"],
        "flux_unit": "erg/(cm2 s)",
        "energy_unit": "keV",
        "plot_style_kwargs": {},
        "contour_style_kwargs": {},
        "legend_kwargs": {},
        "ene_min": 10,
        "ene_max": 100,
        "num_ene": 5,
        "show_legend": False,
        "fit_cmap": "jet",
        "countor_cmap": "jet",
        "sum_sources": True,
    }

    for u1, u2 in zip(good_d_flux_units, good_i_flux_units):

        for e_unit in good_energy_units:

            for x in analysis_to_test:

                _ = plot_point_source_spectra(
                    x, flux_unit=u1, energy_unit=e_unit, num_ene=5
                )

                _ = plot_point_source_spectra(x, **plot_keywords)

                with pytest.raises(InvalidUnitError):
                    _ = plot_point_source_spectra(x, flux_unit=bad_flux_units[0])

            plt.close("all")


def test_fitted_point_source_flux_calculations(analysis_to_test):

    flux_keywords = {
        "use_components": True,
        "components_to_use": ["total", "Powerlaw"],
        "sources_to_use": ["bn090217206"],
        "flux_unit": "erg/(cm2 s)",
        "energy_unit": "keV",
        "sum_sources": True,
    }

    _calculate_point_source_flux(
        1, 10, analysis_to_test[0], flux_unit=good_i_flux_units[0], energy_unit="keV"
    )

    _calculate_point_source_flux(1, 10, analysis_to_test[-2], **flux_keywords)


def test_units_on_energy_range(analysis_to_test):

    _ = plot_point_source_spectra(
        analysis_to_test[0], ene_min=1.0 * u.keV, ene_max=1 * u.MeV
    )

    with pytest.raises(AssertionError):
        plot_point_source_spectra(analysis_to_test[0], ene_min=1.0, ene_max=1 * u.MeV)

    with pytest.raises(AssertionError):
        plot_point_source_spectra(analysis_to_test[0], ene_min=1.0 * u.keV, ene_max=1.0)
