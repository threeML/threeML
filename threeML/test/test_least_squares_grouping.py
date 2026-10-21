import os

import numpy as np
import numpy.testing as npt
import pytest
from astromodels import Blackbody, Line, Model, PointSource, Powerlaw

from threeML import JointLikelihood, LocalMinimization
from threeML.data_list import DataList
from threeML.io.file_utils import within_directory
from threeML.minimizer.minimization import FitFailed, ProfileLikelihood
from threeML.minimizer.tutorial_material import (
    get_joint_likelihood_object_simple_likelihood,
)
from threeML.plugins.OGIPLike import OGIPLike
from threeML.plugins.SpectrumLike import SpectrumLike
from threeML.plugins.XYLike import XYLike
from threeML.utils.binner import Rebinner

from .conftest import get_test_datasets_directory

__example_dir = get_test_datasets_directory()


def _energy_edges():
    energies = np.logspace(1, 3, 51)

    return energies[:-1], energies[1:]


def _get_spectrum_plugin(kind):
    """Simulated spectra covering all the statistics with deviance
    residuals."""

    low_edge, high_edge = _energy_edges()

    source_function = Blackbody(K=1e-1, kT=20.0)
    background_function = Powerlaw(K=1, index=-1.5, piv=100.0)

    n_bins = len(low_edge)

    if kind == "poisson_poisson":
        plugin = SpectrumLike.from_function(
            "fake",
            source_function=source_function,
            background_function=background_function,
            energy_min=low_edge,
            energy_max=high_edge,
        )

    elif kind == "poisson_gaussian":
        plugin = SpectrumLike.from_function(
            "fake",
            source_function=source_function,
            background_function=background_function,
            background_errors=np.ones(n_bins) * 5.0,
            energy_min=low_edge,
            energy_max=high_edge,
        )

    elif kind == "poisson_ideal":
        plugin = SpectrumLike.from_function(
            "fake",
            source_function=source_function,
            background_function=background_function,
            energy_min=low_edge,
            energy_max=high_edge,
        )

        plugin.background_noise_model = "ideal"

    elif kind == "poisson_none":
        plugin = SpectrumLike.from_function(
            "fake",
            source_function=source_function,
            energy_min=low_edge,
            energy_max=high_edge,
        )

    elif kind == "gaussian":
        plugin = SpectrumLike.from_function(
            "fake",
            source_function=source_function,
            source_errors=np.ones(n_bins) * 10.0,
            energy_min=low_edge,
            energy_max=high_edge,
        )

    else:
        raise ValueError(kind)

    return plugin


def _get_jl(plugin):
    model = Model(PointSource("fake", 0, 0, spectral_shape=Blackbody()))

    return JointLikelihood(model, DataList(plugin), verbose=False)


_kinds = [
    "poisson_poisson",
    "poisson_gaussian",
    "poisson_ideal",
    "poisson_none",
    "gaussian",
]


@pytest.mark.parametrize("kind", _kinds)
def test_deviance_residuals_match_likelihood(kind):
    plugin = _get_spectrum_plugin(kind)
    jl = _get_jl(plugin)

    free_parameters = list(jl.likelihood_model.free_parameters.values())

    differences = []

    for K, kT in [(0.1, 20.0), (0.05, 30.0), (0.3, 10.0), (1.0, 50.0)]:
        free_parameters[0].value = K
        free_parameters[1].value = kT

        internal = [p._get_internal_value() for p in free_parameters]

        residuals = jl.deviance_residuals_profile(*internal)
        minus_log_like = jl.minus_log_like_profile(*internal)

        assert residuals.shape == (plugin.n_data_points,)

        differences.append(0.5 * residuals @ residuals - minus_log_like)

    # The sum of the squares of the residuals is -2 logL up to a constant

    npt.assert_allclose(differences, differences[0], rtol=0, atol=1e-6)


@pytest.mark.parametrize("kind", _kinds)
def test_saturated_model_is_maximum(kind):
    plugin = _get_spectrum_plugin(kind)
    _get_jl(plugin)

    evaluator = plugin._likelihood_evaluator

    saturated = evaluator._saturated_model_counts()
    saturated_log_like = evaluator._per_bin_log_like(saturated)

    for factor in np.linspace(0, 3, 31):
        for shift in (0.0, 1.0, 5.0):
            trial = evaluator._per_bin_log_like(saturated * factor + shift)

            assert np.all(trial <= saturated_log_like + 1e-9)


@pytest.mark.parametrize("kind", _kinds)
def test_least_squares_matches_minuit(kind):
    # Simulate once, so that both minimizers fit the same data

    plugin = _get_spectrum_plugin(kind)

    results = {}

    for minimizer in ("minuit", "least_squares"):
        jl = _get_jl(plugin)
        jl.set_minimizer(minimizer)

        fit_results, like_frame = jl.fit(quiet=True)

        results[minimizer] = (fit_results, like_frame.loc["total"].iloc[0])

    minuit_results, minuit_like = results["minuit"]
    ls_results, ls_like = results["least_squares"]

    npt.assert_allclose(
        ls_results["value"].values, minuit_results["value"].values, rtol=1e-3
    )

    # Errors from J^T J are a Gauss-Newton approximation, so they agree only
    # approximately with those from the Hessian

    npt.assert_allclose(
        ls_results["error"].values, minuit_results["error"].values, rtol=0.2
    )

    assert ls_like == pytest.approx(minuit_like, abs=1e-3)


def test_least_squares_setup_and_profiling():
    jl = _get_jl(_get_spectrum_plugin("poisson_poisson"))

    minimizer = LocalMinimization("least_squares")
    minimizer.setup(diff_step=0.001, ftol=1e-10, algorithm="trf")

    jl.set_minimizer(minimizer)

    fit_results, _ = jl.fit(quiet=True)

    errors = jl.get_errors(quiet=True)

    assert np.all(errors["negative_error"].values < 0)
    assert np.all(errors["positive_error"].values > 0)

    best_kT = fit_results["value"].values[1]

    kT = jl.likelihood_model.fake.spectrum.main.Blackbody.kT

    _, _, profile, *_ = jl.get_contours(
        kT, best_kT * 0.9, best_kT * 1.1, 5, progress=False
    )

    # The profile has its minimum in the middle of the scan

    assert np.argmin(profile) == 2

    with pytest.raises(AssertionError):
        minimizer.setup(not_a_valid_key=1)

    with pytest.raises(AssertionError):
        minimizer.set_algorithm("lm")
        jl.set_minimizer(minimizer)
        jl.fit(quiet=True)


def test_profiling_keeps_minimizer_setup():
    # The minimizers used to profile the likelihood (get_errors, get_contours)
    # must have the same setup as the one used for the fit

    least_squares = LocalMinimization("least_squares")
    least_squares.setup(diff_step=0.003, algorithm="dogbox")

    minuit = LocalMinimization("minuit")
    minuit.setup(ftol=0.5)

    for minimization in (least_squares, minuit):
        jl = _get_jl(_get_spectrum_plugin("poisson_poisson"))
        jl.set_minimizer(minimization)
        jl.fit(quiet=True)

        name = list(jl.likelihood_model.free_parameters.keys())[0]

        profile = ProfileLikelihood(jl._minimizer, [name])

        if minimization is least_squares:
            assert profile._optimizer._setup_dict["diff_step"] == 0.003
            assert profile._optimizer._setup_dict["algorithm"] == "dogbox"

        else:
            assert profile._optimizer.minuit.tol == 0.5


def test_least_squares_requires_deviance_residuals():
    # The plugin of the minimization tutorial does not provide residuals

    jl, _ = get_joint_likelihood_object_simple_likelihood()
    jl.set_minimizer("least_squares")

    with pytest.raises((NotImplementedError, FitFailed)):
        jl.fit(quiet=True)


def _get_xy_jl(poisson):
    x = np.linspace(1, 10, 30)

    line = Line(a=2.0, b=10.0)

    if poisson:
        xy = XYLike.from_function("xy", line, x, poisson_data=True)

    else:
        xy = XYLike.from_function("xy", line, x, yerr=np.ones_like(x) * 2.0)

    model = Model(PointSource("xy", 0, 0, spectral_shape=Line()))

    return JointLikelihood(model, DataList(xy), verbose=False), xy


@pytest.mark.parametrize("poisson", [False, True])
def test_xylike_deviance_residuals(poisson):
    jl, xy = _get_xy_jl(poisson)

    free_parameters = list(jl.likelihood_model.free_parameters.values())

    differences = []

    for a, b in [(2.0, 10.0), (1.0, 12.0), (3.0, 5.0)]:
        free_parameters[0].value = b
        free_parameters[1].value = a

        internal = [p._get_internal_value() for p in free_parameters]

        residuals = jl.deviance_residuals_profile(*internal)

        assert residuals.shape == (xy.get_number_of_data_points(),)

        differences.append(
            0.5 * residuals @ residuals - jl.minus_log_like_profile(*internal)
        )

    npt.assert_allclose(differences, differences[0], rtol=0, atol=1e-6)

    # Same fit as MINUIT

    results = {}

    for minimizer in ("minuit", "least_squares"):
        jl.set_minimizer(minimizer)
        results[minimizer], _ = jl.fit(quiet=True)

    npt.assert_allclose(
        results["least_squares"]["value"].values,
        results["minuit"]["value"].values,
        rtol=1e-3,
    )


def test_rebinner_from_ogip_grouping():
    grouping = np.array([1, -1, -1, 1, 1, -1, 0, 1, -1, -1])

    rebinner = Rebinner.from_ogip_grouping(grouping)

    assert rebinner.n_bins == 5

    vector = np.arange(10, dtype=float)

    (rebinned,) = rebinner.rebin(vector)

    npt.assert_array_equal(rebinned, [0 + 1 + 2, 3, 4 + 5, 6, 7 + 8 + 9])

    npt.assert_array_equal(rebinner.grouping, [1, -1, -1, 1, 1, -1, 1, 1, -1, -1])

    # The mask removes channels, splitting the groups at its edges

    mask = np.ones(10, dtype=bool)
    mask[[0, 8]] = False

    rebinner = Rebinner.from_ogip_grouping(grouping, mask)

    (rebinned,) = rebinner.rebin(vector)

    npt.assert_array_equal(rebinned, [1 + 2, 3, 4 + 5, 6, 7, 9])

    with pytest.raises(ValueError):
        Rebinner.from_ogip_grouping(np.array([1, 2, -1]))


def test_rebinner_writes_ogip_grouping():
    # 1 at the first channel of each group, -1 for the following ones

    rebinner = Rebinner(np.array([1, 1, 1, 5, 1, 1, 1]), 3)

    (rebinned,) = rebinner.rebin(np.array([1, 1, 1, 5, 1, 1, 1]))

    npt.assert_array_equal(rebinned, [3, 5, 3])

    npt.assert_array_equal(rebinner.grouping, [1, -1, -1, 1, 1, -1, -1])


def test_rebin_on_grouping_round_trip():
    # Spectra not read from a PHA file have no grouping to use

    with pytest.raises(RuntimeError):
        _get_spectrum_plugin("poisson_poisson").rebin_on_grouping()

    with within_directory(__example_dir):
        ogip = OGIPLike("test_ogip", observation="test.pha{1}")

        # The GROUPING column of this file contains only zeros (no grouping
        # information), so grouping on it has no effect

        npt.assert_array_equal(ogip.get_pha_files()["pha"].file_grouping, 0)

        ogip.set_active_measurements("all")

        n_channels = ogip.n_data_points

        ogip.rebin_on_grouping()

        assert ogip.n_data_points == n_channels
        assert ogip._rebinner is None

        # Grouping with a wrong number of channels is not allowed

        with pytest.raises(RuntimeError):
            ogip.rebin_on_grouping(np.ones(n_channels + 1))

        ogip.rebin_on_background(min_number_of_counts=100)

        n_bins = ogip.n_data_points
        rebinned_counts = ogip.current_observed_counts

        assert ogip.get_number_of_data_points() == n_bins

        try:
            ogip.write_pha("test_grouping_write", overwrite=True)

            written = OGIPLike("written", observation="test_grouping_write.pha{1}")

            file_grouping = written.get_pha_files()["pha"].file_grouping

            npt.assert_array_equal(file_grouping, ogip.grouping)

            written.set_active_measurements("all")
            written.rebin_on_grouping()

            assert written.n_data_points == n_bins

            npt.assert_array_equal(written.current_observed_counts, rebinned_counts)

            # A custom grouping can also be provided

            written.remove_rebinning()

            custom = np.ones(len(file_grouping), dtype=int)
            custom[1::2] = -1

            written.rebin_on_grouping(custom)

            assert written.n_data_points == len(file_grouping) // 2

            with pytest.raises(RuntimeError):
                written.set_active_measurements("all")

        finally:
            for file_name in ("test_grouping_write.pha", "test_grouping_write_bak.pha"):
                if os.path.exists(file_name):
                    os.remove(file_name)
