from pathlib import Path

import pytest

from threeML import *
from threeML.io.network import internet_connection_is_active

skip_if_internet_is_not_available = pytest.mark.skipif(
    not internet_connection_is_active(), reason="No active internet connection"
)


def test_basic_analysis_results(fitted_joint_likelihood_bn090217206_nai):

    jl, fit_results, like_frame = fitted_joint_likelihood_bn090217206_nai

    jl.restore_best_fit()

    expected = [2.531028, -1.1831566000728451]

    assert np.allclose(fit_results["value"], expected, rtol=0.1)


def test_basic_analysis_get_errors(fitted_joint_likelihood_bn090217206_nai):

    jl, fit_results, like_frame = fitted_joint_likelihood_bn090217206_nai

    jl.restore_best_fit()

    err = jl.get_errors()

    assert np.allclose(err["negative_error"], [-0.196, -0.0148], rtol=1e-1)


def test_basic_analysis_contour_1d(fitted_joint_likelihood_bn090217206_nai):

    jl, fit_results, like_frame = fitted_joint_likelihood_bn090217206_nai

    jl.restore_best_fit()

    powerlaw = jl.likelihood_model.bn090217206.spectrum.main.Powerlaw

    res = jl.get_contours(powerlaw.index, -1.3, -1.1, 20)

    expected_result = np.array(
        [
            -1.3,
            -1.28947368,
            -1.27894737,
            -1.26842105,
            -1.25789474,
            -1.24736842,
            -1.23684211,
            -1.22631579,
            -1.21578947,
            -1.20526316,
            -1.19473684,
            -1.18421053,
            -1.17368421,
            -1.16315789,
            -1.15263158,
            -1.14210526,
            -1.13157895,
            -1.12105263,
            -1.11052632,
            -1.1,
        ]
    )

    assert np.allclose(res[0], expected_result, rtol=0.1)


def test_basic_analysis_contour_2d(fitted_joint_likelihood_bn090217206_nai):

    jl, fit_results, like_frame = fitted_joint_likelihood_bn090217206_nai

    jl.restore_best_fit()

    powerlaw = jl.likelihood_model.bn090217206.spectrum.main.Powerlaw

    res = jl.get_contours(powerlaw.index, -1.25, -1.1, 30, powerlaw.K, 1.8, 3.4, 30)

    exp_p1, exp_p2 = (
        np.array(
            [
                -1.25,
                -1.24482759,
                -1.23965517,
                -1.23448276,
                -1.22931034,
                -1.22413793,
                -1.21896552,
                -1.2137931,
                -1.20862069,
                -1.20344828,
                -1.19827586,
                -1.19310345,
                -1.18793103,
                -1.18275862,
                -1.17758621,
                -1.17241379,
                -1.16724138,
                -1.16206897,
                -1.15689655,
                -1.15172414,
                -1.14655172,
                -1.14137931,
                -1.1362069,
                -1.13103448,
                -1.12586207,
                -1.12068966,
                -1.11551724,
                -1.11034483,
                -1.10517241,
                -1.1,
            ]
        ),
        np.array(
            [
                1.8,
                1.85517241,
                1.91034483,
                1.96551724,
                2.02068966,
                2.07586207,
                2.13103448,
                2.1862069,
                2.24137931,
                2.29655172,
                2.35172414,
                2.40689655,
                2.46206897,
                2.51724138,
                2.57241379,
                2.62758621,
                2.68275862,
                2.73793103,
                2.79310345,
                2.84827586,
                2.90344828,
                2.95862069,
                3.0137931,
                3.06896552,
                3.12413793,
                3.17931034,
                3.23448276,
                3.28965517,
                3.34482759,
                3.4,
            ]
        ),
    )

    assert np.allclose(res[0], exp_p1, rtol=0.1)
    assert np.allclose(res[1], exp_p2, rtol=0.1)


def test_basic_bayesian_analysis_results(completed_bn090217206_bayesian_analysis):

    bayes, samples = completed_bn090217206_bayesian_analysis

    expected = (2.3224550250817337, 2.73429304662902)

    res = bayes.results.get_equal_tailed_interval(
        "bn090217206.spectrum.main.Powerlaw.K"
    )

    assert np.allclose(res, expected, rtol=0.1)


def test_basic_analsis_multicomp_results(
    fitted_joint_likelihood_bn090217206_nai_multicomp,
):

    jl, fit_results, like_frame = fitted_joint_likelihood_bn090217206_nai_multicomp

    jl.restore_best_fit()

    expected = np.array([1.88098173e00, -1.20057690e00, 6.50915964e-06, 4.35643006e01])

    assert np.allclose(fit_results["value"].values, expected, rtol=0.1)


def test_basic_bayesian_analysis_results_multicomp(
    completed_bn090217206_bayesian_analysis_multicomp,
):

    bayes, samples = completed_bn090217206_bayesian_analysis_multicomp

    frame = bayes.results.get_data_frame()

    expected_central_values = np.array(
        [1.90814527e00, -1.20941618e00, 6.45755638e-06, 4.36948057e01]
    )
    expected_negative_errors = np.array(
        [-3.02301749e-01, -2.93259914e-02, -1.70958890e-06, -3.92505021e00]
    )
    expected_positive_errors = np.array(
        [2.65259894e-01, 3.24980566e-02, 1.78051424e-06, 4.00921638e00]
    )

    assert np.allclose(frame["value"].values, expected_central_values, rtol=0.1)
    assert np.allclose(
        frame["negative_error"].values, expected_negative_errors, rtol=0.1
    )
    assert np.allclose(
        frame["positive_error"].values, expected_positive_errors, rtol=0.1
    )


@skip_if_internet_is_not_available
def test_gbm_workflow():

    import warnings

    warnings.simplefilter("ignore")

    gbm_catalog = FermiGBMBurstCatalog()
    gbm_catalog.query_sources("GRB080916009")

    grb_info = gbm_catalog.get_detector_information()["GRB080916009"]

    gbm_detectors = grb_info["detectors"]
    source_interval = grb_info["source"]["fluence"]
    background_interval = grb_info["background"]["full"]
    best_fit_model = grb_info["best fit model"]["fluence"]
    model = gbm_catalog.get_model(best_fit_model, "fluence")["GRB080916009"]

    dload = download_GBM_trigger_data("bn080916009", detectors=gbm_detectors)

    fluence_plugins = []
    time_series = {}

    for det in gbm_detectors:

        ts_cspec = TimeSeriesBuilder.from_gbm_cspec_or_ctime(
            det, cspec_or_ctime_file=dload[det]["cspec"], rsp_file=dload[det]["rsp"]
        )

        ts_cspec.set_background_interval(*background_interval.split(","))
        ts_cspec.save_background(f"{det}_bkg.h5", overwrite=True)

        ts_cspec.write_pha_from_binner(
            "test_write", start=0, stop=10, overwrite=True, force_rsp_write=True
        )

        ts_tte = TimeSeriesBuilder.from_gbm_tte(
            det,
            tte_file=dload[det]["tte"],
            rsp_file=dload[det]["rsp"],
            restore_background=f"{det}_bkg.h5",
        )

        time_series[det] = ts_tte

        ts_tte.set_active_time_interval(source_interval)

        ts_tte.view_lightcurve(-40, 100)

        fluence_plugin = ts_tte.to_spectrumlike()

        if det.startswith("b"):

            fluence_plugin.set_active_measurements("250-30000")

        else:

            fluence_plugin.set_active_measurements("9-900")

        fluence_plugin.rebin_on_background(1.0)

        fluence_plugins.append(fluence_plugin)

        ts_tte.create_time_bins(start=0, stop=10, method="constant", dt=1)

    # clean up
    p = Path(".")

    dl_files = p.glob("glg*080916009*")

    [x.unlink() for x in dl_files]

    dl_files = p.glob("test_write*")

    [x.unlink() for x in dl_files]

    dl_files = p.glob("*_bkg.h5")

    [x.unlink() for x in dl_files]
