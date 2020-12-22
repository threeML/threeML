from builtins import range
from builtins import object
import pytest
import os
import numpy.testing as npt
from .conftest import get_test_datasets_directory
from threeML import *
from threeML.io.file_utils import within_directory
from threeML.plugins.OGIPLike import OGIPLike
from threeML.plugins.SwiftXRTLike import SwiftXRTLike
from threeML.utils.OGIP.response import OGIPResponse
from threeML.utils.spectrum.pha_spectrum import PHASpectrum
from threeML.utils.statistics.likelihood_functions import *

__this_dir__ = os.path.join(os.path.abspath(os.path.dirname(__file__)))
__example_dir = get_test_datasets_directory()


class AnalysisBuilder(object):
    def __init__(self, plugin):
        self._plugin = plugin

        self._shapes = {}
        self._shapes["normal"] = Powerlaw
        self._shapes["cpl"] = Cutoff_powerlaw

    @property
    def keys(self):
        return list(self._shapes.keys())

    def get_jl(self, key):
        assert key in self._shapes

        data_list = DataList(self._plugin)

        ps = PointSource("test", 0, 0, spectral_shape=self._shapes[key]())
        model = Model(ps)
        jl = JointLikelihood(model, data_list, verbose=False)
        jl.set_minimizer("minuit")

        return jl


def test_loading_a_generic_pha_file():

    with within_directory(__example_dir):
        ogip = OGIPLike("test_ogip", observation="test.pha{1}")

        pha_info = ogip.get_pha_files()

        assert ogip.name == "test_ogip"
        assert ogip.n_data_points == sum(ogip._mask)
        assert sum(ogip._mask) == ogip.n_data_points
        assert ogip.tstart == 0.0
        assert ogip.tstop == 9.95012
        assert "cons_test_ogip" in ogip.nuisance_parameters
        assert ogip.nuisance_parameters["cons_test_ogip"].fix == True
        assert ogip.nuisance_parameters["cons_test_ogip"].free == False

        assert "pha" in pha_info
        assert "bak" in pha_info
        assert "rsp" in pha_info

        ogip.__repr__()


def test_loading_a_loose_ogip_pha_file():

    with within_directory(__example_dir):
        ogip = OGIPLike("test_ogip", observation="example_integral.pha")

        pha_info = ogip.get_pha_files()

        assert ogip.name == "test_ogip"
        assert ogip.n_data_points == sum(ogip._mask)
        assert sum(ogip._mask) == ogip.n_data_points
        # assert ogip.tstart is None
        # assert ogip.tstop is None
        assert "cons_test_ogip" in ogip.nuisance_parameters
        assert ogip.nuisance_parameters["cons_test_ogip"].fix == True
        assert ogip.nuisance_parameters["cons_test_ogip"].free == False

        assert "pha" in pha_info
        # assert 'bak' in pha_info
        assert "rsp" in pha_info

        ogip.__repr__()


def test_loading_bad_keywords_file():

    with within_directory(__example_dir):
        pha_fn = "example_integral_spi.pha"
        rsp_fn = "example_integral_spi.rsp"

        pha_spectrum = PHASpectrum(pha_fn, rsp_file=rsp_fn)

        assert type(pha_spectrum.is_poisson) == bool

        ogip = OGIPLike("test_ogip", observation=pha_fn, response=rsp_fn)
        ogip.__repr__()


def test_pha_files_in_generic_ogip_constructor_spec_number_in_file_name():
    with within_directory(__example_dir):

        ogip = OGIPLike("test_ogip", observation="test.pha{1}")
        ogip.set_active_measurements("all")
        pha_info = ogip.get_pha_files()

        for key in ["pha", "bak"]:

            assert isinstance(pha_info[key], PHASpectrum)

        assert pha_info["pha"].background_file == "test_bak.pha{1}"
        assert pha_info["pha"].ancillary_file is None
        assert pha_info["pha"].instrument == "GBM_NAI_03"
        assert pha_info["pha"].mission == "GLAST"
        assert pha_info["pha"].is_poisson == True
        assert pha_info["pha"].n_channels == ogip.n_data_points
        assert pha_info["pha"].n_channels == len(pha_info["pha"].rates)

        # Test that Poisson rates cannot call rate error
        assert pha_info["pha"].rate_errors is None

        assert (
            sum(pha_info["pha"].sys_errors == np.zeros_like(pha_info["pha"].rates))
            == pha_info["bak"].n_channels
        )

        assert (
            pha_info["pha"].response_file.split("/")[-1]
            == "glg_cspec_n3_bn080916009_v07.rsp"
        )
        assert pha_info["pha"].scale_factor == 1.0

        assert pha_info["bak"].background_file is None

        # Test that we cannot get a bak file
        #
        #
        # with pytest.raises(KeyError):
        #
        #     _ = pha_info['bak'].background_file

        # Test that we cannot get a anc file
        # with pytest.raises(KeyError):
        #
        #     _ = pha_info['bak'].ancillary_file

        # Test that we cannot get a RSP file

        assert pha_info["bak"].response_file is None

        assert pha_info["bak"].ancillary_file is None

        # with pytest.raises(AttributeError):
        #      _ = pha_info['bak'].response_file

        assert pha_info["bak"].instrument == "GBM_NAI_03"
        assert pha_info["bak"].mission == "GLAST"

        assert pha_info["bak"].is_poisson == False

        assert pha_info["bak"].n_channels == ogip.n_data_points
        assert pha_info["bak"].n_channels == len(pha_info["pha"].rates)

        assert len(pha_info["bak"].rate_errors) == pha_info["bak"].n_channels

        assert (
            sum(pha_info["bak"].sys_errors == np.zeros_like(pha_info["pha"].rates))
            == pha_info["bak"].n_channels
        )

        assert pha_info["bak"].scale_factor == 1.0

        assert isinstance(pha_info["rsp"], OGIPResponse)


def test_pha_files_in_generic_ogip_constructor_spec_number_in_arguments():
    with within_directory(__example_dir):
        ogip = OGIPLike("test_ogip", observation="test.pha", spectrum_number=1)
        ogip.set_active_measurements("all")

        pha_info = ogip.get_pha_files()

        for key in ["pha", "bak"]:

            assert isinstance(pha_info[key], PHASpectrum)

        assert pha_info["pha"].background_file == "test_bak.pha{1}"
        assert pha_info["pha"].ancillary_file is None
        assert pha_info["pha"].instrument == "GBM_NAI_03"
        assert pha_info["pha"].mission == "GLAST"
        assert pha_info["pha"].is_poisson == True
        assert pha_info["pha"].n_channels == ogip.n_data_points
        assert pha_info["pha"].n_channels == len(pha_info["pha"].rates)

        # Test that Poisson rates cannot call rate error
        assert pha_info["pha"].rate_errors is None

        assert (
            sum(pha_info["pha"].sys_errors == np.zeros_like(pha_info["pha"].rates))
            == pha_info["bak"].n_channels
        )
        assert (
            pha_info["pha"].response_file.split("/")[-1]
            == "glg_cspec_n3_bn080916009_v07.rsp"
        )
        assert pha_info["pha"].scale_factor == 1.0

        assert pha_info["bak"].background_file is None

        # Test that we cannot get a bak file
        #
        # with pytest.raises(KeyError):
        #
        #     _ = pha_info['bak'].background_file
        #
        # Test that we cannot get a anc file
        # with pytest.raises(KeyError):
        #
        #     _ = pha_info['bak'].ancillary_file

        assert pha_info["bak"].response_file is None

        assert pha_info["bak"].ancillary_file is None

        # # Test that we cannot get a RSP file
        # with pytest.raises(AttributeError):
        #      _ = pha_info['bak'].response_file

        assert pha_info["bak"].instrument == "GBM_NAI_03"
        assert pha_info["bak"].mission == "GLAST"

        assert pha_info["bak"].is_poisson == False

        assert pha_info["bak"].n_channels == ogip.n_data_points
        assert pha_info["bak"].n_channels == len(pha_info["pha"].rates)

        assert len(pha_info["bak"].rate_errors) == pha_info["bak"].n_channels

        assert (
            sum(pha_info["bak"].sys_errors == np.zeros_like(pha_info["pha"].rates))
            == pha_info["bak"].n_channels
        )

        assert pha_info["bak"].scale_factor == 1.0

        assert isinstance(pha_info["rsp"], OGIPResponse)


def test_ogip_energy_selection():
    with within_directory(__example_dir):
        ogip = OGIPLike("test_ogip", observation="test.pha{1}")

        assert sum(ogip._mask) == sum(ogip.quality.good)

        # Test that  selecting a subset reduces the number of data points
        ogip.set_active_measurements("10-30")

        assert sum(ogip._mask) == ogip.n_data_points
        assert sum(ogip._mask) < 128

        # Test selecting all channels
        ogip.set_active_measurements("all")

        assert sum(ogip._mask) == ogip.n_data_points
        assert sum(ogip._mask) == 128

        # Test channel setting
        ogip.set_active_measurements(exclude=["c0-c1"])

        assert sum(ogip._mask) == ogip.n_data_points
        assert sum(ogip._mask) == 126

        # Test mixed ene/chan setting
        ogip.set_active_measurements(exclude=["0-c1"], verbose=True)

        assert sum(ogip._mask) == ogip.n_data_points
        assert sum(ogip._mask) == 126

        # Test that energies cannot be input backwards
        with pytest.raises(AssertionError):
            ogip.set_active_measurements("50-30")

        with pytest.raises(AssertionError):
            ogip.set_active_measurements("c20-c10")

        with pytest.raises(AssertionError):
            ogip.set_active_measurements("c100-0")

        with pytest.raises(AssertionError):
            ogip.set_active_measurements("c1-c200")

        with pytest.raises(AssertionError):
            ogip.set_active_measurements("10-c200")

        ogip.set_active_measurements("reset")

        assert sum(ogip._mask) == sum(ogip.quality.good)


def test_ogip_rebinner():
    with within_directory(__example_dir):
        ogip = OGIPLike("test_ogip", observation="test.pha{1}")

        n_data_points = 128
        ogip.set_active_measurements("all")

        assert ogip.n_data_points == n_data_points

        ogip.rebin_on_background(min_number_of_counts=100)

        assert ogip.n_data_points < 128

        with pytest.raises(AssertionError):
            ogip.set_active_measurements("all")

        ogip.remove_rebinning()

        assert ogip._rebinner is None

        assert ogip.n_data_points == n_data_points

        ogip.view_count_spectrum()


def test_various_effective_area():
    with within_directory(__example_dir):
        ogip = OGIPLike("test_ogip", observation="test.pha{1}")

        ogip.use_effective_area_correction()

        ogip.fix_effective_area_correction()


def test_simulating_data_sets():
    with within_directory(__example_dir):

        ogip = OGIPLike("test_ogip", observation="test.pha{1}")

        with pytest.raises(AssertionError):
            _ = ogip.simulated_parameters

        n_data_points = 128
        ogip.set_active_measurements("all")

        assert ogip._n_synthetic_datasets == 0

        ab = AnalysisBuilder(ogip)
        _ = ab.get_jl("normal")

        new_ogip = ogip.get_simulated_dataset("sim")

        assert new_ogip.name == "sim"
        assert ogip._n_synthetic_datasets == 1
        assert new_ogip.n_data_points == n_data_points

        assert new_ogip.n_data_points == sum(new_ogip._mask)
        assert sum(new_ogip._mask) == new_ogip.n_data_points
        assert new_ogip.tstart == 0.0

        assert "cons_sim" in new_ogip.nuisance_parameters
        assert new_ogip.nuisance_parameters["cons_sim"].fix == True
        assert new_ogip.nuisance_parameters["cons_sim"].free == False

        pha_info = new_ogip.get_pha_files()

        assert "pha" in pha_info
        assert "bak" in pha_info
        assert "rsp" in pha_info

        del ogip
        del new_ogip

        ogip = OGIPLike("test_ogip", observation="test.pha{1}")

        ab = AnalysisBuilder(ogip)
        _ = ab.get_jl("normal")

        # Now check that generationing a lot of data sets works

        sim_data_sets = [ogip.get_simulated_dataset("sim%d" % i) for i in range(100)]

        assert len(sim_data_sets) == ogip._n_synthetic_datasets

        for i, ds in enumerate(sim_data_sets):

            assert ds.name == "sim%d" % i
            assert sum(ds._mask) == sum(ogip._mask)
            assert ds._rebinner is None


def test_likelihood_ratio_test():
    with within_directory(__example_dir):
        ogip = OGIPLike("test_ogip", observation="test.pha{1}")

        ogip.set_active_measurements("all")

        ab = AnalysisBuilder(ogip)

        jl1 = ab.get_jl("normal")

        res1, _ = jl1.fit(compute_covariance=True)

        jl2 = ab.get_jl("cpl")
        res2, _ = jl2.fit(compute_covariance=True)

    lrt = LikelihoodRatioTest(jl1, jl2)

    null_hyp_prob, TS, data_frame, like_data_frame = lrt.by_mc(
        n_iterations=50, continue_on_failure=True
    )


def test_xrt():
    with within_directory(__example_dir):
        trigger = "GRB110731A"
        dec = -28.546
        ra = 280.52
        xrt_dir = "xrt"
        xrt = SwiftXRTLike(
            "XRT",
            observation=os.path.join(xrt_dir, "xrt_src.pha"),
            background=os.path.join(xrt_dir, "xrt_bkg.pha"),
            response=os.path.join(xrt_dir, "xrt.rmf"),
            arf_file=os.path.join(xrt_dir, "xrt.arf"),
        )

        spectral_model = Powerlaw()

        ptsrc = PointSource(trigger, ra, dec, spectral_shape=spectral_model)
        model = Model(ptsrc)

        data = DataList(xrt)

        jl = JointLikelihood(model, data, verbose=False)


def test_swift_gbm():
    with within_directory(__example_dir):
        gbm_dir = "gbm"
        bat_dir = "bat"

        bat = OGIPLike(
            "BAT",
            observation=os.path.join(bat_dir, "gbm_bat_joint_BAT.pha"),
            response=os.path.join(bat_dir, "gbm_bat_joint_BAT.rsp"),
        )

        bat.set_active_measurements("15-150")
        bat.view_count_spectrum()

        nai6 = OGIPLike(
            "n6",
            os.path.join(gbm_dir, "gbm_bat_joint_NAI_06.pha"),
            os.path.join(gbm_dir, "gbm_bat_joint_NAI_06.bak"),
            os.path.join(gbm_dir, "gbm_bat_joint_NAI_06.rsp"),
            spectrum_number=1,
        )

        nai6.set_active_measurements("8-900")
        nai6.view_count_spectrum()

        bgo0 = OGIPLike(
            "b0",
            os.path.join(gbm_dir, "gbm_bat_joint_BGO_00.pha"),
            os.path.join(gbm_dir, "gbm_bat_joint_BGO_00.bak"),
            os.path.join(gbm_dir, "gbm_bat_joint_BGO_00.rsp"),
            spectrum_number=1,
        )

        bgo0.set_active_measurements("250-10000")
        bgo0.view_count_spectrum()

        bat.use_effective_area_correction(0.2, 1.5)
        bat.fix_effective_area_correction(0.6)
        bat.use_effective_area_correction(0.2, 1.5)

        band = Band()
        model = Model(PointSource("joint_fit", 0, 0, spectral_shape=band))

        band.K = 0.04
        band.xp = 300.0

        data_list = DataList(bat, nai6, bgo0)

        jl = JointLikelihood(model, data_list)

        _ = jl.fit()

        _ = display_spectrum_model_counts(jl, step=False)


def test_pha_write():
    with within_directory(__example_dir):

        ogip = OGIPLike("test_ogip", observation="test.pha{1}")

        ogip.write_pha("test_write", overwrite=True)

        written_ogip = OGIPLike("write_ogip", observation="test_write.pha{1}")

        pha_info = written_ogip.get_pha_files()

        for key in ["pha", "bak"]:

            assert isinstance(pha_info[key], PHASpectrum)

        assert pha_info["pha"].background_file == "test_bak.pha{1}"
        assert pha_info["pha"].ancillary_file is None
        assert pha_info["pha"].instrument == "GBM_NAI_03"
        assert pha_info["pha"].mission == "GLAST"
        assert pha_info["pha"].is_poisson == True
        assert pha_info["pha"].n_channels == len(pha_info["pha"].rates)


def test_pha_write_no_bkg():
    with within_directory(__example_dir):

        # custom remove background
        f = fits.open("test.pha")
        f["SPECTRUM"].data["BACKFILE"] = "NONE"
        f.writeto("test_pha_nobkg.pha", overwrite=True)

        ogip = OGIPLike("test_ogip", observation="test_pha_nobkg.pha{1}")

        ogip.write_pha("test_write_nobkg", overwrite=True)

        written_ogip = OGIPLike("write_ogip", observation="test_write_nobkg.pha{1}")

        pha_info = written_ogip.get_pha_files()

        for key in ["pha"]:
            assert isinstance(pha_info[key], PHASpectrum)

        f = fits.open("test_write_nobkg.pha")
        assert f["SPECTRUM"].data["BACKFILE"][0] == "NONE"

        assert pha_info["pha"].background_file is None
        assert pha_info["pha"].ancillary_file is None
        assert pha_info["pha"].instrument == "GBM_NAI_03"
        assert pha_info["pha"].mission == "GLAST"
        assert pha_info["pha"].is_poisson == True
        assert pha_info["pha"].n_channels == len(pha_info["pha"].rates)


def test_likelihood_functions():
    obs_cnts = np.array([10])
    obs_bkg = np.array([5])
    bkg_err = np.array([1])
    exp_cnts = np.array([5])
    exp_bkg = np.array([5])
    ratio = 1

    ll, b = poisson_log_likelihood_ideal_bkg(
        observed_counts=obs_cnts,
        expected_bkg_counts=exp_bkg,
        expected_model_counts=exp_bkg,
    )

    test = (ll[0], b[0])

    npt.assert_almost_equal(test, (-2.0785616431350551, 5), decimal=4)

    ll, b = poisson_observed_poisson_background(
        observed_counts=obs_cnts,
        background_counts=obs_bkg,
        exposure_ratio=ratio,
        expected_model_counts=exp_cnts,
    )

    test = (ll[0], b[0])

    npt.assert_almost_equal(test, (-3.8188638237465984, 5.0), decimal=4)

    test = poisson_observed_poisson_background_xs(
        observed_counts=obs_cnts,
        background_counts=obs_bkg,
        exposure_ratio=ratio,
        expected_model_counts=exp_cnts,
    )

    assert test == -0.0

    ll, b = poisson_observed_gaussian_background(
        observed_counts=obs_cnts,
        background_counts=obs_bkg,
        background_error=bkg_err,
        expected_model_counts=exp_cnts,
    )
    test = (ll[0], b[0])

    npt.assert_almost_equal(test, (-2.99750018, 5.0), decimal=4)

    # assert test == (-2.99750018, 5.0)
