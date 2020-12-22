from __future__ import division
from __future__ import print_function
from builtins import zip
from builtins import range
from past.utils import old_div
import pytest
import os

from threeML import *


try:

    from threeML.plugins.HAWCLike import HAWCLike

except ImportError:

    has_HAWC = False

else:

    has_HAWC = True

from threeML.io.file_utils import sanitize_filename

# This defines a decorator which can be applied to single tests to
# skip them if the condition is not met
skip_if_hawc_is_not_available = pytest.mark.skipif(
    (os.environ.get("HAWC_3ML_TEST_DATA_DIR") is None) or (not has_HAWC),
    reason="HAWC test dataset or HAWC environment is not available",
)


def is_within_tolerance(truth, value, relative_tolerance=0.01):
    assert truth != 0

    if abs(old_div((truth - value), truth)) <= relative_tolerance:

        return True

    else:

        return False


def is_null_within_tolerance(value, absolute_tolerance):
    if abs(value) <= absolute_tolerance:
        return True
    else:
        return False


_maptree_name = "maptree_256.root"
_response_name = "detector_response.root"


@pytest.fixture(scope="session")
def hawc_point_source_fitted_joint_like():

    data_path = sanitize_filename(
        os.environ.get("HAWC_3ML_TEST_DATA_DIR"), abspath=True
    )

    maptree = os.path.join(data_path, _maptree_name)
    response = os.path.join(data_path, _response_name)

    assert os.path.exists(maptree) and os.path.exists(response), (
        "Data files do not exist at %s" % data_path
    )

    # The simulated source has this spectrum (credits for simulation: Colas Riviere):
    # CutOffPowerLaw,3.15e-11,2.37,42.3
    # at this position:
    # 100,22

    # Define the spectral and spatial models for the source
    spectrum = Cutoff_powerlaw()
    source = PointSource("TestSource", ra=100.0, dec=22.0, spectral_shape=spectrum)

    spectrum.K = old_div(3.15e-11, (u.TeV * u.cm ** 2 * u.s))
    spectrum.K.bounds = (1e-22, 1e-18)  # without units energies are in keV

    spectrum.piv = 1 * u.TeV
    spectrum.piv.fix = True

    spectrum.index = -2.37
    spectrum.index.bounds = (-4, -1)

    spectrum.xc = 42.3 * u.TeV
    spectrum.xc.bounds = (1 * u.TeV, 100 * u.TeV)

    q = source(1 * u.keV)

    assert np.isclose(q.value, 67.3458058177)

    # Set up a likelihood model using the source.
    # Then create a HAWCLike object using the model, the maptree, and detector
    # response.
    lm = Model(source)
    llh = HAWCLike("HAWC", maptree, response)
    llh.set_active_measurements(1, 9)

    # Double check the free parameters
    print("Likelihood model:\n")
    print(lm)

    # Set up the likelihood and run the fit
    print("Performing likelihood fit...\n")
    datalist = DataList(llh)
    jl = JointLikelihood(lm, datalist, verbose=True)

    jl.set_minimizer("ROOT")

    parameter_frame, like = jl.fit(compute_covariance=False)

    return jl, parameter_frame, like


@skip_if_hawc_is_not_available
def test_set_active_measurements():

    data_path = sanitize_filename(
        os.environ.get("HAWC_3ML_TEST_DATA_DIR"), abspath=True
    )

    maptree = os.path.join(data_path, _maptree_name)
    response = os.path.join(data_path, _response_name)

    assert os.path.exists(maptree) and os.path.exists(response), (
        "Data files do not exist at %s" % data_path
    )

    llh = HAWCLike("HAWC", maptree, response)
    # Test one way
    llh.set_active_measurements(1, 9)
    # Test the other way
    llh.set_active_measurements(bin_list=["4", "5", "6", "7", "8", "9"])


@skip_if_hawc_is_not_available
def test_hawc_fullsky_options():

    assert is_plugin_available("HAWCLike"), "HAWCLike is not available!"

    data_path = sanitize_filename(
        os.environ.get("HAWC_3ML_TEST_DATA_DIR"), abspath=True
    )

    maptree = os.path.join(data_path, _maptree_name)
    response = os.path.join(data_path, _response_name)

    assert os.path.exists(maptree) and os.path.exists(response), (
        "Data files do not exist at %s" % data_path
    )

    # The simulated source has this spectrum (credits for simulation: Colas Riviere):
    # CutOffPowerLaw,3.15e-11,2.37,42.3
    # at this position:
    # 100,22

    # Define the spectral and spatial models for the source
    spectrum = Cutoff_powerlaw()
    source = PointSource("TestSource", ra=100.0, dec=22.0, spectral_shape=spectrum)

    spectrum.K = old_div(3.15e-11, (u.TeV * u.cm ** 2 * u.s))
    spectrum.K.bounds = (1e-22, 1e-18)  # without units energies are in keV

    spectrum.piv = 1 * u.TeV
    spectrum.piv.fix = True

    spectrum.index = -2.37
    spectrum.index.bounds = (-4, -1)

    spectrum.xc = 42.3 * u.TeV
    spectrum.xc.bounds = (1 * u.TeV, 100 * u.TeV)

    q = source(1 * u.keV)

    assert np.isclose(q.value, 67.3458058177)

    # Set up a likelihood model using the source.
    # Then create a HAWCLike object using the model, the maptree, and detector
    # response.
    lm = Model(source)

    # Test with fullsky=True, and try to perform a fit to verify that we throw an exception

    llh = HAWCLike("HAWC", maptree, response, fullsky=True)
    llh.set_active_measurements(1, 9)

    # Double check the free parameters
    print("Likelihood model:\n")
    print(lm)

    # Set up the likelihood and run the fit
    print("Performing likelihood fit...\n")
    datalist = DataList(llh)

    with pytest.raises(RuntimeError):

        jl = JointLikelihood(lm, datalist, verbose=False)

    # Now we use set_ROI and this should work
    llh.set_ROI(100.0, 22.0, 2.0)

    jl = JointLikelihood(lm, datalist, verbose=False)

    # Now test that we can use set_ROI even though fullsky=False
    llh = HAWCLike("HAWC", maptree, response, fullsky=False)
    llh.set_active_measurements(1, 9)
    llh.set_ROI(100.0, 22.0, 1.0)

    # Double check the free parameters
    print("Likelihood model:\n")
    print(lm)

    # Set up the likelihood
    print("Performing likelihood fit...\n")
    datalist = DataList(llh)

    jl = JointLikelihood(lm, datalist, verbose=False)


@skip_if_hawc_is_not_available
def test_hawc_point_source_fit(hawc_point_source_fitted_joint_like):
    # Ensure test environment is valid

    assert is_plugin_available("HAWCLike"), "HAWCLike is not available!"

    jl, parameter_frame, like = hawc_point_source_fitted_joint_like
    spectrum = jl.likelihood_model.TestSource.spectrum.main.shape

    # Check that we have converged to the right solution
    # (the true value of course are not exactly the value simulated,
    # they are just the point where the fit should converge)
    assert is_within_tolerance(
        3.3246428894535895e-20,
        parameter_frame["value"]["TestSource.spectrum.main.Cutoff_powerlaw.K"],
    )
    assert is_within_tolerance(
        -2.33736923856,
        parameter_frame["value"]["TestSource.spectrum.main.Cutoff_powerlaw.index"],
    )
    assert is_within_tolerance(
        37478522636.504425,
        parameter_frame["value"]["TestSource.spectrum.main.Cutoff_powerlaw.xc"],
    )

    assert is_within_tolerance(55979.424031, like["-log(likelihood)"]["HAWC"])

    # Print up the TS, significance, and fit parameters, and then plot stuff
    print("\nTest statistic:")
    TS = jl.data_list["HAWC"].calc_TS()
    sigma = np.sqrt(TS)

    print("Test statistic: %g" % TS)
    print("Significance:   %g\n" % sigma)

    assert is_within_tolerance(14366.4, TS)
    assert is_within_tolerance(119.86, sigma)

    # Get the differential flux at 1 TeV
    diff_flux = spectrum(1 * u.TeV)
    # Convert it to 1 / (TeV cm2 s)
    diff_flux_TeV = diff_flux.to(old_div(1, (u.TeV * u.cm ** 2 * u.s)))

    print("Norm @ 1 TeV:  %s \n" % diff_flux_TeV)

    assert is_within_tolerance(3.2371079347638675e-11, diff_flux_TeV.value)

    spectrum.display()


@skip_if_hawc_is_not_available
def test_hawc_extended_source_fit():
    # Ensure test environment is valid

    assert is_plugin_available("HAWCLike"), "HAWCLike is not available!"

    data_path = sanitize_filename(
        os.environ.get("HAWC_3ML_TEST_DATA_DIR"), abspath=True
    )

    maptree = os.path.join(data_path, _maptree_name)
    response = os.path.join(data_path, _response_name)

    assert os.path.exists(maptree) and os.path.exists(response), (
        "Data files do not exist at %s" % data_path
    )

    # The simulated source has this spectrum (credits for simulation: Colas Riviere):
    # CutOffPowerLaw,1.32e-07,2.37,42.3
    # at this position:
    # 100,22
    # with a disk shape with an extension of 1.5 deg

    # Define the spectral and spatial models for the source
    spectrum = Cutoff_powerlaw()

    shape = Disk_on_sphere()

    source = ExtendedSource("ExtSource", spatial_shape=shape, spectral_shape=spectrum)

    shape.lon0 = 100.0
    shape.lon0.fix = True

    shape.lat0 = 22.0
    shape.lat0.fix = True

    shape.radius = 1.5 * u.degree
    shape.radius.bounds = (0.5 * u.degree, 1.55 * u.degree)
    # shape.radius.fix = True

    spectrum.K = 4.39964273e-20
    spectrum.K.bounds = (1e-24, 1e-17)

    spectrum.piv = 1 * u.TeV
    # spectrum.piv.fix = True

    spectrum.index = -2.37
    spectrum.index.bounds = (-4, -1)
    # spectrum.index.fix = True

    spectrum.xc = 42.3 * u.TeV
    spectrum.xc.bounds = (1 * u.TeV, 100 * u.TeV)
    spectrum.xc.fix = True

    # Set up a likelihood model using the source.
    # Then create a HAWCLike object using the model, the maptree, and detector
    # response.
    lm = Model(source)
    llh = HAWCLike("HAWC", maptree, response)
    llh.set_active_measurements(1, 9)

    # Double check the free parameters
    print("Likelihood model:\n")
    print(lm)

    # Set up the likelihood and run the fit
    print("Performing likelihood fit...\n")
    datalist = DataList(llh)
    jl = JointLikelihood(lm, datalist, verbose=True)

    jl.set_minimizer("ROOT")

    parameter_frame, like = jl.fit(compute_covariance=False)

    # Check that we have converged to the right solution
    # (the true value of course are not exactly the value simulated,
    # they are just the point where the fit should converge)
    assert is_within_tolerance(
        4.7805737823025172e-20,
        parameter_frame["value"]["ExtSource.spectrum.main.Cutoff_powerlaw.K"],
    )
    assert is_within_tolerance(
        -2.44931279819,
        parameter_frame["value"]["ExtSource.spectrum.main.Cutoff_powerlaw.index"],
    )
    assert is_within_tolerance(
        1.4273457159139373, parameter_frame["value"]["ExtSource.Disk_on_sphere.radius"]
    )

    assert is_within_tolerance(186389.581117, like["-log(likelihood)"]["HAWC"])

    # Print up the TS, significance, and fit parameters, and then plot stuff
    print("\nTest statistic:")
    TS = llh.calc_TS()
    sigma = np.sqrt(TS)

    assert is_within_tolerance(3510.26, TS)
    assert is_within_tolerance(59.2475, sigma)

    print("Test statistic: %g" % TS)
    print("Significance:   %g\n" % sigma)

    # Get the differential flux at 1 TeV
    diff_flux = spectrum(1 * u.TeV)
    # Convert it to 1 / (TeV cm2 s)
    diff_flux_TeV = diff_flux.to(old_div(1, (u.TeV * u.cm ** 2 * u.s)))

    print("Norm @ 1 TeV:  %s \n" % diff_flux_TeV)

    assert is_within_tolerance(4.66888328668e-11, diff_flux_TeV.value)

    spectrum.display()
    shape.display()


@skip_if_hawc_is_not_available
def test_hawc_display_residuals(hawc_point_source_fitted_joint_like):
    # Ensure test environment is valid

    assert is_plugin_available("HAWCLike"), "HAWCLike is not available!"

    jl, parameter_frame, like = hawc_point_source_fitted_joint_like
    source = jl.likelihood_model.TestSource

    # Check the 'display' functions (plot model&data/residuals vs analysis bins)
    llh = jl.data_list["HAWC"]
    llh.display(radius=0.5)
    llh.display_residuals_at_position(
        source.position.ra.value, source.position.dec.value, radius=0.5
    )

    # Now check the bin-dependent radius
    llh.display_residuals_at_position(
        source.position.ra.value, source.position.dec.value, radius=[0.5] * 9
    )


@skip_if_hawc_is_not_available
def test_null_hyp_prob(hawc_point_source_fitted_joint_like):

    # Ensure test environment is valid

    assert is_plugin_available("HAWCLike"), "HAWCLike is not available!"

    jl, parameter_frame, like = hawc_point_source_fitted_joint_like
    source = jl.likelihood_model.TestSource
    llh = jl.data_list["HAWC"]
    p_value = llh.calc_p_value(
        source.position.ra.value, source.position.dec.value, radius=[0.5] * 9
    )

    assert np.isclose(p_value, 0.88173524636, rtol=0.1)


@skip_if_hawc_is_not_available
def test_radial_profile(hawc_point_source_fitted_joint_like):
    # Ensure test environment is valid

    assert is_plugin_available("HAWCLike"), "HAWCLike is not available!"

    jl, parameter_frame, like = hawc_point_source_fitted_joint_like
    source = jl.likelihood_model.TestSource
    llh = jl.data_list["HAWC"]
    lm = jl.likelihood_model

    correct_radii = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9]
    correct_model = [
        1.006176e07,
        3.775266e06,
        6.518357e05,
        1.390542e05,
        4.952657e04,
        2.139160e04,
        1.067152e04,
        5.250257e03,
        2.334314e03,
        9.489685e02,
    ]
    correct_data = [
        9.851449e06,
        3.862865e06,
        6.724352e05,
        1.395860e05,
        4.972864e04,
        3.894414e04,
        -5.817591e04,
        1.270122e04,
        4.575470e03,
        -1.882920e04,
    ]
    correct_error = [
        2.360076e05,
        8.138953e04,
        4.063915e04,
        3.042109e04,
        2.590773e04,
        2.452525e04,
        2.383261e04,
        2.194618e04,
        1.990404e04,
        1.751396e04,
    ]

    correct_bins = ["4", "5", "6", "7", "8", "9"]

    subtracted_data = [d - m for m, d in zip(correct_model, correct_data)]

    max_radius = 2.0
    n_bins = 10
    bins_to_use = ["4", "5", "6", "7", "8", "9"]

    (
        radii,
        excess_model,
        excess_data,
        excess_error,
        list_of_bin_names,
    ) = llh.get_radial_profile(
        source.position.ra.value,
        source.position.dec.value,
        bins_to_use,
        max_radius,
        n_bins,
    )

    # Un-comment the next lines to re-generate the "correct_" values if needed
    # print 'model, data, error:'
    # for v in [excess_model, excess_data, excess_error]:
    #     print '[',
    #     for vv in v:
    #         print '%e,' %vv,
    #     print ']'

    assert len(radii) == n_bins
    assert len(excess_model) == n_bins
    assert len(excess_data) == n_bins
    assert len(excess_error) == n_bins

    assert list_of_bin_names == correct_bins

    for i in range(0, n_bins):
        assert is_within_tolerance(radii[i], correct_radii[i])
        assert is_within_tolerance(excess_model[i], correct_model[i])
        assert is_within_tolerance(excess_data[i], correct_data[i])
        assert is_within_tolerance(excess_error[i], correct_error[i])

    # Now again subtracting the model from data and model
    (
        radii,
        excess_model,
        excess_data,
        excess_error,
        list_of_bin_names,
    ) = llh.get_radial_profile(
        source.position.ra.value,
        source.position.dec.value,
        bins_to_use,
        max_radius,
        n_bins,
        model_to_subtract=lm,
        subtract_model_from_model=True,
    )

    assert len(radii) == n_bins
    assert len(excess_model) == n_bins
    assert len(excess_data) == n_bins
    assert len(excess_error) == n_bins

    assert list_of_bin_names == correct_bins

    for i in range(0, n_bins):
        assert is_within_tolerance(radii[i], correct_radii[i])
        assert is_null_within_tolerance(excess_model[i], 0.01 * correct_model[i])
        assert is_within_tolerance(excess_data[i], correct_data[i] - correct_model[i])
        assert is_within_tolerance(excess_error[i], correct_error[i])


@skip_if_hawc_is_not_available
def test_CommonNorm_fit():
    assert is_plugin_available("HAWCLike"), "HAWCLike is not available!"

    data_path = sanitize_filename(
        os.environ.get("HAWC_3ML_TEST_DATA_DIR"), abspath=True
    )

    maptree = os.path.join(data_path, _maptree_name)
    response = os.path.join(data_path, _response_name)

    assert os.path.exists(maptree) and os.path.exists(response), (
        "Data files do not exist at %s" % data_path
    )
    # The simulated source has this spectrum (credits for simulation: Colas Riviere):
    # CutOffPowerLaw,3.15e-11,2.37,42.3
    # at this position:
    # 100,22

    # Define the spectral and spatial models for the source
    spectrum = Cutoff_powerlaw()
    source = PointSource("TestSource", ra=100.0, dec=22.0, spectral_shape=spectrum)

    spectrum.K = old_div(3.15e-11, (u.TeV * u.cm ** 2 * u.s))
    spectrum.K.bounds = (1e-22, 1e-18)  # without units energies are in keV
    spectrum.K.fix = True

    spectrum.piv = 1 * u.TeV
    spectrum.piv.fix = True

    spectrum.index = -2.37
    spectrum.index.bounds = (-4, -1)
    spectrum.index.free = False

    spectrum.xc = 42.3 * u.TeV
    spectrum.xc.bounds = (1 * u.TeV, 100 * u.TeV)
    spectrum.xc.free = False

    q = source(1 * u.keV)

    assert np.isclose(q.value, 67.3458058177)

    # Set up a likelihood model using the source.
    # Then create a HAWCLike object using the model, the maptree, and detector
    # response.
    lm = Model(source)
    llh = HAWCLike("HAWC", maptree, response)
    llh.set_active_measurements(1, 9)

    llh.activate_CommonNorm()

    # Double check the free parameters
    print("Likelihood model:\n")
    print(lm)

    # Set up the likelihood and run the fit
    print("Performing likelihood fit...\n")
    datalist = DataList(llh)
    jl = JointLikelihood(lm, datalist, verbose=True)

    jl.set_minimizer("ROOT")

    parameter_frame, like = jl.fit(compute_covariance=False)

    assert np.isclose(lm.HAWC_ComNorm.value, 1.0756519971562115, rtol=1e-2)


@skip_if_hawc_is_not_available
def test_hawc_get_number_of_data_points(hawc_point_source_fitted_joint_like):

    # Ensure test environment is valid

    assert is_plugin_available("HAWCLike"), "HAWCLike is not available!"

    jl, parameter_frame, like = hawc_point_source_fitted_joint_like
    llh = jl.data_list["HAWC"]

    assert llh.get_number_of_data_points() == 13428


@skip_if_hawc_is_not_available
def test_hawc_write_map(hawc_point_source_fitted_joint_like):

    # Ensure test environment is valid

    assert is_plugin_available("HAWCLike"), "HAWCLike is not available!"

    jl, parameter_frame, like = hawc_point_source_fitted_joint_like
    llh = jl.data_list["HAWC"]

    file_name = "__hawc_map.root"

    llh.write_map(file_name)

    assert os.path.exists(file_name)

    os.remove(file_name)
