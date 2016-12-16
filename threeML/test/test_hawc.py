import pytest

from threeML import *

from threeML.plugins.HAWCLike import HAWCLike
from threeML.io.file_utils import sanitize_filename

# This defines a decorator which can be applied to single tests to
# skip them if the condition is not met
skip_if_hawc_is_not_available = pytest.mark.skipif(os.environ.get('HAWC_3ML_TEST_DATA_DIR') is None,
                                                   reason="HAWC test data are not available")

def is_within_tolerance(truth, value, relative_tolerance=0.01):

    assert truth != 0

    if abs((truth-value) / truth) <= relative_tolerance:

        return True

    else:

        return False


@skip_if_hawc_is_not_available
def test_hawc_point_source_fit():

    # Ensure test environment is valid

    assert is_plugin_available("HAWCLike"), "HAWCLike is not available!"

    data_path = sanitize_filename(os.environ.get('HAWC_3ML_TEST_DATA_DIR'), abspath=True)

    maptree = os.path.join(data_path, 'maptree_1024.root')
    response = os.path.join(data_path, 'detector_response.root')

    assert os.path.exists(maptree) and os.path.exists(response), "Data files do not exist at %s" % data_path

    # The simulated source has this spectrum (credits for simulation: Colas Riviere):
    # CutOffPowerLaw,3.15e-11,2.37,42.3
    # at this position:
    # 100,22

    # Define the spectral and spatial models for the source
    spectrum = Cutoff_powerlaw()
    source = PointSource("TestSource", ra=100.0, dec=22.0, spectral_shape=spectrum)

    spectrum.K = 3.15e-11 / (u.TeV * u.cm ** 2 * u.s)
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

    # Check that we have converged to the right solution
    # (the true value of course are not exactly the value simulated,
    # they are just the point where the fit should converge)
    assert is_within_tolerance(3.08428116752e-20, parameter_frame['value']['TestSource.spectrum.main.Cutoff_powerlaw.K'])
    assert is_within_tolerance(-2.35433951408, parameter_frame['value']['TestSource.spectrum.main.Cutoff_powerlaw.index'])
    assert is_within_tolerance(43592372025.7, parameter_frame['value']['TestSource.spectrum.main.Cutoff_powerlaw.xc'])

    assert is_within_tolerance(590025.107095, like['-log(likelihood)']['HAWC'])

    # Print up the TS, significance, and fit parameters, and then plot stuff
    print("\nTest statistic:")
    TS = llh.calc_TS()
    sigma = np.sqrt(TS)

    assert is_within_tolerance(15370.8, TS)
    assert is_within_tolerance(123.979, sigma)

    print("Test statistic: %g" % TS)
    print("Significance:   %g\n" % sigma)

    # Get the differential flux at 1 TeV
    diff_flux = spectrum(1 * u.TeV)
    # Convert it to 1 / (TeV cm2 s)
    diff_flux_TeV = diff_flux.to(1 / (u.TeV * u.cm ** 2 * u.s))

    print("Norm @ 1 TeV:  %s \n" % diff_flux_TeV)

    assert is_within_tolerance(3.01433375231e-11, diff_flux_TeV.value)

    spectrum.display()