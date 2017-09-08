import pytest

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
skip_if_hawc_is_not_available = pytest.mark.skipif((os.environ.get('HAWC_3ML_TEST_DATA_DIR') is None) or (not has_HAWC),
                                                   reason="HAWC test dataset or HAWC environment is not available")

def is_within_tolerance(truth, value, relative_tolerance=0.01):

    assert truth != 0

    if abs((truth-value) / truth) <= relative_tolerance:

        return True

    else:

        return False


_maptree_name = "maptree_256.root"
_response_name = "detector_response.root"


#@skip_if_hawc_is_not_available
def test_hawc_point_source_fit():

    # Ensure test environment is valid

    assert is_plugin_available("HAWCLike"), "HAWCLike is not available!"

    data_path = sanitize_filename(os.environ.get('HAWC_3ML_TEST_DATA_DIR'), abspath=True)

    maptree = os.path.join(data_path, _maptree_name)
    response = os.path.join(data_path, _response_name)

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
    assert is_within_tolerance(3.07920784548e-20, parameter_frame['value']['TestSource.spectrum.main.Cutoff_powerlaw.K'])
    assert is_within_tolerance(-2.33736923856, parameter_frame['value']['TestSource.spectrum.main.Cutoff_powerlaw.index'])
    assert is_within_tolerance(41889862104.0, parameter_frame['value']['TestSource.spectrum.main.Cutoff_powerlaw.xc'])

    assert is_within_tolerance(55979.423676, like['-log(likelihood)']['HAWC'])

    # Print up the TS, significance, and fit parameters, and then plot stuff
    print("\nTest statistic:")
    TS = llh.calc_TS()
    sigma = np.sqrt(TS)

    print("Test statistic: %g" % TS)
    print("Significance:   %g\n" % sigma)

    assert is_within_tolerance(14366.4, TS)
    assert is_within_tolerance(119.86, sigma)

    # Get the differential flux at 1 TeV
    diff_flux = spectrum(1 * u.TeV)
    # Convert it to 1 / (TeV cm2 s)
    diff_flux_TeV = diff_flux.to(1 / (u.TeV * u.cm ** 2 * u.s))

    print("Norm @ 1 TeV:  %s \n" % diff_flux_TeV)

    assert is_within_tolerance(3.00657105936e-11, diff_flux_TeV.value)

    spectrum.display()

#@skip_if_hawc_is_not_available
def test_hawc_extended_source_fit():

    # Ensure test environment is valid

    assert is_plugin_available("HAWCLike"), "HAWCLike is not available!"

    data_path = sanitize_filename(os.environ.get('HAWC_3ML_TEST_DATA_DIR'), abspath=True)

    maptree = os.path.join(data_path, _maptree_name)
    response = os.path.join(data_path, _response_name)

    assert os.path.exists(maptree) and os.path.exists(response), "Data files do not exist at %s" % data_path

    # The simulated source has this spectrum (credits for simulation: Colas Riviere):
    # CutOffPowerLaw,1.32e-07,2.37,42.3
    # at this position:
    # 100,22
    # with a disk shape with an extension of 1.5 deg

    # Define the spectral and spatial models for the source
    spectrum = Cutoff_powerlaw()

    shape = Disk_on_sphere()

    source = ExtendedSource("ExtSource",
                            spatial_shape=shape,
                            spectral_shape=spectrum)

    shape.lon0 = 100.0
    shape.lon0.fix = True

    shape.lat0 = 22.0
    shape.lat0.fix = True

    shape.radius = 1.5 * u.degree
    shape.radius.bounds = (0.5 * u.degree, 1.55 * u.degree)
    #shape.radius.fix = True

    spectrum.K = 4.39964273e-20
    spectrum.K.bounds = (1e-24, 1e-17)

    spectrum.piv = 1 * u.TeV
    #spectrum.piv.fix = True

    spectrum.index = -2.37
    spectrum.index.bounds = (-4, -1)
    #spectrum.index.fix = True

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
    assert is_within_tolerance(4.64056469931e-20, parameter_frame['value']['ExtSource.spectrum.main.Cutoff_powerlaw.K'])
    assert is_within_tolerance(-2.44931279819, parameter_frame['value']['ExtSource.spectrum.main.Cutoff_powerlaw.index'])
    assert is_within_tolerance(1.45222982526, parameter_frame['value']['ExtSource.Disk_on_sphere.radius'])

    assert is_within_tolerance(186389.106099, like['-log(likelihood)']['HAWC'])

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
    diff_flux_TeV = diff_flux.to(1 / (u.TeV * u.cm ** 2 * u.s))

    print("Norm @ 1 TeV:  %s \n" % diff_flux_TeV)

    assert is_within_tolerance(4.53214528088e-11, diff_flux_TeV.value)

    spectrum.display()
    shape.display()
    
    
#@skip_if_hawc_is_not_available
def test_hawc_display_residuals():

    # Ensure test environment is valid

    assert is_plugin_available("HAWCLike"), "HAWCLike is not available!"

    data_path = sanitize_filename(os.environ.get('HAWC_3ML_TEST_DATA_DIR'), abspath=True)

    maptree = os.path.join(data_path, _maptree_name)
    response = os.path.join(data_path, _response_name)

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

    #Check the 'display' functions (plot model&data/residuals vs analysis bins)
    llh.display(radius=0.5)
    llh.display_residuals_at_position( source.position.ra.value, source.position.dec.value, radius = 0.5 )
    
@skip_if_hawc_is_not_available
def test_radial_profile():
    # Ensure test environment is valid

    assert is_plugin_available("HAWCLike"), "HAWCLike is not available!"

    data_path = sanitize_filename(os.environ.get('HAWC_3ML_TEST_DATA_DIR'), abspath=True)

    maptree = os.path.join(data_path, _maptree_name)
    response = os.path.join(data_path, _response_name)

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


    correct_radii = [ 0.1,  0.3,  0.5,  0.7,  0.9,  1.1,  1.3,  1.5,  1.7,  1.9]
    correct_model = [  1.00635816e+07,   3.77671396e+06,   6.52140500e+05,   1.39108253e+05,   4.95474715e+04,   2.14023029e+04,   1.06772849e+04,   5.25318866e+03,   2.33563298e+03,   9.49504012e+02]
    correct_data = [  9.85388626e+06,   3.86417068e+06,  6.72710742e+05,   1.39626894e+05,   4.97488619e+04,   3.89451456e+04,  -5.81909772e+04,   1.26997077e+04,   4.58937031e+03,  -1.88383873e+04]
    correct_error = [ 236050.10257665,   81412.05436244,   40652.3404534,    30430.46607895,   25915.97033859,   24532.79588011,   23840.33882012,   21953.01507234,   19910.52807621,   17519.43963043]
    correct_bins = ['4', '5', '6', '7', '8', '9']

    subtracted_data = [ -2.09695382e+05,   8.74567227e+04,   2.05702422e+04,   5.18641216e+02,   2.01390472e+02,   1.75428427e+04,  -6.88682621e+04,   7.44651905e+03,   2.25373733e+03,  -1.97878914e+04]
    
    max_radius = 2.0
    n_bins = 10
    bins_to_use =  ['4', '5', '6', '7', '8', '9']
    
    radii, excess_model, excess_data, excess_error, list_of_bin_names = llh.get_radial_profile( source.position.ra.value, source.position.dec.value,bins_to_use, max_radius, n_bins)
   
    assert len(radii) == n_bins
    assert len(excess_model) == n_bins
    assert len(excess_data) == n_bins
    assert len(excess_error) == n_bins
   
    assert list_of_bin_names == correct_bins
   
    for i in range(0, n_bins):
      assert is_within_tolerance( radii[i], correct_radii[i])
      assert is_within_tolerance( excess_model[i], correct_model[i])
      assert is_within_tolerance( excess_data[i], correct_data[i])
      assert is_within_tolerance( excess_error[i], correct_error[i])
      
    
    radii, excess_model, excess_data, excess_error, list_of_bin_names = llh.get_radial_profile( source.position.ra.value, source.position.dec.value,bins_to_use, max_radius, n_bins, lm)

    assert len(radii) == n_bins
    assert len(excess_model) == n_bins
    assert len(excess_data) == n_bins
    assert len(excess_error) == n_bins
   
    assert list_of_bin_names == correct_bins
   
    for i in range(0, n_bins):
      assert is_within_tolerance( radii[i], correct_radii[i])
      assert is_within_tolerance( excess_model[i], correct_model[i])
      assert is_within_tolerance( excess_data[i], subtracted_data[i])
      assert is_within_tolerance( excess_error[i], correct_error[i])
     

def test_CommonNorm_fit():

    assert is_plugin_available("HAWCLike"), "HAWCLike is not available!"

    data_path = sanitize_filename(os.environ.get('HAWC_3ML_TEST_DATA_DIR'), abspath=True)

    maptree = os.path.join(data_path, _maptree_name)
    response = os.path.join(data_path, _response_name)

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

    assert np.isclose(lm.HAWC_ComNorm.value, 1.02567968495, rtol=1e-2)