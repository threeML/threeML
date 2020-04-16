from __future__ import division
from __future__ import print_function

# NOTE: XSpec must be loaded before any other plugin/package from threeML because otherwise it could
# complain about conflicting CFITSIO libraries
from past.utils import old_div
import os

if os.environ.get("HEADAS") is not None:

    # Try to import xspec

    try:

        import xspec

    except ImportError:

        has_pyxspec = False

    else:

        has_pyxspec = True

else:

    has_pyxspec = False

import pytest
import numpy as np
import os

from astromodels import Powerlaw
from threeML.io.package_data import get_path_of_data_file
from threeML.utils.OGIP.response import InstrumentResponse, OGIPResponse


skip_if_pyxspec_is_not_available = pytest.mark.skipif(
    not has_pyxspec, reason="No pyXspec installed"
)


def get_matrix_elements():

    # In[5]: np.diagflat([1, 2, 3, 4])[:3, :]

    matrix = np.diagflat([1.0, 2.0, 3.0, 4.0])[:3, :]

    # Now matrix is:
    # array([[1, 0, 0, 0],
    #        [0, 2, 0, 0],
    #        [0, 0, 3, 0]])

    mc_energies = [1.0, 2.0, 3.0, 4.0, 5.0]

    ebounds = [1.0, 2.5, 4.5, 5.0]

    return matrix, mc_energies, ebounds


@skip_if_pyxspec_is_not_available
def test_OGIP_response_against_xspec():

    # Test for various photon indexes
    for index in [-0.5, 0.0, 0.5, 1.5, 2.0, 3.0, 4.0]:

        print("Processing index %s" % index)

        # First reset xspec
        xspec.AllData.clear()

        # Create a model in XSpec

        mo = xspec.Model("po")

        # Change the default value for the photon index
        # (remember that in XSpec the definition of the powerlaw is norm * E^(-PhoIndex),
        # so PhoIndex is positive normally. This is the opposite of astromodels.
        mo.powerlaw.PhoIndex = index
        mo.powerlaw.norm = 12.2

        # Now repeat the same in 3ML

        # Generate the astromodels function and set it to the same values as the XSpec power law
        # (the pivot in XSpec is set to 1). Remember also that the definition in xspec has the
        # sign of the photon index opposite
        powerlaw = Powerlaw()
        powerlaw.piv = 1.0
        powerlaw.index = -mo.powerlaw.PhoIndex.values[0]
        powerlaw.K = mo.powerlaw.norm.values[0]

        # Exploit the fact that the power law integral is analytic
        powerlaw_integral = Powerlaw()
        # Remove transformation
        powerlaw_integral.K._transformation = None
        powerlaw_integral.K.bounds = (None, None)
        powerlaw_integral.index = powerlaw.index.value + 1
        powerlaw_integral.K = old_div(powerlaw.K.value, (powerlaw.index.value + 1))

        powerlaw_integral.display()

        integral_function = lambda e1, e2: powerlaw_integral(e2) - powerlaw_integral(e1)

        # Now check that the two convoluted model give the same number of counts in each channel

        # Fake a spectrum so we can actually compute the convoluted model

        # Get path of response file
        rsp_file = get_path_of_data_file("ogip_test_gbm_n6.rsp")

        fs1 = xspec.FakeitSettings(
            rsp_file, exposure=1.0, fileName="_fake_spectrum.pha"
        )

        xspec.AllData.fakeit(noWrite=True, applyStats=False, settings=fs1)

        # Get the expected counts
        xspec_counts = mo.folded(1)

        # Now get the convolution from 3ML
        rsp = OGIPResponse(rsp_file)

        rsp.set_function(integral_function)

        threeML_counts = rsp.convolve()

        # Compare them
        assert np.allclose(xspec_counts, threeML_counts)

        # Now do the same with a matrix with a ARF

        # First reset xspec
        xspec.AllData.clear()

        # Then load rsp and arf in XSpec

        rsp_file = get_path_of_data_file("ogip_test_xmm_pn.rmf")

        arf_file = get_path_of_data_file("ogip_test_xmm_pn.arf")

        fs1 = xspec.FakeitSettings(
            rsp_file, arf_file, exposure=1.0, fileName="_fake_spectrum.pha"
        )

        xspec.AllData.fakeit(noWrite=True, applyStats=False, settings=fs1)

        # Get the expected counts
        xspec_counts = mo.folded(1)

        # Now get the convolution from 3ML
        rsp = OGIPResponse(rsp_file, arf_file=arf_file)

        rsp.set_function(integral_function)

        threeML_counts = rsp.convolve()

        # Compare them
        assert np.allclose(xspec_counts, threeML_counts)


@skip_if_pyxspec_is_not_available
def test_response_against_xspec():

    # Make a response and write to a FITS OGIP file
    matrix, mc_energies, ebounds = get_matrix_elements()

    rsp = InstrumentResponse(matrix, ebounds, mc_energies)

    temp_file = "__test.rsp"

    rsp.to_fits(temp_file, "TEST", "TEST", overwrite=True)

    # Test for various photon indexes

    for index in np.linspace(-2.0, 2.0, 10):

        if index == 1.0:

            # This would make the integral of the power law different, so let's just
            # skip it

            continue

        # First reset xspec
        xspec.AllData.clear()

        # Create a model in XSpec

        mo = xspec.Model("po")

        # Change the default value for the photon index
        # (remember that in XSpec the definition of the powerlaw is norm * E^(-PhoIndex),
        # so PhoIndex is positive normally. This is the opposite of astromodels.
        mo.powerlaw.PhoIndex = index
        mo.powerlaw.norm = 12.2

        # Now repeat the same in 3ML

        # Generate the astromodels function and set it to the same values as the XSpec power law
        # (the pivot in XSpec is set to 1). Remember also that the definition in xspec has the
        # sign of the photon index opposite
        powerlaw = Powerlaw()
        powerlaw.piv = 1.0
        powerlaw.index = -mo.powerlaw.PhoIndex.values[0]
        powerlaw.K = mo.powerlaw.norm.values[0]

        # Exploit the fact that the power law integral is analytic
        powerlaw_integral = Powerlaw()
        powerlaw_integral.K._transformation = None
        powerlaw_integral.K.bounds = (None, None)
        powerlaw_integral.index = powerlaw.index.value + 1
        powerlaw_integral.K = old_div(powerlaw.K.value, (powerlaw.index.value + 1))

        integral_function = lambda e1, e2: powerlaw_integral(e2) - powerlaw_integral(e1)

        # Now check that the two convoluted model give the same number of counts in each channel

        # Fake a spectrum so we can actually compute the convoluted model

        # Get path of response file

        fs1 = xspec.FakeitSettings(
            temp_file, exposure=1.0, fileName="_fake_spectrum.pha"
        )

        xspec.AllData.fakeit(noWrite=True, applyStats=False, settings=fs1)

        # Get the expected counts
        xspec_counts = mo.folded(1)

        # Now get the convolution from 3ML

        rsp.set_function(integral_function)

        threeML_counts = rsp.convolve()

        # Compare them
        assert np.allclose(xspec_counts, threeML_counts)

    os.remove(temp_file)
