import numpy as np
import os
import pytest
import warnings

from threeML.io.package_data import get_path_of_data_file
from threeML.utils.OGIP.response import (
    InstrumentResponseSet,
    InstrumentResponse,
    OGIPResponse,
)
from threeML.utils.time_interval import TimeInterval


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


def get_matrix_set_elements():

    matrix, mc_energies, ebounds = get_matrix_elements()

    rsp_a = InstrumentResponse(matrix, ebounds, mc_energies)

    # Make another matrix with the same matrix but divided by 2
    other_matrix = matrix / 2.0

    rsp_b = InstrumentResponse(other_matrix, ebounds, mc_energies)

    # Remember: the second matrix is like the first one divided by two, and it covers twice as much time.
    # They cover 0-10 s the first one, and 10-30 the second one.

    # Fake an exposure getter by using a fixed 10% deadtime
    livetime_fraction = 0.9
    exposure_getter = lambda t1, t2: livetime_fraction * (t2 - t1)

    # Fake a count getter
    law = lambda x: 1.23 * x
    # The counts getter is the integral of the law
    counts_getter = (
        lambda t1, t2: 1.23 * 0.5 * (t2 ** 2.0 - t1 ** 2.0) * livetime_fraction
    )

    return [rsp_a, rsp_b], exposure_getter, counts_getter


def get_matrix_set_elements_with_coverage(reference_time=0.0):

    [rsp_a, rsp_b], exposure_getter, counts_getter = get_matrix_set_elements()

    # By making the coverage interval twice for the second matrix we restore parity with the first one,
    # so that the weighting by exposure should simply return the first matrix

    rsp_a._coverage_interval = TimeInterval(0.0, 10.0) + reference_time
    rsp_b._coverage_interval = TimeInterval(10.0, 30.0) + reference_time

    return [rsp_a, rsp_b], exposure_getter, counts_getter


def test_instrument_response_constructor():

    # Make a fake test matrix

    matrix, mc_energies, ebounds = get_matrix_elements()

    rsp = InstrumentResponse(matrix, ebounds, mc_energies)

    assert np.all(rsp.matrix == matrix)
    assert np.all(rsp.ebounds == ebounds)
    assert np.all(rsp.monte_carlo_energies == mc_energies)

    # Now with coverage interval

    with pytest.raises(AssertionError):

        _ = InstrumentResponse(matrix, ebounds, mc_energies, "10-20")

    rsp = InstrumentResponse(matrix, ebounds, mc_energies, TimeInterval(10.0, 20.0))

    assert rsp.rsp_filename is None
    assert rsp.arf_filename is None
    assert rsp.coverage_interval == TimeInterval(10.0, 20.0)

    # Check that we do not accept nans in the matrix
    matrix[2, 2] = np.nan

    with pytest.raises(AssertionError):

        _ = InstrumentResponse(matrix, ebounds, mc_energies, "10-20")


def test_instrument_response_replace_matrix():

    matrix, mc_energies, ebounds = get_matrix_elements()

    rsp = InstrumentResponse(matrix, ebounds, mc_energies)

    new_matrix = matrix / 2.0

    rsp.replace_matrix(new_matrix)

    assert np.all(rsp.matrix == new_matrix)

    with pytest.raises(AssertionError):

        rsp.replace_matrix(np.random.uniform(0, 1, 100).reshape(10, 10))


def test_instrument_response_set_function_and_convolve():

    # A very basic test. More tests will be made against XSpec later

    matrix, mc_energies, ebounds = get_matrix_elements()

    rsp = InstrumentResponse(matrix, ebounds, mc_energies)

    # Integral of a constant, so we know easily what the output should be

    #integral_function = lambda e1, e2: e2 - e1

    def integral_function():
        return np.array(mc_energies)[1:] - np.array(mc_energies)[:-1]

    
    rsp.set_function(integral_function)

    folded_counts = rsp.convolve()

    assert np.all(folded_counts == [1.0, 2.0, 3.0])


def test__instrument_response_energy_to_channel():

    matrix, mc_energies, ebounds = get_matrix_elements()

    rsp = InstrumentResponse(matrix, ebounds, mc_energies)

    assert rsp.energy_to_channel(1.5) == 0
    assert rsp.energy_to_channel(2.6) == 1
    assert rsp.energy_to_channel(4.75) == 2
    assert rsp.energy_to_channel(100.0) == 3


def test_instrument_response_plot_response():

    matrix, mc_energies, ebounds = get_matrix_elements()

    rsp = InstrumentResponse(matrix, ebounds, mc_energies)

    rsp.plot_matrix()


def test_OGIP_response_first_channel():

    # Get path of response file
    rsp_file = get_path_of_data_file("ogip_test_gbm_n6.rsp")

    rsp = OGIPResponse(rsp_file)

    assert rsp.first_channel == 1


def test_OGIP_response_arf_rsp_accessors():

    # Then load rsp and arf in XSpec

    rsp_file = get_path_of_data_file("ogip_test_xmm_pn.rmf")

    arf_file = get_path_of_data_file("ogip_test_xmm_pn.arf")

    rsp = OGIPResponse(rsp_file, arf_file=arf_file)

    assert rsp.arf_filename == arf_file
    assert rsp.rsp_filename == rsp_file


def test_response_write_to_fits1():

    matrix, mc_energies, ebounds = get_matrix_elements()

    rsp = InstrumentResponse(matrix, ebounds, mc_energies)

    temp_file = "__test.rsp"

    rsp.to_fits(temp_file, "TEST", "TEST", overwrite=True)

    # Now check that reloading gives back the same matrix
    rsp_reloaded = OGIPResponse(temp_file)

    assert np.allclose(rsp_reloaded.matrix, rsp.matrix)
    assert np.allclose(rsp_reloaded.ebounds, rsp.ebounds)
    assert np.allclose(rsp_reloaded.monte_carlo_energies, rsp.monte_carlo_energies)

    os.remove(temp_file)


def test_response_write_to_fits2():

    # Now do the same for a response read from a file

    rsp_file = get_path_of_data_file("ogip_test_gbm_n6.rsp")

    rsp = OGIPResponse(rsp_file)

    temp_file = "__test.rsp"

    rsp.to_fits(temp_file, "TEST", "TEST", overwrite=True)

    rsp_reloaded = OGIPResponse(temp_file)

    assert np.allclose(rsp_reloaded.matrix, rsp.matrix)
    assert np.allclose(rsp_reloaded.ebounds, rsp.ebounds)
    assert np.allclose(rsp_reloaded.monte_carlo_energies, rsp.monte_carlo_energies)

    os.remove(temp_file)


def test_response_write_to_fits3():

    # Now do the same for a file with a ARF

    rsp_file = get_path_of_data_file("ogip_test_xmm_pn.rmf")

    arf_file = get_path_of_data_file("ogip_test_xmm_pn.arf")

    rsp = OGIPResponse(rsp_file, arf_file=arf_file)

    temp_file = "__test.rsp"

    rsp.to_fits(temp_file, "TEST", "TEST", overwrite=True)

    rsp_reloaded = OGIPResponse(temp_file)

    assert np.allclose(rsp_reloaded.matrix, rsp.matrix)
    assert np.allclose(rsp_reloaded.ebounds, rsp.ebounds)
    assert np.allclose(rsp_reloaded.monte_carlo_energies, rsp.monte_carlo_energies)

    os.remove(temp_file)


def test_response_set_constructor():

    [rsp_aw, rsp_bw], exposure_getter, counts_getter = get_matrix_set_elements()

    with pytest.raises(RuntimeError):

        # This should raise because there is no time information for the matrices

        _ = InstrumentResponseSet([rsp_aw, rsp_bw], exposure_getter, counts_getter)

    # Add the time information

    (
        [rsp_a, rsp_b],
        exposure_getter,
        counts_getter,
    ) = get_matrix_set_elements_with_coverage()

    # This should work now
    rsp_set = InstrumentResponseSet([rsp_a, rsp_b], exposure_getter, counts_getter)

    assert rsp_set[0] == rsp_a
    assert rsp_set[1] == rsp_b

    # Check that the constructor order the matrices by time when needed
    # This should work now
    rsp_set = InstrumentResponseSet([rsp_b, rsp_a], exposure_getter, counts_getter)

    assert rsp_set[0] == rsp_a
    assert rsp_set[1] == rsp_b

    # Now test construction from the .from_rsp2 method
    rsp2_file = get_path_of_data_file("ogip_test_gbm_b0.rsp2")

    with warnings.catch_warnings():

        warnings.simplefilter("error", np.VisibleDeprecationWarning)

        rsp_set = InstrumentResponseSet.from_rsp2_file(
            rsp2_file, exposure_getter, counts_getter
        )

    assert len(rsp_set) == 3

    # Now test that we cannot initialize a response set with matrices which have non-contiguous coverage intervals
    matrix, mc_energies, ebounds = get_matrix_elements()

    rsp_c = InstrumentResponse(matrix, ebounds, mc_energies, TimeInterval(0.0, 10.0))
    rsp_d = InstrumentResponse(matrix, ebounds, mc_energies, TimeInterval(20.0, 30.0))

    with pytest.raises(RuntimeError):

        _ = InstrumentResponseSet([rsp_c, rsp_d], exposure_getter, counts_getter)


def test_response_set_weighting():

    (
        [rsp_a, rsp_b],
        exposure_getter,
        counts_getter,
    ) = get_matrix_set_elements_with_coverage()

    rsp_set = InstrumentResponseSet([rsp_a, rsp_b], exposure_getter, counts_getter)

    # here we are waiting by exposure. We have:

    # weight1 = (0.9 * 5.0) = 4.5
    # weight2 = (0.9 * 15.0) = 13.5
    # sum = weight1 + weight2 = 18.0
    # new_matrix = rsp_a * weight1/sum + rsp_b * weight2 / sum

    # but rsp_b = rsp_a / 2.0, so:

    # new_matrix = rsp_a * weight1 / sum + rsp_a / 2.0 * weight2 / sum = 1 / sum * rsp_a * (weight1 + weight2 / 2.0)

    # so in the end:

    # new_matrix = 0.625 * rsp_a

    weighted_matrix = rsp_set.weight_by_exposure("5.0 - 25.0")

    assert np.allclose(weighted_matrix.matrix, 0.625 * rsp_a.matrix)

    # here we are waiting by exposure. We have:

    # weight1 = 55.35
    # weight2 = 442.8

    # so:

    # new_matrix = 1 / sum * rsp_a * (weight1 + weight2 / 2.0) = 0.5555555555555555 * rsp_a

    weighted_matrix = rsp_set.weight_by_counts("0.0 - 30.0")

    assert np.allclose(weighted_matrix.matrix, 0.5555555555555555 * rsp_a.matrix)

    # Here we weight by counts in the interval 5.0 - 25.0
    # With the same math as before:

    weighted_matrix = rsp_set.weight_by_counts("5.0 - 25.0")

    assert np.allclose(weighted_matrix.matrix, 0.5625000000000001 * rsp_a.matrix)


def test_response_set_weighting_with_reference_time():

    # Now repeat the same tests but using a reference time
    ref_time = 123.456

    (
        [rsp_a, rsp_b],
        exposure_getter,
        counts_getter,
    ) = get_matrix_set_elements_with_coverage(reference_time=ref_time)

    rsp_set = InstrumentResponseSet(
        [rsp_a, rsp_b], exposure_getter, counts_getter, reference_time=ref_time
    )

    assert rsp_set.reference_time == ref_time

    weighted_matrix = rsp_set.weight_by_exposure("5.0 - 25.0")

    assert np.allclose(weighted_matrix.matrix, 0.625 * rsp_a.matrix)

    weighted_matrix = rsp_set.weight_by_counts("0.0 - 30.0")

    assert np.allclose(weighted_matrix.matrix, 0.5555555555555555 * rsp_a.matrix)

    weighted_matrix = rsp_set.weight_by_counts("5.0 - 25.0")

    assert np.allclose(weighted_matrix.matrix, 0.5625000000000001 * rsp_a.matrix)


def test_response_set_weighting_with_disjoint_intervals():

    ref_time = 123.456

    (
        [rsp_a, rsp_b],
        exposure_getter,
        counts_getter,
    ) = get_matrix_set_elements_with_coverage(reference_time=ref_time)

    rsp_set = InstrumentResponseSet(
        [rsp_a, rsp_b], exposure_getter, counts_getter, reference_time=ref_time
    )

    assert rsp_set.reference_time == ref_time

    weighted_matrix = rsp_set.weight_by_exposure("5.0 - 12.0", "25.0-28.0")

    # weight1 = (0.9 * 5.0) = 4.5
    # weight2 = (0.9 * 2.0) = 1.8
    # weight3 = (0.9 * 3.0) = 2.7
    # sum = weight1 + weight2 + weight3 = 8.2
    # new_matrix = rsp_a * weight1/sum + rsp_b * weight2 / sum + rsp_b * weight3 / sum

    # but rsp_b = rsp_a / 2.0, so:

    # new_matrix = rsp_a * weight1 / sum + rsp_a / 2.0 * weight2 / sum + rsp_a / 2.0 * weight3 / sum

    # so in the end:

    # new_matrix = 1.0 / (w1 + w2 + w3) * (w1 + w2 / 2.0 + w3 / 2.0) * rsp_a = 0.75 * rsp_a

    assert np.allclose(weighted_matrix.matrix, 0.75 * rsp_a.matrix)

    # Now the same with counts

    weighted_matrix = rsp_set.weight_by_counts("5.0 - 12.0", "25.0-28.0")

    w1 = counts_getter(5.0, 10.0)
    w2 = counts_getter(10.0, 12.0)
    w3 = counts_getter(25.0, 28.0)

    factor = 1.0 / (w1 + w2 + w3) * (w1 + w2 / 2.0 + w3 / 2.0)

    assert np.allclose(weighted_matrix.matrix, factor * rsp_a.matrix)
