import pytest
import numpy as np
import os

__author__ = 'drjfunk'

from threeML.plugins.OGIPLike import OGIPLike
from threeML.plugins.OGIP.pha import PHA
from threeML.plugins.OGIP.response import Response

from threeML.io.file_utils import within_directory


def get_path(file):
    return os.path.join(os.path.dirname(__file__), file)


def test_loading_a_generic_pha_file():
    with within_directory('test'):
        ogip = OGIPLike('test_ogip', pha_file='test.pha{1}')

        pha_info = ogip.get_pha_files()

        assert ogip.name == 'test_ogip'
        assert ogip.n_data_points == 128
        assert sum(ogip._mask) == ogip.n_data_points
        assert ogip.tstart is None
        assert ogip.tstop is None
        assert 'cons_test_ogip' in ogip.nuisance_parameters
        assert ogip.nuisance_parameters['cons_test_ogip'].fix == True
        assert ogip.nuisance_parameters['cons_test_ogip'].free == False

        assert 'pha' in pha_info
        assert 'bak' in pha_info
        assert 'rsp' in pha_info


def test_pha_files_in_generic_ogip_constructor_spec_number_in_file_name():
    with within_directory('test'):

        ogip = OGIPLike('test_ogip', pha_file='test.pha{1}')

        pha_info = ogip.get_pha_files()

        for key in ['pha', 'bak']:

            assert isinstance(pha_info[key], PHA)

        assert pha_info['pha'].background_file == 'test_bak.pha{1}'
        assert pha_info['pha'].ancillary_file is None
        assert pha_info['pha'].instrument == 'GBM_NAI_03'
        assert pha_info['pha'].mission == 'GLAST'
        assert pha_info['pha'].is_poisson() == True
        assert pha_info['pha'].n_channels == ogip.n_data_points
        assert pha_info['pha'].n_channels == len(pha_info['pha'].rates)

        # Test that Poisson rates cannot call rate error
        with pytest.raises(AssertionError):

            _ = pha_info['pha'].rate_errors

        assert sum(pha_info['pha'].sys_errors == np.zeros_like(pha_info['pha'].rates)) == pha_info['bak'].n_channels

        assert pha_info['pha'].response_file == '../examples/gbm/bn080916009/glg_cspec_n3_bn080916009_v07.rsp'
        assert pha_info['pha'].scale_factor == 1.0

        # Test that we cannot get a bak file

        with pytest.raises(KeyError):

            _ = pha_info['bak'].background_file

        # Test that we cannot get a anc file
        with pytest.raises(KeyError):

            _ = pha_info['bak'].ancillary_file

            # Test that we cannot get a RSP file
            with pytest.raises(KeyError):
                _ = pha_info['bak'].response_file

        assert pha_info['bak'].instrument == 'GBM_NAI_03'
        assert pha_info['bak'].mission == 'GLAST'

        assert pha_info['bak'].is_poisson() == False

        assert pha_info['bak'].n_channels == ogip.n_data_points
        assert pha_info['bak'].n_channels == len(pha_info['pha'].rates)

        assert len(pha_info['bak'].rate_errors) == pha_info['bak'].n_channels

        assert sum(pha_info['bak'].sys_errors == np.zeros_like(pha_info['pha'].rates)) == pha_info['bak'].n_channels

        assert pha_info['bak'].scale_factor == 1.0

        assert isinstance(pha_info['rsp'], Response)


def test_pha_files_in_generic_ogip_constructor_spec_number_in_arguments():
    with within_directory('test'):
        ogip = OGIPLike('test_ogip', pha_file='test.pha', spectrum_number=1)

        pha_info = ogip.get_pha_files()

        for key in ['pha', 'bak']:

            assert isinstance(pha_info[key], PHA)

        assert pha_info['pha'].background_file == 'test_bak.pha{1}'
        assert pha_info['pha'].ancillary_file is None
        assert pha_info['pha'].instrument == 'GBM_NAI_03'
        assert pha_info['pha'].mission == 'GLAST'
        assert pha_info['pha'].is_poisson() == True
        assert pha_info['pha'].n_channels == ogip.n_data_points
        assert pha_info['pha'].n_channels == len(pha_info['pha'].rates)

        # Test that Poisson rates cannot call rate error
        with pytest.raises(AssertionError):

            _ = pha_info['pha'].rate_errors

        assert sum(pha_info['pha'].sys_errors == np.zeros_like(pha_info['pha'].rates)) == pha_info['bak'].n_channels
        assert pha_info['pha'].response_file == '../examples/gbm/bn080916009/glg_cspec_n3_bn080916009_v07.rsp'
        assert pha_info['pha'].scale_factor == 1.0

        # Test that we cannot get a bak file

        with pytest.raises(KeyError):

            _ = pha_info['bak'].background_file

        # Test that we cannot get a anc file
        with pytest.raises(KeyError):

            _ = pha_info['bak'].ancillary_file

            # Test that we cannot get a RSP file
            with pytest.raises(KeyError):
                _ = pha_info['bak'].response_file

        assert pha_info['bak'].instrument == 'GBM_NAI_03'
        assert pha_info['bak'].mission == 'GLAST'

        assert pha_info['bak'].is_poisson() == False

        assert pha_info['bak'].n_channels == ogip.n_data_points
        assert pha_info['bak'].n_channels == len(pha_info['pha'].rates)

        assert len(pha_info['bak'].rate_errors) == pha_info['bak'].n_channels

        assert sum(pha_info['bak'].sys_errors == np.zeros_like(pha_info['pha'].rates)) == pha_info['bak'].n_channels

        assert pha_info['bak'].scale_factor == 1.0

        assert isinstance(pha_info['rsp'], Response)


def test_ogip_energy_selection():
    with within_directory('test'):
        ogip = OGIPLike('test_ogip', pha_file='test.pha{1}')

        # assert sum(ogip._mask) == sum(ogip._quality_to_mask())

        ogip.set_active_measurements("10-30")

        assert sum(ogip._mask) == ogip.n_data_points
        assert sum(ogip._mask) < 128

        ogip.set_active_measurements("all")

        assert sum(ogip._mask) == ogip.n_data_points
        assert sum(ogip._mask) == 128

        ogip.set_active_measurements(exclude=['c0-c1'])

        assert sum(ogip._mask) == ogip.n_data_points
        assert sum(ogip._mask) == 126

        ogip.set_active_measurements(exclude=['0-c1'])

        assert sum(ogip._mask) == ogip.n_data_points
        assert sum(ogip._mask) == 126

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


            # ogip.set_active_measurements('reset')
            # assert sum(ogip._mask) == sum(ogip._quality_to_mask())


def test_ogip_rebinner():
    with within_directory('test'):
        ogip = OGIPLike('test_ogip', pha_file='test.pha{1}')

        n_data_points = 128
        ogip.set_active_measurements("all")

        assert ogip.n_data_points == n_data_points

        ogip.rebin_on_background(min_number_of_counts=100)

        assert ogip.n_data_points < 128

        with pytest.raises(AssertionError):
            ogip.set_active_measurements('all')

        ogip.remove_rebinning()

        assert ogip._rebinner is None

        assert ogip.n_data_points == n_data_points
