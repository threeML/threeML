import os

import numpy as np
import pytest

from threeML.io.file_utils import within_directory
from threeML.utils.time_interval import TimeIntervalSet
from threeML.utils.time_series.event_list import EventListWithDeadTime, EventList
from threeML.utils.data_builders.time_series_builder import TimeSeriesBuilder
from threeML.io.file_utils import within_directory
from threeML.plugins.DispersionSpectrumLike import DispersionSpectrumLike
from threeML.plugins.OGIPLike import OGIPLike

import astropy.io.fits as fits

__this_dir__ = os.path.join(os.path.abspath(os.path.dirname(__file__)))
__example_dir = os.path.join(__this_dir__, '../../examples')


def is_within_tolerance(truth, value, relative_tolerance=0.01):
    assert truth != 0

    if abs((truth - value) / truth) <= relative_tolerance:

        return True

    else:

        return False


def test_event_list_constructor():
    dummy_times = np.linspace(0, 10, 10)
    dummy_deadtime = np.zeros_like(dummy_times)
    dummy_energy = np.zeros_like(dummy_times)
    start = 0
    stop = 10

    evt_list = EventList(arrival_times=dummy_times,
                         energies=dummy_energy,
                         n_channels=1,
                         start_time=start,
                         stop_time=stop)

    # should only have 10 events

    assert evt_list.n_events == 10

    with pytest.raises(RuntimeError):
        evt_list.bins

    with pytest.raises(AttributeError):
        evt_list.text_bins

    assert evt_list.poly_intervals is None

    with pytest.raises(AttributeError):
        evt_list.tmax_list

    with pytest.raises(AttributeError):
        evt_list.tmin_list

    assert evt_list.polynomials is None

    assert evt_list._instrument == 'UNKNOWN'

    assert evt_list._mission == 'UNKNOWN'

def test_unbinned_fit():

    with within_directory(__this_dir__):


        start, stop = 0, 50

        poly = [1]

        arrival_times = np.loadtxt('test_event_data.txt')

        evt_list = EventListWithDeadTime(arrival_times=arrival_times,
                                         energies=np.zeros_like(arrival_times),
                                         n_channels=1,
                                         start_time=arrival_times[0],
                                         stop_time=arrival_times[-1],
                                         dead_time=np.zeros_like(arrival_times)
                                         )

        evt_list.set_polynomial_fit_interval("%f-%f" % (start + 1, stop - 1), unbinned=True)

        results = evt_list.get_poly_info()['coefficients']

        evt_list.set_active_time_intervals("0-1")

        assert evt_list.time_intervals == TimeIntervalSet.from_list_of_edges([0, 1])

        assert evt_list._poly_counts.sum() > 0

        evt_list.__repr__()

def test_binned_fit():
    with within_directory(__this_dir__):
        start, stop = 0, 50

        poly = [1]



        arrival_times = np.loadtxt('test_event_data.txt')

        evt_list = EventListWithDeadTime(arrival_times=arrival_times,
                             energies=np.zeros_like(arrival_times),
                             n_channels=1,
                             start_time=arrival_times[0],
                             stop_time=arrival_times[-1],
                             dead_time=np.zeros_like(arrival_times)
                             )

        evt_list.set_polynomial_fit_interval("%f-%f" % (start + 1, stop - 1), unbinned=False)

        evt_list.set_active_time_intervals("0-1")

        results = evt_list.get_poly_info()['coefficients']

        assert evt_list.time_intervals == TimeIntervalSet.from_list_of_edges([0,1])


        assert evt_list._poly_counts.sum() > 0

        evt_list.__repr__()

def test_read_gbm_cspec():

    with within_directory(__example_dir):
        data_dir = os.path.join('gbm', 'bn080916009')

        nai3 = TimeSeriesBuilder.from_gbm_cspec_or_ctime('NAI3',
                                              os.path.join(data_dir, "glg_cspec_n3_bn080916009_v01.pha"),
                                              rsp_file=os.path.join(data_dir, "glg_cspec_n3_bn080916009_v00.rsp2"),
                                              poly_order=-1)



        nai3.set_active_time_interval('0-1')
        nai3.set_background_interval('-200--10','100-200')


        speclike = nai3.to_spectrumlike()

        assert isinstance(speclike,DispersionSpectrumLike)

        nai3.write_pha_from_binner('test_from_nai3',start=0, stop=2, overwrite=True)

def test_read_gbm_tte():
    with within_directory(__example_dir):

        data_dir = os.path.join('gbm', 'bn080916009')

        nai3 = TimeSeriesBuilder.from_gbm_tte('NAI3',
                                             os.path.join(data_dir, "glg_tte_n3_bn080916009_v01.fit.gz"),
                                             rsp_file=os.path.join(data_dir, "glg_cspec_n3_bn080916009_v00.rsp2"),
                                             poly_order=-1)



        nai3.set_active_time_interval('0-1')
        nai3.set_background_interval('-20--10','100-200')


        speclike = nai3.to_spectrumlike()

        assert isinstance(speclike,DispersionSpectrumLike)


        # test binning


        # should not have bins yet



        with pytest.raises(RuntimeError):
            nai3.bins

        # First catch the errors


        # This is without specifying the correct options name





        with pytest.raises(RuntimeError):
            nai3.create_time_bins(start=0, stop=10, method='constant')

        with pytest.raises(RuntimeError):
            nai3.create_time_bins(start=0, stop=10, method='significance')

        with pytest.raises(RuntimeError):
            nai3.create_time_bins(start=0, stop=10, method='constant', p0=.1)

        with pytest.raises(RuntimeError):
            nai3.create_time_bins(start=0, stop=10, method='significance', dt=1)

        # now incorrect options

        with pytest.raises(RuntimeError):
            nai3.create_time_bins(start=0, stop=10, method='not_a_method')

        # Now test values



        nai3.create_time_bins(start=0, stop=10, method='constant', dt=1)

        assert len(nai3.bins) == 10

        assert nai3.bins.argsort() == range(len(nai3.bins))

        nai3.create_time_bins(start=0, stop=10, method='bayesblocks', p0=.1)

        assert nai3.bins.argsort() == range(len(nai3.bins))

        assert len(nai3.bins) == 5

        nai3.create_time_bins(start=0, stop=10, method='significance', sigma=40)

        assert nai3.bins.argsort() == range(len(nai3.bins))

        assert len(nai3.bins) == 5

        nai3.view_lightcurve(use_binner=True)

        nai3.write_pha_from_binner('test_from_nai3', overwrite=True)



def test_reading_of_written_pha():
    with within_directory(__example_dir):



        # check the number of items written

        with fits.open('test_from_nai3.rsp') as f:

            # 2 ext + 5 rsp ext
            assert len(f) == 7


        # make sure we can read spectrum number

        ogip = OGIPLike('test',observation='test_from_nai3.pha',spectrum_number=1)
        ogip = OGIPLike('test', observation='test_from_nai3.pha', spectrum_number=2)





def test_read_lle():
    with within_directory(__example_dir):
        data_dir = 'lat'


        lle = TimeSeriesBuilder.from_lat_lle('lle', os.path.join(data_dir, "gll_lle_bn080916009_v10.fit"),
                              os.path.join(data_dir, "gll_pt_bn080916009_v10.fit"),
                              rsp_file=os.path.join(data_dir, "gll_cspec_bn080916009_v10.rsp"),
                              poly_order=-1)



        lle.view_lightcurve()

        lle.set_active_time_interval("0-10")

        lle.set_background_interval("-150-0", "100-250")

        speclike = lle.to_spectrumlike()

        assert isinstance(speclike, DispersionSpectrumLike)


        # will test background with lle data


        old_coefficients, old_errors = lle.get_background_parameters()

        old_tmin_list = lle._time_series.poly_intervals



        lle.save_background('temp_lle', overwrite=True)

        lle = TimeSeriesBuilder.from_lat_lle('lle', os.path.join(data_dir, "gll_lle_bn080916009_v10.fit"),
                                             os.path.join(data_dir, "gll_pt_bn080916009_v10.fit"),
                                             rsp_file=os.path.join(data_dir, "gll_cspec_bn080916009_v10.rsp"),
                                             restore_background='temp_lle.h5')





        new_coefficients, new_errors = lle.get_background_parameters()

        new_tmin_list = lle._time_series.poly_intervals


        assert new_coefficients == old_coefficients

        assert new_errors == old_errors



        assert old_tmin_list == new_tmin_list














