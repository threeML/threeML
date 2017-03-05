from threeML.plugins.OGIP.eventlist import EventListWithDeadTime, EventList
from threeML.utils.time_interval import TimeIntervalSet
import numpy as np
import pytest
from threeML.io.file_utils import within_directory
import os

__this_dir__ = os.path.join(os.path.abspath(os.path.dirname(__file__)))


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

    with pytest.raises(AttributeError):
        evt_list.bins

    with pytest.raises(AttributeError):
        evt_list.text_bins

    with pytest.raises(AttributeError):
        evt_list.poly_intervals

    with pytest.raises(AttributeError):
        evt_list.tmax_list

    with pytest.raises(AttributeError):
        evt_list.tmin_list

    assert evt_list.polynomials == None

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


