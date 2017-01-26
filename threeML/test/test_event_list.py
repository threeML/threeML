from threeML.plugins.OGIP.eventlist import EventListWithDeadTime, EventList
from threeML.utils.time_interval import TimeIntervalSet

import numpy as np
import pytest


class PoissonEventGenerator(object):
    def __init__(self, poly_coeff=[]):

        self._coeff = poly_coeff
        self._coeff.reverse()

        self._function = np.poly1d(self._coeff)

    def non_homogeneous_poisson_generator(self, tstart, tstop):
        """
        Non-homogeneous poisson process generator
        for a given max rate and time range, this function
        generates time tags sampled from the energy integrated
        lightcurve.


        """

        num_time_steps = 1000

        time_grid = np.linspace(tstart, tstop + 1., num_time_steps)
        tmp = self._function(time_grid)

        fmax = tmp.max()

        time = tstart

        arrival_times = [tstart]

        while time < tstop:

            time = time - (1. / fmax) * np.log(np.random.rand())
            test = np.random.rand()

            p_test = self._function(time) / fmax

            if test <= p_test:
                arrival_times.append(time)

        return np.array(arrival_times)


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
    start, stop = 0, 50

    poly = [1]

    pe = PoissonEventGenerator(poly_coeff=poly)

    arrival_times = pe.non_homogeneous_poisson_generator(start, stop)

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
    start, stop = 0, 50

    poly = [1]

    pe = PoissonEventGenerator(poly_coeff=poly)

    arrival_times = pe.non_homogeneous_poisson_generator(start, stop)

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

        ####
