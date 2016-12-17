from threeML.plugins.OGIP.eventlist import EventList

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
