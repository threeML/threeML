# Creates a generic event list reader that can create PHA objects on the fly

import numpy as np


class EventList(object):
    def __init__(self, arrival_times, energies, ra=None, dec=None):
        """
        Container for event style data which are tagged with time and energy/PHA.


        Args:
            arrival_times:
            energies:
            ra:
            dec:
        """

        self._arrival_times = np.asarray(arrival_times)
        self._energies = np.asarray(energies)

        assert self._arrival_times.shape[0] == self._energies.shape[
            0], "Arrival time (%d) and energies (%d) have different shapes" % (
        self._arrival_times.shape[0], self._energies.shape[0])

    @staticmethod
    def _parse_time_interval(time_interval):
        # The following regular expression matches any two numbers, positive or negative,
        # like "-10 --5","-10 - -5", "-10-5", "5-10" and so on

        tokens = re.match('(\-?\+?[0-9]+\.?[0-9]*)\s*-\s*(\-?\+?[0-9]+\.?[0-9]*)', time_interval).groups()

        return map(float, tokens)

    def set_active_time_interval(self, arg):
        '''Set the time interval to be used during the analysis.
        For now, only one interval can be selected. This may be
        updated in the future to allow for self consistent time
        resolved analysis.
        Specified as 'tmin-tmax'. Intervals are in seconds. Example:

        set_active_time_interval("0.0-10.0")

        which will set the energy range 0-10. seconds.
        '''

        self.tmin, self.tmax = self._parse_time_interval(arg)

        # First build the mas for the events in time
        timemask = np.logical_and(self._arrival_times >= self.tmin,
                                  self._arrival_times <= self.tmax)
        tmpcounts = []  # Temporary list to hold the total counts per chan
        tmpbackground = []  # Temporary list to hold the background counts per chan
        tmpbackgrounderr = []  # Temporary list to hold the background counts per chan

        for chan in range(self.ttefile.nchans):
            channelmask = self.ttefile.pha == chan
            countsmask = np.logical_and(channelmask, timemask)
            totalcounts = len(self.ttefile.events[countsmask])

            # Now integrate the appropriate background polynomial
            backgroundcounts = self._polynomials[chan].integral(self.tmin, self.tmax)
            backgrounderror = self._polynomials[chan].integralError(self.tmin, self.tmax)

            tmpcounts.append(totalcounts)
            tmpbackground.append(backgroundcounts)
            tmpbackgrounderr.append(backgrounderror)

        counts = np.array(tmpcounts)
        bkgCounts = np.array(tmpbackground)
        bkgErr = np.array(tmpbackgrounderr)

        # Calculate the exposure using the GBM dead time (Meegan et al. 2009)
        totaldeadtime = self.ttefile.deadtime[timemask].sum()

        exposure = (self.tmax - self.tmin) - totaldeadtime

    def get_pha(self):
        """
        Return a


        Returns:

        """
