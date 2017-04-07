__author__ = 'grburgess'

import astropy.io.fits as fits
import numpy as np
import pandas as pd
import warnings
import collections
import re

from threeML.plugins.EventListLike import EventListLike
from threeML.plugins.OGIP.eventlist import EventListWithDeadTimeFraction
from threeML.io.cern_root_utils.io_utils import open_ROOT_file
from threeML.io.cern_root_utils.tobject_to_numpy import tree_to_ndarray

__instrument_name = "Fermi GBM TTE (all detectors)"


class BinningMethodError(RuntimeError):
    pass


class POLARLike(EventListLike):
    def __init__(self, name, polar_root_file, rsp_file, source_intervals, background_selections=None,
                 restore_background=None,
                 trigger_time=0.,
                 poly_order=-1, unbinned=True, verbose=True):
        """

        :param name:
        :param polar_root_file:
        :param rsp_file:
        :param source_intervals:
        :param background_selections:
        :param restore_background:
        :param trigger_time:
        :param poly_order:
        :param unbinned:
        :param verbose:
        """


        self._default_unbinned = unbinned

        # extract the polar varaibles

        self._polar_data = POLARData(polar_root_file, trigger_time, rsp_file)

        # TODO: get the response matrix out

        # Create the the event list

        event_list = EventListWithDeadTimeFraction(arrival_times=self._polar_data.time,
                                                   energies=self._polar_data.pha,
                                                   n_channels=20, # TODO: cahnge!
                                                   start_time=self._polar_data.time.min(),
                                                   stop_time=self._polar_data.time.max(),
                                                   dead_time_fraction=self._polar_data.dead_time_fraction,
                                                   verbose=verbose,
                                                   rsp_file='nothing') # TODO: cahnge

        # pass to the super class

        super(POLARLike, self).__init__(name,
                                        event_list,
                                        rsp_file=rsp, # TODO: change!
                                        source_intervals=source_intervals,
                                        background_selections=background_selections,
                                        poly_order=poly_order,
                                        unbinned=unbinned,
                                        verbose=verbose,
                                        restore_poly_fit=restore_background)

    def set_active_time_interval(self, *intervals, **kwargs):
        """
        Set the time interval to be used during the analysis.
        For now, only one interval can be selected. This may be
        updated in the future to allow for self consistent time
        resolved analysis.
        Specified as 'tmin-tmax'. Intervals are in seconds. Example:

        set_active_time_interval("0.0-10.0")

        which will set the energy range 0-10. seconds.
        :param options:
        :param intervals:
        :return:
        """

        super(POLARLike, self).set_active_time_interval(*intervals, **kwargs)

    def view_lightcurve(self, start=-10, stop=20., dt=1., use_binner=False, energy_selection=None,
                        significance_level=None):
        """

        :param use_binner: use the bins created via a binner
        :param start: start time to view
        :param stop:  stop time to view
        :param dt:  dt of the light curve
        :param energy_selection: string containing energy interval
        :return: fig
        """
        super(POLARLike, self).view_lightcurve(start=start,
                                               stop=stop,
                                               dt=dt,
                                               use_binner=use_binner,
                                               energy_selection=energy_selection,
                                               significance_level=significance_level,
                                               instrument='gbm')

    def _output(self):
        super_out = super(POLARLike, self)._output()
        return super_out

        #return super_out.append(self._gbm_tte_file._output())


class POLARData(object):
    def __init__(self, polar_root_file, reference_time=0., rsp_file=None):
        """
        container class that converts raw POLAR root data into useful python
        variables


        :param polar_root_file: path to polar event file
        :param reference_time: reference time of the events (tunix?)
        :param rsp_file: path to rsp file
        """


        # open the event file
        with open_ROOT_file(polar_root_file) as f:

            tmp = tree_to_ndarray(f.Get('polar_out'))

            # extract the pedestal corrected ADC channels
            # which are non-integer and possibly
            # less than zero
            pha = tmp['Energy']

            # non-zero ADC channels are invalid
            idx = pha >= 0
            pha = pha[idx]

            # get the dead time fraction
            self._dead_time_fraction = tmp['dead_ratio'][idx]

            # get the arrival time, in tunix of the events
            self._time = tmp['tunix'][idx] - reference_time

            # digitize the ADC channels into bins
            # these bins are preliminary

            self._binned_pha = np.digitize(pha, [0, 204.75,409.5, 614.25, 819, 1023.75, 1228.5,
                                                 1433.25, 1638, 1842.75, 2047.5, 2252.25, 2457,
                                                 2661.75, 2866.5,3071.25, 3276, 3480.75, 3685.5,
                                                 3890.25, 4095])

        # # open the event file
        # with open_ROOT_file(rsp_file) as f:
        #
        #






    @property
    def pha(self):
        return self._binned_pha

    @property
    def time(self):
        return self._time

    @property
    def dead_time_fraction(self):
        return self._dead_time_fraction


    @property
    def rsp(self):

        return self._rsp
