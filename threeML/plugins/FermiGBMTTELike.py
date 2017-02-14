__author__ = 'grburgess'

import astropy.io.fits as fits
import numpy as np
import pandas as pd
import warnings
import collections
import re

from threeML.plugins.EventListLike import EventListLike
from threeML.plugins.OGIP.eventlist import EventListWithDeadTime
from threeML.plugins.OGIP.response import InstrumentResponseSet
from threeML.utils.fermi_relative_mission_time import compute_fermi_relative_mission_times

__instrument_name = "Fermi GBM TTE (all detectors)"


class BinningMethodError(RuntimeError):
    pass


class FermiGBMTTELike(EventListLike):
    def __init__(self, name, tte_file, rsp_file, source_intervals, background_selections=None, restore_background=None,
                 trigger_time=None,
                 poly_order=-1, unbinned=True, verbose=True):
        """
        A plugin to natively bin, view, and handle Fermi GBM TTE data.
        A TTE event file are required as well as the associated response



        Background selections are specified as
        a comma separated string e.g. "-10-0,10-20"

        Initial source selection is input as a string e.g. "0-5"

        One can choose a background polynomial order by hand (up to 4th order)
        or leave it as the default polyorder=-1 to decide by LRT test

        :param name: name for your choosing
        :param tte_file: GBM tte event file
        :param background_selections: comma sep. background intervals as string
        :param source_intervals: comma sep. source intervals as string
        :param rsp_file: Associated TTE CSPEC response file
        :param trigger_time: trigger time if needed
        :param poly_order: 0-4 or -1 for auto
        :param unbinned: unbinned likelihood fit (bool)
        :param verbose: verbose (bool)



        """

        self._default_unbinned = unbinned

        # Load the relevant information from the TTE file

        self._gbm_tte_file = GBMTTEFile(tte_file)

        # Set a trigger time if one has not been set

        if trigger_time is not None:
            self._gbm_tte_file.trigger_time = trigger_time

        # Create the the event list

        event_list = EventListWithDeadTime(
            arrival_times=self._gbm_tte_file.arrival_times - self._gbm_tte_file.trigger_time,
            energies=self._gbm_tte_file.energies,
            n_channels=self._gbm_tte_file.n_channels,
            start_time=self._gbm_tte_file.tstart - self._gbm_tte_file.trigger_time,
            stop_time=self._gbm_tte_file.tstop - self._gbm_tte_file.trigger_time,
            dead_time=self._gbm_tte_file.deadtime,
            first_channel=1,
            rsp_file=rsp_file,
            instrument=self._gbm_tte_file.det_name,
            mission=self._gbm_tte_file.mission,
            verbose=verbose)

        # we need to see if this is an RSP2

        test = re.match('^.*\.rsp2$', rsp_file)

        if test is not None:

            self._rsp_is_weighted = True

            self._rsp_set = InstrumentResponseSet.from_rsp2_file(rsp2_file=rsp_file,
                                                                 counts_getter=event_list.counts_over_interval,
                                                                 exposure_getter=event_list.exposure_over_interval,
                                                                 reference_time=self._gbm_tte_file.trigger_time)

            rsp_file = self._rsp_set.weight_by_counts(*[interval.replace(' ', '')
                                                        for interval in source_intervals.split(',')])

        else:

            self._rsp_is_weighted = False

        # pass to the super class

        super(FermiGBMTTELike, self).__init__(name,
                                              event_list,
                                              rsp_file=rsp_file,
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

        if self._rsp_is_weighted and not self._startup:
            self._rsp = self._rsp_set.weight_by_counts(*intervals)

        super(FermiGBMTTELike, self).set_active_time_interval(*intervals, **kwargs)

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
        super(FermiGBMTTELike, self).view_lightcurve(start=start,
                                                     stop=stop,
                                                     dt=dt,
                                                     use_binner=use_binner,
                                                     energy_selection=energy_selection,
                                                     significance_level=significance_level,
                                                     instrument='gbm')

    def _output(self):
        super_out = super(FermiGBMTTELike, self)._output()
        return super_out.append(self._gbm_tte_file._output())


class GBMTTEFile(object):
    def __init__(self, ttefile):
        """

        A simple class for opening and easily accessing Fermi GBM
        TTE Files.

        :param ttefile: The filename of the TTE file to be stored

        """

        tte = fits.open(ttefile)

        self._events = tte['EVENTS'].data['TIME']
        self._pha = tte['EVENTS'].data['PHA']

        try:
            self._trigger_time = tte['PRIMARY'].header['TRIGTIME']


        except:

            # For continuous data
            warnings.warn("There is no trigger time in the TTE file. Must be set manually or using MET relative times.")

            self._trigger_time = 0

        self._start_events = tte['PRIMARY'].header['TSTART']
        self._stop_events = tte['PRIMARY'].header['TSTOP']

        self._utc_start = tte['PRIMARY'].header['DATE-OBS']
        self._utc_stop = tte['PRIMARY'].header['DATE-END']

        self._n_channels = tte['EBOUNDS'].header['NAXIS2']

        self._det_name = "%s_%s" % (tte['PRIMARY'].header['INSTRUME'], tte['PRIMARY'].header['DETNAM'])

        self._telescope = tte['PRIMARY'].header['TELESCOP']

        self._calculate_deattime()

    @property
    def trigger_time(self):

        return self._trigger_time

    @trigger_time.setter
    def trigger_time(self, val):

        assert self._start_events <= val <= self._stop_events, "Trigger time must be within the interval (%f,%f)" % (
            self._start_events, self._stop_events)

        self._trigger_time = val

    @property
    def tstart(self):
        return self._start_events

    @property
    def tstop(self):
        return self._stop_events

    @property
    def arrival_times(self):
        return self._events

    @property
    def n_channels(self):
        return self._n_channels

    @property
    def energies(self):
        return self._pha

    @property
    def mission(self):
        """
        Return the name of the mission
        :return:
        """
        return self._telescope

    @property
    def det_name(self):
        """
        Return the name of the instrument and detector

        :return:
        """

        return self._det_name

    @property
    def deadtime(self):
        return self._deadtime

    def _calculate_deattime(self):
        """
        Computes an array of deadtimes following the perscription of Meegan et al. (2009).

        The array can be summed over to obtain the total dead time

        """
        self._deadtime = np.zeros_like(self._events)
        overflow_mask = self._pha == self._n_channels  # specific to gbm! should work for CTTE

        # From Meegan et al. (2009)
        # Dead time for overflow (note, overflow sometimes changes)
        self._deadtime[overflow_mask] = 10.E-6  # s

        # Normal dead time
        self._deadtime[~overflow_mask] = 2.E-6  # s

    def _compute_mission_times(self):

        mission_dict = {}

        if self.trigger_time == 0:
            return None

        # Complements to Volodymyr Savchenko

        xtime_url = "https://heasarc.gsfc.nasa.gov/cgi-bin/Tools/xTime/xTime.pl"

        pattern = """<tr>.*?<th scope=row><label for="(.*?)">(.*?)</label></th>.*?<td align=center>.*?</td>.*?<td>(.*?)</td>.*?</tr>"""

        args = dict(
            time_in_sf=self._trigger_time,
            timesys_in="u",
            timesys_out="u",
            apply_clock_offset="yes")

        if has_requests:

            try:

                content = requests.get(xtime_url, params=args).content

                mission_info = re.findall(pattern, content, re.S)

                mission_dict['UTC'] = mission_info[0][-1]
                mission_dict[mission_info[7][1]] = mission_info[7][2]  # LIGO
                mission_dict[mission_info[8][1]] = mission_info[8][2]  # NUSTAR
                mission_dict[mission_info[12][1]] = mission_info[12][2]  # RXTE
                mission_dict[mission_info[16][1]] = mission_info[16][2]  # SUZAKU
                mission_dict[mission_info[20][1]] = mission_info[20][2]  # SWIFT
                mission_dict[mission_info[24][1]] = mission_info[24][2]  # CHANDRA

            except:

                warnings.warn("You do not have the requests library, cannot get time system from Heasarc "
                              "at this point.")

                return None

        else:

            warnings.warn("You do not have the requests library, cannot get time system from Heasarc at this point.")

            return None

        return mission_dict

    def __repr__(self):

        return self._output().to_string()

    def _output(self):

        """
                Examine the currently selected interval
                If connected to the internet, will also look up info for other instruments to compare with
                Fermi.

                :return: none
                """
        mission_dict = compute_fermi_relative_mission_times(self._trigger_time)

        fermi_dict = collections.OrderedDict()

        fermi_dict['Fermi Trigger Time'] = "%.3f" % self._trigger_time
        fermi_dict['Fermi MET OBS Start'] = "%.3f" % self._start_events
        fermi_dict['Fermi MET OBS Stop'] = "%.3f" % self._stop_events
        fermi_dict['Fermi UTC OBS Start'] = self._utc_start
        fermi_dict['Fermi UTC OBS Stop'] = self._utc_stop

        fermi_df = pd.Series(fermi_dict, index=fermi_dict.keys())

        if mission_dict is not None:
            mission_df = pd.Series(mission_dict, index=mission_dict.keys())

            fermi_df = fermi_df.append(mission_df)

        return fermi_df
