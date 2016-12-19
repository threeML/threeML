__author__ = 'grburgess'

import astropy.io.fits as fits
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import re

try:

    import requests

except ImportError:

    has_requests = False

else:

    has_requests = True

# import copy

# from threeML.plugins.OGIPLike import OGIPLike
from threeML.plugins.EventListLike import EventListLike
from threeML.plugins.OGIP.eventlist import EventListWithDeadTime
from threeML.io.rich_display import display

from threeML.io.step_plot import step_plot
#from threeML.plugins.OGIP.pha import PHAWrite

from threeML.config.config import threeML_config

__instrument_name = "Fermi GBM TTE (all detectors)"


class BinningMethodError(RuntimeError):
    pass


class FermiGBMTTELike(EventListLike):
    def __init__(self, name, tte_file, background_selections, source_intervals, rsp_file, trigger_time=None,
                 poly_order=-1, unbinned=True, verbose=True):
        """
        A plugin to natively bin, view, and handle Fermi GBM TTE data.
        An LLE event file and FT2 (1 sec) are required as well as the associated response



        Background selections are specified as
        a comma separated string e.g. "-10-0,10-20"

        Initial source selection is input as a string e.g. "0-5"

        One can choose a background polynomial order by hand (up to 4th order)
        or leave it as the default polyorder=-1 to decide by LRT test



        """

        self._default_unbinned = unbinned

        # Load the relevant information from the TTE file

        self._gbm_tte_file = GBMTTEFile(tte_file)

        # Set a trigger time if one has not been set

        if trigger_time is not None:
            self._gbm_tte_file.triggertime = trigger_time

        # Create the the event list

        event_list = EventListWithDeadTime(
            arrival_times=self._gbm_tte_file.arrival_times - self._gbm_tte_file.triggertime,
                energies=self._gbm_tte_file.energies,
            n_channels=self._gbm_tte_file.n_channels,
            start_time=self._gbm_tte_file.start_events - self._gbm_tte_file.triggertime,
            stop_time=self._gbm_tte_file.stop_events - self._gbm_tte_file.triggertime,
            dead_time=self._gbm_tte_file.deadtime,
                                   first_channel=0,
                                   rsp_file=rsp_file, instrument=self._gbm_tte_file.det_name,
                                   mission=self._gbm_tte_file.mission)

        # pass to the super class

        EventListLike.__init__(self, name, event_list, background_selections, source_intervals, rsp_file,
                               poly_order, unbinned, verbose)




    def view_lightcurve(self, start=-10, stop=20., dt=1., use_binner=False, energy_selection=None):
        """

        :param use_binner: use the bins created via a binner
        :param start: start time to view
        :param stop:  stop time to view
        :param dt:  dt of the light curve
        :param energy_selection: string containing energy interval
        :return: fig
        """

        if energy_selection is not None:

            energy_selection = [interval.replace(' ', '') for interval in energy_selection.split(',')]

            valid_channels = []
            mask = np.array([False] * self._evt_list.n_events)

            for selection in energy_selection:

                ee = map(float, selection.split("-"))

                if len(ee) != 2:
                    raise RuntimeError('Energy selection is not valid! Form: <low>-<high>.')

                emin, emax = sorted(ee)

                idx1 = self._rsp.energy_to_channel(emin)
                idx2 = self._rsp.energy_to_channel(emax)

                # Update the allowed channels
                valid_channels.extend(range(idx1, idx2))

                this_mask = np.logical_and(self._evt_list.energies >= idx1, self._evt_list.energies <= idx2)

                np.logical_or(mask, this_mask, out=mask)

        else:

            mask = np.array([True] * self._evt_list.n_events)
            valid_channels = range(self._gbm_tte_file.n_channels)

        if use_binner:

            bin_start, bin_stop = self._evt_list.bins
            bins = bin_start.tolist() + [bin_stop.tolist()[-1]]  # type: list

            # perhaps we want to look a little before or after the binner
            if start < bins[0]:

                pre_bins = np.arange(start, bins[0], dt).tolist()[:-1]

                pre_bins.extend(bins)

                bins = pre_bins

            if stop > bins[-1]:

                post_bins = np.arange(bins[-1], stop, dt)

                bins.extend(post_bins[1:])

        else:

            bins = np.arange(start, stop + dt, dt)

        cnts, bins = np.histogram(self._gbm_tte_file.arrival_times[mask] - self._gbm_tte_file.triggertime, bins=bins)
        time_bins = np.array([[bins[i], bins[i + 1]] for i in range(len(bins) - 1)])

        width = np.diff(bins)

        bkg = []
        for j, tb in enumerate(time_bins):
            tmpbkg = 0.
            for i in valid_channels:
                poly = self._evt_list.polynomials[i]

                tmpbkg += poly.integral(tb[0], tb[1]) / (width[j])

            bkg.append(tmpbkg)

        gbm_light_curve_plot(time_bins, cnts, bkg, width,
                             selection=zip(self._evt_list.tmin_list, self._evt_list._tmax_list),
                             bkg_selections=self._evt_list.poly_intervals)

    def peek(self):

        print "TTE File Info:"

        self._evt_list.peek()

        print 'Timing Info:'

        self._gbm_tte_file.peek()




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
            self.triggertime = tte['PRIMARY'].header['TRIGTIME']


        except:

            # For continuous data
            warnings.warn("There is no trigger time in the TTE file. Must me set manually or using MET relative times.")

            self.triggertime = 0

        self._start_events = tte['PRIMARY'].header['TSTART']
        self._stop_events = tte['PRIMARY'].header['TSTOP']

        self._utc_start = tte['PRIMARY'].header['DATE-OBS']
        self._utc_stop = tte['PRIMARY'].header['DATE-END']

        self._n_channels = tte['EBOUNDS'].header['NAXIS2']

        self._det_name = "%s_%s" % (tte['PRIMARY'].header['INSTRUME'], tte['PRIMARY'].header['DETNAM'])

        self._telescope = tte['PRIMARY'].header['TELESCOP']

        self._calculate_deattime()

    @property
    def start_events(self):
        return self._start_events

    @property
    def stop_events(self):
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

        if self.triggertime == 0:
            return None

        # Complements to Volodymyr Savchenko

        xtime_url = "https://heasarc.gsfc.nasa.gov/cgi-bin/Tools/xTime/xTime.pl"

        pattern = """<tr>.*?<th scope=row><label for="(.*?)">(.*?)</label></th>.*?<td align=center>.*?</td>.*?<td>(.*?)</td>.*?</tr>"""

        args = dict(
                time_in_sf=self.triggertime,
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

    def peek(self):
        """
        Examine the currently selected interval
        If connected to the internet, will also look up info for other instruments to compare with
        Fermi.

        :return: none
        """

        mission_dict = self._compute_mission_times()

        fermi_dict = {}

        fermi_dict['Fermi Trigger Time'] = self.triggertime
        fermi_dict['Fermi MET OBS Start'] = self._start_events
        fermi_dict['Fermi MET OBS Stop'] = self._stop_events
        fermi_dict['Fermi UTC OBS Start'] = self._utc_start
        fermi_dict['Fermi UTC OBS Stop'] = self._utc_stop

        if mission_dict is not None:
            mission_df = pd.Series(mission_dict)

            display(mission_df)

        fermi_df = pd.Series(fermi_dict)

        display(fermi_df)





def gbm_light_curve_plot(time_bins, cnts, bkg, width, selection, bkg_selections):
    fig, ax = plt.subplots()

    max_cnts = max(cnts / width)
    top = max_cnts + max_cnts * .2
    min_cnts = min(cnts[cnts > 0] / width[cnts > 0])
    bottom = min_cnts - min_cnts * .05
    mean_time = map(np.mean, time_bins)

    all_masks = []

    # purple: #8da0cb

    step_plot(time_bins, cnts / width, ax,
              color=threeML_config['gbm']['lightcurve color'], label="Light Curve")

    for tmin, tmax in selection:
        tmp_mask = np.logical_and(time_bins[:, 0] >= tmin, time_bins[:, 1] <= tmax)

        all_masks.append(tmp_mask)

    if len(all_masks) > 1:

        for mask in all_masks[1:]:
            step_plot(time_bins[mask], cnts[mask] / width[mask], ax,
                      color=threeML_config['gbm']['selection color'],
                      fill=True,
                      fill_min=min_cnts)

    step_plot(time_bins[all_masks[0]], cnts[all_masks[0]] / width[all_masks[0]], ax,
              color=threeML_config['gbm']['selection color'],
              fill=True,
              fill_min=min_cnts, label="Selection")

    all_masks = []
    for tmin, tmax in bkg_selections:
        tmp_mask = np.logical_and(time_bins[:, 0] >= tmin, time_bins[:, 1] <= tmax)

        all_masks.append(tmp_mask)

    if len(all_masks) > 1:

        for mask in all_masks[1:]:

            step_plot(time_bins[mask], cnts[mask] / width[mask], ax,
                      color=threeML_config['gbm']['background selection color'],
                      fill=True,
                      fillAlpha=.4,
                      fill_min=min_cnts)

    step_plot(time_bins[all_masks[0]], cnts[all_masks[0]] / width[all_masks[0]], ax,
              color=threeML_config['gbm']['background selection color'],
              fill=True,
              fill_min=min_cnts, fillAlpha=.4, label="Bkg. Selections")

    ax.plot(mean_time, bkg, threeML_config['gbm']['background color'], lw=2., label="Background")

    # ax.fill_between(selection, bottom, top, color="#fc8d62", alpha=.4)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Rate (cnts/s)")
    ax.set_ylim(bottom, top)
    ax.set_xlim(time_bins.min(), time_bins.max())
    ax.legend()
