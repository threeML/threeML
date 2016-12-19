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

from threeML.plugins.EventListLike import EventListLike
from threeML.plugins.OGIP.eventlist import EventListWithLiveTime
from threeML.io.rich_display import display

from threeML.io.step_plot import step_plot

from threeML.config.config import threeML_config

__instrument_name = "Fermi LAT LLE"


class BinningMethodError(RuntimeError):
    pass


class FermiLATLLELike(EventListLike):
    def __init__(self, name, lle_file, ft2_file, background_selections, source_intervals, rsp_file, trigger_time=None,
                 poly_order=-1, unbinned=False, verbose=True):
        """
        A plugin to natively bin, view, and handle Fermi LAT LLE data.
        An LLE event file and FT2 (1 sec) are required as well as the associated response



        Background selections are specified as
        a comma separated string e.g. "-10-0,10-20"

        Initial source selection is input as a string e.g. "0-5"

        One can choose a background polynomial order by hand (up to 4th order)
        or leave it as the default polyorder=-1 to decide by LRT test

        :param name: name of the plugin
        :param lle_file: lle event file
        :param ft2_file: fermi FT2 file
        :param background_selections: comma sep. background intervals as string
        :param source_intervals: comma sep. source intervals as string
        :param rsp_file: lle response file
        :param trigger_time: trigger time if needed
        :param poly_order: 0-4 or -1 for auto
        :param unbinned: unbinned likelihood fit (bool)
        :param verbose: verbose (bool)


        """


        self._default_unbinned = unbinned

        self._lat_lle_file = LLEFile(lle_file, ft2_file, rsp_file)

        if trigger_time is not None:
            self._lat_lle_file.triggertime = trigger_time

        event_list = EventListWithLiveTime(
                arrival_times=self._lat_lle_file.arrival_times - self._lat_lle_file.triggertime,
                energies=self._lat_lle_file.energies,
                n_channels=self._lat_lle_file.n_channels,
                live_time=self._lat_lle_file.livetime,
                live_time_starts=self._lat_lle_file.livetime_start - self._lat_lle_file.triggertime,
                live_time_stops=self._lat_lle_file.livetime_stop - self._lat_lle_file.triggertime,
                start_time=self._lat_lle_file._start_events - self._lat_lle_file.triggertime,
                stop_time=self._lat_lle_file._stop_events - self._lat_lle_file.triggertime,
                first_channel=1,
                rsp_file=rsp_file,
                instrument=self._lat_lle_file.instrument,
                mission=self._lat_lle_file.mission)

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
            valid_channels = range(self._lat_lle_file.n_channels)

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

        cnts, bins = np.histogram(self._lat_lle_file.arrival_times[mask] - self._lat_lle_file.triggertime, bins=bins)
        time_bins = np.array([[bins[i], bins[i + 1]] for i in range(len(bins) - 1)])

        width = np.diff(bins)

        bkg = []
        for j, tb in enumerate(time_bins):
            tmpbkg = 0.
            for i in valid_channels:
                poly = self._evt_list.polynomials[i]

                tmpbkg += poly.integral(tb[0], tb[1]) / (width[j])

            bkg.append(tmpbkg)

        lle_light_curve_plot(time_bins, cnts, bkg, width,
                             selection=zip(self._evt_list.tmin_list, self._evt_list.tmax_list),
                             bkg_selections=self._evt_list.poly_intervals)

    def peek(self):

        print("LLE File Info:")

        self._evt_list.peek()

        print('Timing Info:')

        self._lat_lle_file.peek()


class LLEFile(object):
    def __init__(self, lle_file, ft2_file, rsp_file):
        """
        Class to read the LLE and FT2 files

        Inspired heavily by G. Vianello



        :param lle_file:
        :param ft2_file:
        """

        with fits.open(rsp_file) as rsp_:

            data = rsp_['EBOUNDS'].data

            self._emin = data.E_MIN
            self._emax = data.E_MAX
            self._channels = data.CHANNEL

        with fits.open(lle_file) as ft1_:

            data = ft1_['EVENTS'].data

            self._events = data.TIME  # - trigger_time
            self._energy = data.ENERGY * 1E3  # keV

            self._start_events = ft1_['PRIMARY'].header['TSTART']
            self._stop_events = ft1_['PRIMARY'].header['TSTOP']
            self._utc_start = ft1_['PRIMARY'].header['DATE-OBS']
            self._utc_stop = ft1_['PRIMARY'].header['DATE-END']
            self._instrument = ft1_['PRIMARY'].header['INSTRUME']
            self._telescope = ft1_['PRIMARY'].header['TELESCOP'] + "_LLE"

            try:
                self.triggertime = ft1_['EVENTS'].header['TRIGTIME']


            except:

                # For whatever reason
                warnings.warn(
                        "There is no trigger time in the LLE file. Must me set manually or using MET relative times.")

                self.triggertime = 0

        self._bin_energies_into_pha()

        with fits.open(ft2_file) as ft2_:

            ft2_tstart = ft2_['SC_DATA'].data.field("START")  # - trigger_time
            ft2_tstop = ft2_['SC_DATA'].data.field("STOP")  # - trigger_time
            ft2_livetime = ft2_['SC_DATA'].data.field("LIVETIME")

        ft2_bin_size = 1.0  # seconds

        if not np.all(ft2_livetime <= 1.0):

            warnings.warn("You are using a 30s FT2 file. You should use a 1s Ft2 file otherwise the livetime "
                          "correction will not be accurate!")

            ft2_bin_size = 30.0  # s

        # Now we just need the live time fraction for the righ interval

        livetime = ft2_livetime  # / (ft2_tstop - ft2_tstart)

        # Keep only the needed entries (plus a padding)
        idx = (ft2_tstart >= self._start_events - 10 * ft2_bin_size) & (
            ft2_tstop <= self._stop_events + 10 * ft2_bin_size)

        self._tstart = ft2_tstart[idx]
        self._tstop = ft2_tstop[idx]
        self._livetime = livetime[idx]

        # Now sort all vectors
        idx = np.argsort(self._tstart)

        self._tstart = self._tstart[idx]
        self._tstop = self._tstop[idx]
        self._livetime = self._livetime[idx]

    def _bin_energies_into_pha(self):
        """

        bins the LLE data into PHA channels

        :return:
        """

        self._pha = np.zeros_like(self._energy, dtype=int)

        for emin, emax, channel in zip(self._emin, self._emax, self._channels):

            idx = np.logical_and(emin <= self._energy, self._energy < emax)

            self._pha[idx] = channel

        # There are some events outside of the energy bounds. We will dump those


        self._filter_idx = self._pha > 0

        self._n_channels = len(self._channels)

    @property
    def start_events(self):
        return self._start_events

    @property
    def stop_events(self):
        return self._stop_events

    @property
    def arrival_times(self):
        return self._events[self._filter_idx]

    @property
    def energies(self):
        return self._pha[self._filter_idx]

    @property
    def n_channels(self):
        return self._n_channels

    @property
    def mission(self):
        """
        Return the name of the mission
        :return:
        """
        return self._instrument

    @property
    def instrument(self):
        """
        Return the name of the instrument and detector

        :return:
        """

        return self._telescope

    @property
    def livetime(self):
        return self._livetime

    @property
    def livetime_start(self):
        return self._tstart

    @property
    def livetime_stop(self):
        return self._tstop

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


def lle_light_curve_plot(time_bins, cnts, bkg, width, selection, bkg_selections):
    fig, ax = plt.subplots()

    max_cnts = max(cnts / width)
    top = max_cnts + max_cnts * .2
    min_cnts = min(cnts[cnts > 0] / width[cnts > 0])
    bottom = min_cnts - min_cnts * .05
    mean_time = map(np.mean, time_bins)

    all_masks = []

    # purple: #8da0cb

    step_plot(time_bins, cnts / width, ax,
              color=threeML_config['lle']['lightcurve color'], label="Light Curve")

    for tmin, tmax in selection:
        tmp_mask = np.logical_and(time_bins[:, 0] >= tmin, time_bins[:, 1] <= tmax)

        all_masks.append(tmp_mask)

    if len(all_masks) > 1:

        for mask in all_masks[1:]:
            step_plot(time_bins[mask], cnts[mask] / width[mask], ax,
                      color=threeML_config['lle']['selection color'],
                      fill=True,
                      fill_min=min_cnts)

    step_plot(time_bins[all_masks[0]], cnts[all_masks[0]] / width[all_masks[0]], ax,
              color=threeML_config['lle']['selection color'],
              fill=True,
              fill_min=min_cnts, label="Selection")

    all_masks = []
    for tmin, tmax in bkg_selections:
        tmp_mask = np.logical_and(time_bins[:, 0] >= tmin, time_bins[:, 1] <= tmax)

        all_masks.append(tmp_mask)

    if len(all_masks) > 1:

        for mask in all_masks[1:]:

            step_plot(time_bins[mask], cnts[mask] / width[mask], ax,
                      color=threeML_config['lle']['background selection color'],
                      fill=True,
                      fillAlpha=.4,
                      fill_min=min_cnts)

    step_plot(time_bins[all_masks[0]], cnts[all_masks[0]] / width[all_masks[0]], ax,
              color=threeML_config['lle']['background selection color'],
              fill=True,
              fill_min=min_cnts, fillAlpha=.4, label="Bkg. Selections")

    ax.plot(mean_time, bkg, threeML_config['lle']['background color'], lw=2., label="Background")

    # ax.fill_between(selection, bottom, top, color="#fc8d62", alpha=.4)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Rate (cnts/s)")
    ax.set_ylim(bottom, top)
    ax.set_xlim(time_bins.min(), time_bins.max())
    ax.legend()
