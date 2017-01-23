__author__ = 'grburgess'

import astropy.io.fits as fits
import numpy as np

import pandas as pd
import warnings


from threeML.plugins.EventListLike import EventListLike
from threeML.plugins.OGIP.eventlist import EventListWithLiveTime
from threeML.io.rich_display import display
from threeML.utils.fermi_relative_mission_time import compute_fermi_relative_mission_times

from threeML.io.plugin_plots import binned_light_curve_plot

__instrument_name = "Fermi LAT LLE"


class BinningMethodError(RuntimeError):
    pass


class FermiLATLLELike(EventListLike):
    def __init__(self, name, lle_file, ft2_file, rsp_file, source_intervals, background_selections=None, restore_background=None,
                 trigger_time=None, poly_order=-1, unbinned=False, verbose=True):
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
            self._lat_lle_file.trigger_time = trigger_time

        # Mark channels less than 50 MeV as bad

        channel_50MeV = np.searchsorted(self._lat_lle_file.energy_edges[0],50000) - 1

        native_quality = np.zeros(self._lat_lle_file.n_channels,dtype=int)

        idx = np.arange(self._lat_lle_file.n_channels) < channel_50MeV

        native_quality[idx] = 5

        event_list = EventListWithLiveTime(
                arrival_times=self._lat_lle_file.arrival_times - self._lat_lle_file.trigger_time,
                energies=self._lat_lle_file.energies,
                n_channels=self._lat_lle_file.n_channels,
                live_time=self._lat_lle_file.livetime,
                live_time_starts=self._lat_lle_file.livetime_start - self._lat_lle_file.trigger_time,
                live_time_stops=self._lat_lle_file.livetime_stop - self._lat_lle_file.trigger_time,
                start_time=self._lat_lle_file.tstart - self._lat_lle_file.trigger_time,
                stop_time=self._lat_lle_file.tstop - self._lat_lle_file.trigger_time,
                quality=native_quality,
                first_channel=1,
                rsp_file=rsp_file,
                instrument=self._lat_lle_file.instrument,
                mission=self._lat_lle_file.mission,
                verbose=verbose)

        super(FermiLATLLELike,self).__init__(name,
                                             event_list,
                                             rsp_file=rsp_file,
                                             source_intervals=source_intervals,
                                             background_selections=background_selections,
                                             poly_order=poly_order,
                                             unbinned=unbinned,
                                             verbose=verbose,
                                             restore_poly_fit=restore_background
                                             )





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
            mask = np.array([False] * self._event_list.n_events)

            for selection in energy_selection:

                ee = map(float, selection.split("-"))

                if len(ee) != 2:
                    raise RuntimeError('Energy selection is not valid! Form: <low>-<high>.')

                emin, emax = sorted(ee)

                idx1 = self._rsp.energy_to_channel(emin)
                idx2 = self._rsp.energy_to_channel(emax)

                # Update the allowed channels
                valid_channels.extend(range(idx1, idx2))

                this_mask = np.logical_and(self._event_list.energies >= idx1, self._event_list.energies <= idx2)

                np.logical_or(mask, this_mask, out=mask)

        else:

            mask = np.array([True] * self._event_list.n_events)
            valid_channels = range(self._lat_lle_file.n_channels)

        if use_binner:

            bin_start, bin_stop = self._event_list.bins
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

        cnts, bins = np.histogram(self._lat_lle_file.arrival_times[mask] - self._lat_lle_file.trigger_time, bins=bins)
        time_bins = np.array([[bins[i], bins[i + 1]] for i in range(len(bins) - 1)])

        width = np.diff(bins)

        bkg = []
        for j, tb in enumerate(time_bins):
            tmpbkg = 0.
            for i in valid_channels:
                poly = self._event_list.polynomials[i]

                tmpbkg += poly.integral(tb[0], tb[1]) / (width[j])

            bkg.append(tmpbkg)

        binned_light_curve_plot(time_bins,
                                cnts,
                                bkg,
                                width,
                                selection=zip(self._event_list.time_intervals.start_times, self._event_list.time_intervals.stop_times),
                                bkg_selections=zip(self._event_list.poly_intervals.start_times, self._event_list.poly_intervals.stop_times),
                                instrument='lle')

    def peek(self):

        print("LLE File Info:")

        self._event_list.peek()

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

            self._tstart = ft1_['PRIMARY'].header['TSTART']
            self._tstop = ft1_['PRIMARY'].header['TSTOP']
            self._utc_start = ft1_['PRIMARY'].header['DATE-OBS']
            self._utc_stop = ft1_['PRIMARY'].header['DATE-END']
            self._instrument = ft1_['PRIMARY'].header['INSTRUME']
            self._telescope = ft1_['PRIMARY'].header['TELESCOP'] + "_LLE"
            self._gti_start = ft1_['GTI'].data['START']
            self._gti_stop = ft1_['GTI'].data['STOP']

            try:
                self._trigger_time = ft1_['EVENTS'].header['TRIGTIME']


            except:

                # For whatever reason
                warnings.warn(
                        "There is no trigger time in the LLE file. Must be set manually or using MET relative times.")

                self._trigger_time = 0

        # bin the energies into PHA channels
        # and filter out over/underflow
        self._bin_energies_into_pha()

        # filter events outside of GTIs

        self._apply_gti_to_events()

        with fits.open(ft2_file) as ft2_:

            ft2_tstart = ft2_['SC_DATA'].data.field("START")  # - trigger_time
            ft2_tstop = ft2_['SC_DATA'].data.field("STOP")  # - trigger_time
            ft2_livetime = ft2_['SC_DATA'].data.field("LIVETIME")

        ft2_bin_size = 1.0  # seconds

        if not np.all(ft2_livetime <= 1.0):

            warnings.warn("You are using a 30s FT2 file. You should use a 1s Ft2 file otherwise the livetime "
                          "correction will not be accurate!")

            ft2_bin_size = 30.0  # s

        # Keep only the needed entries (plus a padding)
        idx = (ft2_tstart >= self._tstart - 10 * ft2_bin_size) & (
            ft2_tstop <= self._tstop + 10 * ft2_bin_size)

        self._ft2_tstart = ft2_tstart[idx]
        self._ft2_tstop = ft2_tstop[idx]
        self._livetime = ft2_livetime[idx]

        # now filter the livetime by GTI

        self._apply_gti_to_live_time()





        # Now sort all vectors
        idx = np.argsort(self._ft2_tstart)

        self._ft2_tstart = self._ft2_tstart[idx]
        self._ft2_tstop = self._ft2_tstop[idx]
        self._livetime = self._livetime[idx]

    def _apply_gti_to_live_time(self):
        """
        This function applies the GTIs to the live time intervals

        It will remove any livetime interval not falling within the
        boundaries of a GTI. The FT2 bins are assumed to have the same
        boundaries as the GTI.

        Events falling outside the GTI boundaries are already removed.

        :return: none
        """

        # First negate all FT2 entries

        filter_idx = np.zeros_like(self._livetime, dtype=bool)

        # now loop through each GTI interval

        for start, stop in zip(self._gti_start, self._gti_stop):

            # create an index of all the FT2 bins falling within this interval

            tmp_idx = np.logical_and(start <= self._ft2_tstart, self._ft2_tstop <= stop)

            # add them to the already selected idx
            filter_idx = np.logical_or(filter_idx, tmp_idx)

        # Now filter the whole list
        self._ft2_tstart = self._ft2_tstart[filter_idx]
        self._ft2_tstop = self._ft2_tstop[filter_idx]
        self._livetime = self._livetime[filter_idx]

    def _apply_gti_to_events(self):
        """

        This created a filter index for events falling outside of the
        GTI. It must be run after the events are binned in energy because
        a filter is set up in that function for events that have energies
        outside the EBOUNDS of the DRM

        :return: none
        """

        # initial filter
        filter_idx = np.zeros_like(self._events, dtype=bool)

        # loop throught the GTI intervals
        for start, stop in zip(self._gti_start, self._gti_stop):

            # capture all the events within that interval
            tmp_idx = np.logical_and(start <= self._events, self._events <= stop)

            # combine with the already selected events
            filter_idx = np.logical_or(filter_idx, tmp_idx)

        # filter from the energy selection
        self._filter_idx = np.logical_and(self._filter_idx, filter_idx)

    def is_in_gti(self, time):
        """

        Checks if a time falls within
        a GTI

        :param time: time in MET
        :return: bool
        """

        in_gti = False

        for start, stop in zip(self._gti_start, self._gti_stop):

            if (start <= time) and (time <= stop):

                in_gti = True

        return in_gti




    def _bin_energies_into_pha(self):
        """

        bins the LLE data into PHA channels

        :return:
        """

        edges = np.append(self._emin, self._emax[-1])

        self._pha = np.digitize(self._energy, edges)


        # There are some events outside of the energy bounds. We will dump those


        self._filter_idx = self._pha > 0

        self._n_channels = len(self._channels)

    @property
    def trigger_time(self):
        """
        Gets the trigger time in MET
        :return: trigger time in MET
        """
        return self._trigger_time

    @trigger_time.setter
    def trigger_time(self, val):

        assert self._tstart <= val <= self._tstop, "Trigger time must be within the interval (%f,%f)" % (
            self._tstart, self._tstop)

        self._trigger_time = val

    @property
    def tstart(self):
        return self._tstart

    @property
    def tstop(self):
        return self._tstop

    @property
    def arrival_times(self):
        """
        The GTI/energy filtered arrival times in MET
        :return:
        """
        return self._events[self._filter_idx]

    @property
    def energies(self):
        """
        The GTI/energy filtered pha energies
        :return:
        """
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
    def energy_edges(self):

        return np.vstack((self._emin,self._emax))

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
        return self._ft2_tstart

    @property
    def livetime_stop(self):
        return self._ft2_tstop



    def peek(self):
        """
        Examine the currently selected interval
        If connected to the internet, will also look up info for other instruments to compare with
        Fermi.

        :return: none
        """

        mission_dict = compute_fermi_relative_mission_times(self._trigger_time)

        fermi_dict = {}

        fermi_dict['Fermi Trigger Time'] = self.trigger_time
        fermi_dict['Fermi MET OBS Start'] = self._tstart
        fermi_dict['Fermi MET OBS Stop'] = self._tstop
        fermi_dict['Fermi UTC OBS Start'] = self._utc_start
        fermi_dict['Fermi UTC OBS Stop'] = self._utc_stop

        if mission_dict is not None:
            mission_df = pd.Series(mission_dict)

            display(mission_df)

        fermi_df = pd.Series(fermi_dict)

        display(fermi_df)



