import collections
import warnings

import astropy.io.fits as fits
import numpy as np
import pandas as pd

from threeML.utils.fermi_relative_mission_time import (
    compute_fermi_relative_mission_times,
)
from threeML.io.logging import setup_logger


log = setup_logger(__name__)

class LLEFile(object):
    def __init__(self, lle_file, ft2_file, rsp_file):
        """
        Class to read the LLE and FT2 files

        Inspired heavily by G. Vianello



        :param lle_file:
        :param ft2_file:
        """

        with fits.open(rsp_file) as rsp_:

            data = rsp_["EBOUNDS"].data

            self._emin = data.E_MIN
            self._emax = data.E_MAX
            self._channels = data.CHANNEL

        with fits.open(lle_file) as ft1_:

            data = ft1_["EVENTS"].data

            self._events = data.TIME  # - trigger_time
            self._energy = data.ENERGY * 1e3  # keV

            self._tstart = ft1_["PRIMARY"].header["TSTART"]
            self._tstop = ft1_["PRIMARY"].header["TSTOP"]
            self._utc_start = ft1_["PRIMARY"].header["DATE-OBS"]
            self._utc_stop = ft1_["PRIMARY"].header["DATE-END"]
            self._instrument = ft1_["PRIMARY"].header["INSTRUME"]
            self._telescope = ft1_["PRIMARY"].header["TELESCOP"] + "_LLE"
            self._gti_start = ft1_["GTI"].data["START"]
            self._gti_stop = ft1_["GTI"].data["STOP"]

            try:
                self._trigger_time = ft1_["EVENTS"].header["TRIGTIME"]

            except:

                # For whatever reason
                log.warning(
                    "There is no trigger time in the LLE file. Must be set manually or using MET relative times."
                )

                self._trigger_time = 0

        # bin the energies into PHA channels
        # and filter out over/underflow
        self._bin_energies_into_pha()

        # filter events outside of GTIs

        self._apply_gti_to_events()

        with fits.open(ft2_file) as ft2_:

            ft2_tstart = ft2_["SC_DATA"].data.field("START")  # - trigger_time
            ft2_tstop = ft2_["SC_DATA"].data.field("STOP")  # - trigger_time
            ft2_livetime = ft2_["SC_DATA"].data.field("LIVETIME")

        ft2_bin_size = 1.0  # seconds

        if not np.all(ft2_livetime <= 1.0):

            log.warning(
                "You are using a 30s FT2 file. You should use a 1s Ft2 file otherwise the livetime "
                "correction will not be accurate!"
            )

            ft2_bin_size = 30.0  # s

        # Keep only the needed entries (plus a padding)
        idx = (ft2_tstart >= self._tstart - 10 * ft2_bin_size) & (
            ft2_tstop <= self._tstop + 10 * ft2_bin_size
        )

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

        assert self._tstart <= val <= self._tstop, (
            "Trigger time must be within the interval (%f,%f)"
            % (self._tstart, self._tstop)
        )

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

        return np.vstack((self._emin, self._emax))

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

        fermi_dict["Fermi Trigger Time"] = "%.3f" % self._trigger_time
        fermi_dict["Fermi MET OBS Start"] = "%.3f" % self._tstart
        fermi_dict["Fermi MET OBS Stop"] = "%.3f" % self._tstop
        fermi_dict["Fermi UTC OBS Start"] = self._utc_start
        fermi_dict["Fermi UTC OBS Stop"] = self._utc_stop

        fermi_df = pd.Series(fermi_dict, index=fermi_dict.keys())

        if mission_dict is not None:
            mission_df = pd.Series(mission_dict, index=mission_dict.keys())

            fermi_df = fermi_df.append(mission_df)

        return fermi_df
