import collections
import re
import warnings

import astropy.io.fits as fits
import numpy as np
import pandas as pd
import requests

from threeML.io.logging import setup_logger
from threeML.utils.fermi_relative_mission_time import \
    compute_fermi_relative_mission_times
from threeML.utils.spectrum.pha_spectrum import PHASpectrumSet

log = setup_logger(__name__)


class GBMTTEFile(object):
    def __init__(self, ttefile: str) -> None:
        """

        A simple class for opening and easily accessing Fermi GBM
        TTE Files.

        :param ttefile: The filename of the TTE file to be stored

        """

        tte = fits.open(ttefile)

        self._events = tte["EVENTS"].data["TIME"]
        self._pha = tte["EVENTS"].data["PHA"]

        # the GBM TTE data are not always sorted in TIME.
        # we will now do this for you. We should at some
        # point check with NASA if this is on purpose.

        # but first we must check that there are NO duplicated events
        # and then warn the user

        if not len(self._events) == len(np.unique(self._events)):

            log.warning(
                "The TTE file %s contains duplicate time tags and is thus invalid. Contact the FSSC "
                % ttefile
            )

        # sorting in time
        sort_idx = self._events.argsort()

        if not np.alltrue(self._events[sort_idx] == self._events):

            # now sort both time and energy
            log.warning(
                "The TTE file %s was not sorted in time but contains no duplicate events. We will sort the times, but use caution with this file. Contact the FSSC."
            )
            self._events = self._events[sort_idx]
            self._pha = self._pha[sort_idx]

        try:
            self._trigger_time = tte["PRIMARY"].header["TRIGTIME"]

        except:

            # For continuous data
            log.warning(
                "There is no trigger time in the TTE file. Must be set manually or using MET relative times."
            )

            log.debug("set trigger time to zero")
            self._trigger_time = 0

        self._start_events = tte["PRIMARY"].header["TSTART"]
        self._stop_events = tte["PRIMARY"].header["TSTOP"]

        self._utc_start = tte["PRIMARY"].header["DATE-OBS"]
        self._utc_stop = tte["PRIMARY"].header["DATE-END"]

        self._n_channels = tte["EBOUNDS"].header["NAXIS2"]

        self._det_name = "%s_%s" % (
            tte["PRIMARY"].header["INSTRUME"],
            tte["PRIMARY"].header["DETNAM"],
        )

        self._telescope = tte["PRIMARY"].header["TELESCOP"]

        self._calculate_deadtime()

    @property
    def trigger_time(self) -> float:

        return self._trigger_time

    @trigger_time.setter
    def trigger_time(self, val) -> None:

        assert self._start_events <= val <= self._stop_events, (
            "Trigger time must be within the interval (%f,%f)"
            % (self._start_events, self._stop_events)
        )

        self._trigger_time = val

    @property
    def tstart(self) -> float:
        return self._start_events

    @property
    def tstop(self) -> float:
        return self._stop_events

    @property
    def arrival_times(self) -> np.ndarray:
        return self._events

    @property
    def n_channels(self) -> int:
        return self._n_channels

    @property
    def energies(self) -> np.ndarray:
        return self._pha

    @property
    def mission(self) -> str:
        """
        Return the name of the mission
        :return:
        """
        return self._telescope

    @property
    def det_name(self) -> str:
        """
        Return the name of the instrument and detector

        :return:
        """

        return self._det_name

    @property
    def deadtime(self) -> np.ndarray:
        return self._deadtime

    def _calculate_deadtime(self) -> None:
        """
        Computes an array of deadtimes following the perscription of Meegan et al. (2009).

        The array can be summed over to obtain the total dead time

        """
        self._deadtime = np.zeros_like(self._events)
        overflow_mask = self._pha == 127  # specific to gbm! should work for CTTE

        # From Meegan et al. (2009)
        # Dead time for overflow (note, overflow sometimes changes)
        self._deadtime[overflow_mask] = 10.0e-6  # s

        # Normal dead time
        self._deadtime[~overflow_mask] = 2.0e-6  # s

    def _compute_mission_times(self) -> None:

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
            apply_clock_offset="yes",
        )

        try:

            content = requests.get(xtime_url, params=args).content

            mission_info = re.findall(pattern, content, re.S)

            mission_dict["UTC"] = mission_info[0][-1]
            mission_dict[mission_info[7][1]] = mission_info[7][2]  # LIGO
            mission_dict[mission_info[8][1]] = mission_info[8][2]  # NUSTAR
            mission_dict[mission_info[12][1]] = mission_info[12][2]  # RXTE
            mission_dict[mission_info[16][1]] = mission_info[16][2]  # SUZAKU
            mission_dict[mission_info[20][1]] = mission_info[20][2]  # SWIFT
            mission_dict[mission_info[24][1]] = mission_info[24][2]  # CHANDRA

        except:

            log.warning(
                "You do not have the requests library, cannot get time system from Heasarc "
                "at this point."
            )

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

        fermi_dict["Fermi Trigger Time"] = "%.3f" % self._trigger_time
        fermi_dict["Fermi MET OBS Start"] = "%.3f" % self._start_events
        fermi_dict["Fermi MET OBS Stop"] = "%.3f" % self._stop_events
        fermi_dict["Fermi UTC OBS Start"] = self._utc_start
        fermi_dict["Fermi UTC OBS Stop"] = self._utc_stop

        fermi_df = pd.Series(fermi_dict, index=fermi_dict.keys())

        if mission_dict is not None:
            mission_df = pd.Series(mission_dict, index=mission_dict.keys())

            fermi_df = fermi_df.append(mission_df)

        return fermi_df


class GBMCdata(object):
    def __init__(self, cdata_file: str, rsp_file: str) -> None:

        self.spectrum_set = PHASpectrumSet(cdata_file, rsp_file=rsp_file)

        cdata = fits.open(cdata_file)

        try:

            self._trigger_time = cdata["PRIMARY"].header["TRIGTIME"]

        except:

            # For continuous data
            log.warning(
                "There is no trigger time in the TTE file. Must be set manually or using MET relative times."
            )

            self._trigger_time = 0

        self._start_events = cdata["PRIMARY"].header["TSTART"]
        self._stop_events = cdata["PRIMARY"].header["TSTOP"]

        self._utc_start = cdata["PRIMARY"].header["DATE-OBS"]
        self._utc_stop = cdata["PRIMARY"].header["DATE-END"]

        self._n_channels = cdata["EBOUNDS"].header["NAXIS2"]

        self._det_name = "%s_%s" % (
            cdata["PRIMARY"].header["INSTRUME"],
            cdata["PRIMARY"].header["DETNAM"],
        )

        self._telescope = cdata["PRIMARY"].header["TELESCOP"]

    @property
    def trigger_time(self) -> float:

        return self._trigger_time

    @trigger_time.setter
    def trigger_time(self, val) -> None:

        assert self._start_events <= val <= self._stop_events, (
            "Trigger time must be within the interval (%f,%f)"
            % (self._start_events, self._stop_events)
        )

        self._trigger_time = val

    @property
    def tstart(self) -> float:
        return self._start_events

    @property
    def tstop(self) -> float:
        return self._stop_events

    @property
    def n_channels(self) -> int:
        return self._n_channels

    @property
    def energies(self) -> np.ndarray:
        return self._pha

    @property
    def mission(self) -> str:
        """
        Return the name of the mission
        :return:
        """
        return self._telescope

    @property
    def det_name(self) -> str:
        """
        Return the name of the instrument and detector

        :return:
        """

        return self._det_name

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
            apply_clock_offset="yes",
        )

        try:

            content = requests.get(xtime_url, params=args).content

            mission_info = re.findall(pattern, content, re.S)

            mission_dict["UTC"] = mission_info[0][-1]
            mission_dict[mission_info[7][1]] = mission_info[7][2]  # LIGO
            mission_dict[mission_info[8][1]] = mission_info[8][2]  # NUSTAR
            mission_dict[mission_info[12][1]] = mission_info[12][2]  # RXTE
            mission_dict[mission_info[16][1]] = mission_info[16][2]  # SUZAKU
            mission_dict[mission_info[20][1]] = mission_info[20][2]  # SWIFT
            mission_dict[mission_info[24][1]] = mission_info[24][2]  # CHANDRA

        except:

            log.warning(
                "You do not have the requests library, cannot get time system from Heasarc "
                "at this point."
            )

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

        fermi_dict["Fermi Trigger Time"] = "%.3f" % self._trigger_time
        fermi_dict["Fermi MET OBS Start"] = "%.3f" % self._start_events
        fermi_dict["Fermi MET OBS Stop"] = "%.3f" % self._stop_events
        fermi_dict["Fermi UTC OBS Start"] = self._utc_start
        fermi_dict["Fermi UTC OBS Stop"] = self._utc_stop

        fermi_df = pd.Series(fermi_dict, index=fermi_dict.keys())

        if mission_dict is not None:
            mission_df = pd.Series(mission_dict, index=mission_dict.keys())

            fermi_df = fermi_df.append(mission_df)

        return fermi_df
