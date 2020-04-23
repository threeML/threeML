import re
import collections
import requests

from threeML.io.network import internet_connection_is_active


def compute_fermi_relative_mission_times(trigger_time):
    """

    If the user has the requests library, this function looks
    online to the HEASARC xtime utility and computes other mission
    times relative to the input MET



    :param trigger_time: a fermi MET
    :return: mission time in a python dictionary
    """
    mission_dict = collections.OrderedDict()

    if trigger_time == 0:
        return None

    # Complements to Volodymyr Savchenko

    xtime_url = "https://heasarc.gsfc.nasa.gov/cgi-bin/Tools/xTime/xTime.pl"

    pattern = """<tr>.*?<th scope=row><label for="(.*?)">(.*?)</label></th>.*?<td align=center>.*?</td>.*?<td>(.*?)</td>.*?</tr>"""

    args = dict(
        time_in_sf=trigger_time,
        timesys_in="u",
        timesys_out="u",
        apply_clock_offset="yes",
    )

    if internet_connection_is_active():

        content = requests.get(xtime_url, params=args).content

        mission_info = re.findall(pattern, content, re.S)

        mission_dict["UTC"] = mission_info[0][-1]
        mission_dict[mission_info[7][1]] = mission_info[7][2]  # LIGO
        mission_dict[mission_info[8][1]] = mission_info[8][2]  # NUSTAR
        mission_dict[mission_info[12][1]] = mission_info[12][2]  # RXTE
        mission_dict[mission_info[16][1]] = mission_info[16][2]  # SUZAKU
        mission_dict[mission_info[20][1]] = mission_info[20][2]  # SWIFT
        mission_dict[mission_info[24][1]] = mission_info[24][2]  # CHANDRA

        return mission_dict

    else:

        return None
