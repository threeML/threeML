import collections
import re

import requests

from threeML.io.network import internet_connection_is_active


def compute_fermi_relative_mission_times(trigger_time):
    """If the user has the requests library, this function looks online to the
    HEASARC xtime utility and computes other mission times relative to the
    input MET.

    :param trigger_time: a fermi MET
    :return: mission time in a python dictionary
    """
    if not isinstance(trigger_time, str):
        try:
            trigger_time = str(trigger_time)
        except Exception:
            raise TypeError(
                "trigger_time must be convertible to string."
                + f" You passed a {type(trigger_time)}"
            )

    mission_dict = collections.OrderedDict()

    if trigger_time == "0":
        return None

    # Complements to Volodymyr Savchenko

    xtime_url = "https://heasarc.gsfc.nasa.gov/cgi-bin/Tools/xTime/xTime.pl"

    pattern = re.compile(
        r"""
        <tr>\s*
          <th[^>]*\bscope=["']?row["']?[^>]*>\s*
            <label[^>]*\bfor=["']([^"']+)["'][^>]*>
              (.*?)                # label text
            </label>
            .*?
          </th>\s*
          <td[^>]*\balign=["']?center["']?[^>]*>.*?</td>\s*
          <td[^>]*>(.*?)</td>\s*
          (?:
              <!--.*?-->           # allow full HTML comments
            | (?!</tr>).           # or any char not starting </tr>
          )*?
        </tr>
        """,
        re.S | re.X,
    )
    args = dict(
        time_in_sf=trigger_time,
        timesys_in="u",
        timesys_out="u",
        apply_clock_offset="yes",
    )

    if internet_connection_is_active():
        content = requests.get(xtime_url, params=args).content.decode("utf8")
        mission_info = re.findall(pattern, content)

        mission_dict["UTC"] = mission_info[0][-1]
        for i in range(1, len(mission_info), 1):
            key = re.sub(r"<[^>]+>", "", mission_info[i][1])
            val = mission_info[i][2]
            mission_dict[key] = val
        return mission_dict

    else:  # pragma: no cover
        return None
