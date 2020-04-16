from future import standard_library

standard_library.install_aliases()
from builtins import map
from builtins import str
from builtins import range
import numpy as np
import pandas as pd
import re
import urllib.request, urllib.error, urllib.parse

import astropy.table as astro_table

from threeML.catalogs.VirtualObservatoryCatalog import VirtualObservatoryCatalog
from threeML.exceptions.custom_exceptions import custom_warnings
from threeML.config.config import threeML_config
from threeML.io.get_heasarc_table_as_pandas import get_heasarc_table_as_pandas
from threeML.io.rich_display import display

import astropy.time as astro_time


_gcn_match = re.compile("^\d{4}GCN\D?\.*(\d*)\.*\d\D$")
_trigger_name_match = re.compile("^GRB \d{6}[A-Z]$")


class SwiftGRBCatalog(VirtualObservatoryCatalog):
    def __init__(self, update=False):
        """
        The Swift GRB catalog. Search for GRBs  by trigger
        number, location,  T90, and date range.

        :param update: force update the XML VO table
        """

        self._update = update

        super(SwiftGRBCatalog, self).__init__(
            "swiftgrb",
            threeML_config["catalogs"]["Swift"]["Swift GRB catalog"],
            "Swift GRB catalog",
        )

        # collect all the instruments also seeing the GRBs
        self._build_other_obs_instruments()

    def apply_format(self, table):
        new_table = table[
            "name",
            "ra",
            "dec",
            "trigger_time",
            "redshift",
            "bat_t90",
            "bat_detection",
            "xrt_detection",
            "xrt_flare",
            "uvot_detection",
            "radio_detection",
            "opt_detection",
        ]

        new_table["ra"].format = "5.3f"
        new_table["dec"].format = "5.3f"

        return new_table.group_by("trigger_time")

    def _get_vo_table_from_source(self):

        self._vo_dataframe = get_heasarc_table_as_pandas(
            "swiftgrb", update=self._update, cache_time_days=1.0
        )

    def _source_is_valid(self, source):

        warn_string = (
            "The trigger %s is not valid. Must be in the form GRB080916009" % source
        )

        match = _trigger_name_match.match(source)

        if match is None:

            custom_warnings.warn(warn_string)

            answer = False

        else:

            answer = True

        return answer

    def _build_other_obs_instruments(self):
        """
        builds a list of all the other instruments that observed Swift GRBs

        :return:
        """

        obs_inst_ = list(
            map(
                np.unique,
                [
                    np.asarray(self._vo_dataframe.other_obs),
                    np.asarray(self._vo_dataframe.other_obs2),
                    np.asarray(self._vo_dataframe.other_obs3),
                    np.asarray(self._vo_dataframe.other_obs4),
                ],
            )
        )

        self._other_observings_instruments = [
            x for x in np.unique(np.concatenate(obs_inst_)) if x != ""
        ]

    @property
    def other_observing_instruments(self):

        return self._other_observings_instruments

    def query_other_observing_instruments(self, *instruments):
        """
        search for observations that were also seen by the requested instrument.
        to see what instruments are available, use the .other_observing_instruments call


        :param instruments: other instruments
        :return:
        """

        all_queries = []

        for instrument in instruments:

            assert instrument in self._other_observings_instruments, (
                "Other instrument choices include %s"
                % (" ,".join(self._other_observings_instruments))
            )

            query_string = (
                ' other_obs == "%s" | other_obs2 == "%s" |other_obs3 == "%s" |other_obs4 == "%s"'
                % tuple([instrument] * 4)
            )

            result = self._vo_dataframe.query(query_string)

            all_queries.append(result)

            query_results = pd.concat(all_queries)

            table = astro_table.Table.from_pandas(query_results)

            name_column = astro_table.Column(name="name", data=query_results.index)
            table.add_column(name_column, index=0)

            out = self.apply_format(table)

            self._last_query_results = query_results

        return out

    @staticmethod
    def _get_fermiGBM_trigger_number_from_gcn(gcn_url):
        """
        this is a custom function that parses GBM GCNs to find the burst number
        that can later be used to download GBM data. It contains a lot of regex statements
        to handle the variability in the GCNs


        :param gcn_url: url to gbm gcn
        :return:
        """

        data = urllib.request.urlopen(gcn_url)

        data_decode = []

        for x in data.readlines():

            try:

                tmp = str(x, "utf-8")

                data_decode.append(tmp)

            except (UnicodeDecodeError):

                pass

        string = "".join(data_decode).replace("\n", "")
        try:

            trigger_number = (
                re.search("trigger *\d* */ *(\d{9}|\d{6}\.\d{3})", string)
                .group(1)
                .replace(".", "")
            )

        except (AttributeError):

            try:

                trigger_number = (
                    re.search("GBM *(\d{9}|\d{6}\.\d{3}), *trigger *\d*", string)
                    .group(1)
                    .replace(".", "")
                )

            except (AttributeError):

                try:

                    trigger_number = (
                        re.search(
                            "trigger *\d* *, *trigcat *(\d{9}|\d{6}\.\d{3})", string
                        )
                        .group(1)
                        .replace(".", "")
                    )

                except (AttributeError):

                    try:

                        trigger_number = (
                            re.search(
                                "trigger *.* */ *\D{0,3}(\d{9}|\d{6}\.\d{3})", string
                            )
                            .group(1)
                            .replace(".", "")
                        )

                    except (AttributeError):

                        try:

                            trigger_number = (
                                re.search(
                                    "Trigger number*.* */ *GRB *(\d{9}|\d{6}\.\d{3})",
                                    string,
                                )
                                .group(1)
                                .replace(".", "")
                            )

                        except (AttributeError):

                            trigger_number = None

        return trigger_number

    def get_other_observation_information(self):
        """
        returns a structured pandas table containing the other observing instruments, their GCNs and if obtainable,
        their trigger numbers/ data identifiers. Currently, the trigger number is only obtained for Fermi-LAT-GBM.

        :return:
        """

        assert (
            self._last_query_results is not None
        ), "You have to run a query before getting observing information"

        # Loop over the table and build a source for each entry
        sources = {}

        for name, row in self._last_query_results.T.items():

            # First we want to get the the detectors used in the SCAT file

            obs_instrument = {}

            for obs in ["xrt", "uvot", "bat", "opt", "radio"]:

                obs_detection = "%s_detection" % obs

                if obs in ["xrt", "uvot", "bat"]:

                    obs_ref = "%s_pos_ref" % obs

                else:

                    obs_ref = "%s_ref" % obs

                detect = row[obs_detection]

                if detect == "Y":  # or detect== 'U':

                    observed = True

                else:

                    observed = False

                if observed:

                    reference = self._parse_redshift_reference(row[obs_ref])

                    # gcn = "https://gcn.gsfc.nasa.gov/gcn3/%s.gcn3" % gcn_number

                    info = {"reference": reference, "observed": detect}

                else:

                    info = {"GCN": None, "observed": detect}

                obs_instrument[obs] = info

            sources[name] = obs_instrument

        sources = pd.concat(
            list(map(pd.DataFrame, list(sources.values()))), keys=list(sources.keys())
        )

        return sources

    def get_other_instrument_information(self):
        """
        Return the detectors used for spectral analysis as well as their background
        intervals. Peak flux and fluence intervals are also returned as well as best fit models

        :return: observing information dataframe indexed by source
        """

        assert (
            self._last_query_results is not None
        ), "You have to run a query before getting observing information"

        sources = {}

        for name, row in self._last_query_results.T.items():

            obs_instrument = {}

            # loop over the observation indices
            for obs in range(1, 5):

                if obs == 1:

                    obs_base = "other_obs"

                else:

                    obs_base = "other_obs%d" % obs

                obs_ref = "%s_ref" % obs_base

                obs = row[obs_base]

                # this means that nothing in this column saw the grb
                if obs == "":

                    observed = False

                else:

                    observed = True

                if observed:

                    # if we saw it then lets get the GCN
                    gcn_number = _gcn_match.search(row[obs_ref]).group(1)
                    # gcn_number = filter(lambda x: x != '', row[obs_ref].split('.'))[1]

                    # make the URL
                    gcn = "https://gcn.gsfc.nasa.gov/gcn3/%s.gcn3" % gcn_number

                    # just for Fermi GBM, lets get the trigger number

                    # TODO: add more instruments
                    if obs == "Fermi-GBM":

                        info = {
                            "GCN": gcn,
                            "trigger number": self._get_fermiGBM_trigger_number_from_gcn(
                                str(gcn)
                            ),
                        }

                    else:

                        info = {"GCN": gcn, "trigger number": None}

                    obs_instrument[obs] = info

            sources[name] = obs_instrument

        # build the data frame
        sources = pd.concat(
            list(map(pd.DataFrame, list(sources.values()))), keys=list(sources.keys())
        )

        display(sources)

        return sources

    def get_redshift(self):
        """
        Get the redshift and redshift type from the searched sources


        :return:
        """

        assert (
            self._last_query_results is not None
        ), "You have to run a query before getting observing information"

        redshift_df = (
            self._last_query_results.loc[
                :, ["redshift", "redshift_err", "redshift_type", "redshift_ref"]
            ]
        ).copy(deep=True)

        redshift_df = redshift_df.rename(
            columns={
                "redshift": "z",
                "redshift_err": "z err",
                "redshift_type": "z type",
                "redshift_ref": "reference",
            }
        )

        redshift_df["reference"] = redshift_df["reference"].apply(
            self._parse_redshift_reference
        )

        return redshift_df

    @staticmethod
    def _parse_redshift_reference(reference):

        if reference == "":

            url = None

        elif "GCN" in reference:
            gcn_number = _gcn_match.search(reference).group(1)

            url = "https://gcn.gsfc.nasa.gov/gcn3/%s.gcn3" % gcn_number

        else:

            url = "http://adsabs.harvard.edu/abs/%s" % reference

        return url
