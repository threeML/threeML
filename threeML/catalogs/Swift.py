import numpy as np
import pandas as pd
import re
import urllib2

from threeML.catalogs.VirtualObservatoryCatalog import VirtualObservatoryCatalog
from threeML.exceptions.custom_exceptions import InvalidUTC
from threeML.config.config import threeML_config
from threeML.io.get_heasarc_table_as_pandas import get_heasarc_table_as_pandas
from threeML.io.rich_display import display

import astropy.time as astro_time


_gcn_match = re.compile("^\d{4}GCN\D?\.*(\d*)\.*\d\D$")
_trigger_name_match = re.compile("^(GRB|grb)? ?(\d{6}[a-zA-Z]?)$")



class SwiftGRBCatalog(VirtualObservatoryCatalog):
    def __init__(self, update=False):
        """
        The Swift GRB catalog. Search for GRBs  by trigger
        number, location,  T90, and date range.

        :param update: force update the XML VO table
        """

        super(SwiftGRBCatalog, self).__init__('swiftgrb',
                                              threeML_config['catalogs']['Swift']['Swift GRB catalog'],
                                              'Swift GRB catalog')

        self._vo_dataframe, self._all_table = get_heasarc_table_as_pandas('swiftgrb',
                                                                          update=update,
                                                                          cache_time_days=1.)
        # collect all the instruments also seeing the GRBs
        self._build_other_obs_instruments()

    def apply_format(self, table):
        new_table = table['name',
                          'ra', 'dec',
                          'trigger_time',
                          'redshift',
                          'bat_t90',
                          'bat_detection',
                          'xrt_detection',
                          'xrt_flare',
                          'uvot_detection',
                          'radio_detection',
                          'opt_detection'

        ]

        new_table['ra'].format = '5.3f'
        new_table['dec'].format = '5.3f'

        return new_table.group_by('trigger_time')

    def _build_other_obs_instruments(self):
        """
        builds a list of all the other instruments that observed Swift GRBs

        :return:
        """

        obs_inst_ = map(np.unique, [np.asarray(self._vo_dataframe.other_obs),
                                    np.asarray(self._vo_dataframe.other_obs2),
                                    np.asarray(self._vo_dataframe.other_obs3),
                                    np.asarray(self._vo_dataframe.other_obs4)])

        self._other_observings_instruments = filter(lambda x: x != '', np.unique(np.concatenate(obs_inst_)))

    @property
    def other_observing_instruments(self):

        return self._other_observings_instruments

    def search_trigger_name(self, *trigger_names):
        """
        Find the information on the given trigger names.

        :param trigger_names: trigger numbers (str) e.g. 'GRB 080810', 'GRB 080810A', 'grb 080810a'
        :return:
        """

        # check the trigger names

        _valid_trigger_args = ['GRB 080810', 'GRB 080810A', 'grb 080810a']

        valid_names = []

        # use regex to enforce trigger name style
        for trigger in trigger_names:
            assert_string = "The trigger %s is not valid. Must be in the form %s" % (trigger,
                                                                                     ', or '.join(
                                                                                         _valid_trigger_args))
            assert type(trigger) == str, "triggers must be strings"

            trigger = trigger.upper()

            search = _trigger_name_match.match(trigger)

            assert search is not None, assert_string

            assert search.group(2) is not None, assert_string

            valid_names.append("GRB %s" % search.group(2))

        n_entries = self._vo_dataframe.shape[0]

        idx = np.zeros(n_entries, dtype=bool)

        # search through the index for the requested triggers

        for name in valid_names:
            name = name.upper()

            condition = self._vo_dataframe.index == name

            idx = np.logical_or(idx, condition)

        self._last_query_results = self._vo_dataframe[idx]

        table = self.apply_format(self._all_table[np.asarray(idx)])

        return table

    def search_t90(self, t90_greater=None, t90_less=None):
        """
        search for GBM GRBs by their T90 values.

        Example:
            T90s >= 2 -> search_t90(t90_greater=2)
            T90s <= 10 -> search_t90(t90_less=10)
            2 <= T90s <= 10 search_t90(t90_greater, t90_less=10)

        :param t90_greater: value for T90s greater
        :param t90_less: value for T90s less
        :return:
        """

        assert t90_greater is not None or t90_less is not None, 'You must specify either the greater or less argument'

        n_entries = self._vo_dataframe.shape[0]

        # Create a dummy index first

        idx = np.ones(n_entries, dtype=bool)

        # find the entries greater
        if t90_greater is not None:
            idx_tmp = self._vo_dataframe.bat_t90 >= t90_greater

            idx = np.logical_and(idx, idx_tmp)

        # find the entries less
        if t90_less is not None:
            idx_tmp = self._vo_dataframe.bat_t90 <= t90_less

            idx = np.logical_and(idx, idx_tmp)

        table = self.apply_format(self._all_table[np.asarray(idx)])

        return table

    def search_mjd(self, mjd_start, mjd_stop):
        """
        search for triggers in a range of MJD

        :param mjd_start: start of MJD range
        :param mjd_stop:  stop of MJD range
        :return: table of results
        """

        assert mjd_start < mjd_stop, "start must come before stop"

        # the table lists things in UTC by default,
        # so we convert back to MJD which is easily searchable

        time = astro_time.Time(np.array(self._vo_dataframe.trigger_time).tolist(), format='isot', scale='utc')

        table_mjd = time.mjd

        idx = np.logical_and(mjd_start <= table_mjd,
                             table_mjd <= mjd_stop)

        self._last_query_results = self._vo_dataframe[idx]

        table = self.apply_format(self._all_table[np.asarray(idx)])

        return table

    def search_utc(self, utc_start, utc_stop):
        """
        Search for triggers in a range of UTC values. UTC time must be specified
         in the as follows '1999-01-01T00:00:00.123456789' '2010-01-01T00:00:00'
        :param utc_start: start of UTC interval
        :param utc_stop: stop of UTC interval
        :return:
        """

        # use astropy to read the UTC format
        try:
            utc_start, utc_stop = astro_time.Time([utc_start, utc_stop], format='isot', scale='utc')

        except(ValueError):

            raise InvalidUTC(
                "one of %s, %s is not a valid UTC string. Exmaple: '1999-01-01T00:00:00.123456789' or '2010-01-01T00:00:00'" % (
                    utc_start, utc_stop))

        # convert the UTC format to MJD and use the MJD search
        return self.search_mjd(utc_start.mjd, utc_stop.mjd)

    def search_other_observing_instruments(self, instrument):
        """
        search for observations that were also seen by the requested instrument.
        to see what instruments are available, use the .other_observing_instruments call


        :param instrument: another instrument
        :return:
        """

        assert instrument in self._other_observings_instruments, "Other instrument choices include %s" % (
            ' ,'.join(self._other_observings_instruments))

        idx = np.zeros(self._vo_dataframe.shape[0])

        # the swift table has four columns of instruments
        # we scroll through them
        for obs in range(1, 5):

            if obs == 1:
                obs_base = "other_obs"

            else:

                obs_base = "other_obs%d" % obs

            tmp_idx_ = self._vo_dataframe[obs_base] == instrument

            idx = np.logical_or(tmp_idx_, idx)

        self._last_query_results = self._vo_dataframe[idx]

        table = self.apply_format(self._all_table[np.asarray(idx)])

        return table

    @staticmethod
    def _get_fermiGBM_trigger_number_from_gcn(gcn_url):
        """
        this is a custom function that parses GBM GCNs to find the burst number
        that can later be used to download GBM data. It contains a lot of regex statements
        to handle the variability in the GCNs


        :param gcn_url: url to gbm gcn
        :return:
        """

        data = urllib2.urlopen(gcn_url)

        string = ''.join(data.readlines()).replace('\n', '')
        try:

            trigger_number = re.search("trigger *\d* */ *(\d{9}|\d{6}\.\d{3})", string).group(1).replace('.', '')

        except(AttributeError):

            try:

                trigger_number = re.search("GBM *(\d{9}|\d{6}\.\d{3}), *trigger *\d*", string).group(1).replace('.', '')

            except(AttributeError):

                try:

                    trigger_number = re.search("trigger *\d* *, *trigcat *(\d{9}|\d{6}\.\d{3})", string).group(
                        1).replace('.', '')

                except(AttributeError):

                    try:

                        trigger_number = re.search("trigger *.* */ *\D{0,3}(\d{9}|\d{6}\.\d{3})", string).group(
                            1).replace('.', '')

                    except(AttributeError):

                        try:

                            trigger_number = re.search("Trigger number*.* */ *GRB *(\d{9}|\d{6}\.\d{3})", string).group(
                                1).replace('.', '')

                        except(AttributeError):

                            trigger_number = None

        return trigger_number

    def search_redshift(self, z_greater=None, z_less=None):
        """
        search on redshift range

        :param z_greater: values greater than this will be returned
        :param z_less: values less than this will be returned
        :return: grb table
        """

        assert z_greater is not None or z_less is not None, 'You must specify either the greater or less argument'

        idx = np.isfinite(self._vo_dataframe.redshift)

        # find the entries greater
        if z_greater is not None:
            idx_tmp = self._vo_dataframe.redshift >= z_greater

            idx = np.logical_and(idx, idx_tmp)

        # find the entries less
        if z_less is not None:
            idx_tmp = self._vo_dataframe.redshift <= z_less

            idx = np.logical_and(idx, idx_tmp)

        self._last_query_results = self._vo_dataframe[idx]

        table = self.apply_format(self._all_table[np.asarray(idx)])

        return table



    def get_other_observation_information(self):
        """
        returns a structured pandas table containing the other observing instruments, their GCNs and if obtainable,
        their trigger numbers/ data identifiers. Currently, the trigger number is only obtained for Fermi-GBM.

        :return:
        """

        assert self._last_query_results is not None, "You have to run a query before getting observing information"

        # Loop over the table and build a source for each entry
        sources = {}

        for name, row in self._last_query_results.T.iteritems():

            # First we want to get the the detectors used in the SCAT file

            obs_instrument = {}

            for obs in ['xrt', 'uvot', 'bat', 'opt', 'radio']:

                obs_detection = "%s_detection" % obs

                if obs in ['xrt', 'uvot', 'bat']:

                    obs_ref = "%s_pos_ref" % obs

                else:

                    obs_ref = "%s_ref" % obs

                detect = row[obs_detection]

                if detect == 'Y':  # or detect== 'U':

                    observed = True


                else:

                    observed = False

                if observed:

                    reference = self._parse_redshift_reference(row[obs_ref])


                    #gcn = "https://gcn.gsfc.nasa.gov/gcn3/%s.gcn3" % gcn_number

                    info = {'reference': reference, 'observed': detect}


                else:

                    info = {'GCN': None, 'observed': detect}

                obs_instrument[obs] = info

            sources[name] = obs_instrument

        sources = pd.concat(map(pd.DataFrame, sources.values()), keys=sources.keys())

        return sources

    def get_other_instrument_information(self):
        """
        Return the detectors used for spectral analysis as well as their background
        intervals. Peak flux and fluence intervals are also returned as well as best fit models

        :return: observing information dataframe indexed by source
        """

        assert self._last_query_results is not None, "You have to run a query before getting observing information"

        sources = {}

        for name, row in self._last_query_results.T.iteritems():

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
                if obs == '':

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
                    if obs == 'Fermi-GBM':

                        info = {'GCN': gcn, 'trigger number': self._get_fermiGBM_trigger_number_from_gcn(gcn)}

                    else:

                        info = {'GCN': gcn, 'trigger number': None}

                    obs_instrument[obs] = info

            sources[name] = obs_instrument

        # build the data frame
        sources = pd.concat(map(pd.DataFrame, sources.values()), keys=sources.keys())

        display(sources)

        return sources

    def get_redshift(self):
        """
        Get the redshift and redshift type from the searched sources


        :return:
        """

        assert self._last_query_results is not None, "You have to run a query before getting observing information"

        redshift_df = (self._last_query_results.loc[:,['redshift','redshift_err','redshift_type','redshift_ref']]).copy(deep=True)

        redshift_df = redshift_df.rename(columns={"redshift": "z", "redshift_err": "z err",'redshift_type': 'z type','redshift_ref':'reference'})

        redshift_df['reference'] = redshift_df['reference'].apply(self._parse_redshift_reference)

        return redshift_df


    @staticmethod
    def _parse_redshift_reference(reference):

        if 'GCN' in reference:
            gcn_number = _gcn_match.search(reference).group(1)

            url = "https://gcn.gsfc.nasa.gov/gcn3/%s.gcn3" % gcn_number

        else:

            url =  "http://adsabs.harvard.edu/abs/%s" % reference


        return url






