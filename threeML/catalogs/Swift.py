import numpy
from threeML.catalogs.VirtualObservatoryCatalog import VirtualObservatoryCatalog

from astromodels import *
from astromodels.utils.angular_distance import angular_distance

from threeML.exceptions.custom_exceptions import custom_warnings
from threeML.config.config import threeML_config
from threeML.io.get_heasarc_table_as_pandas import get_heasarc_table_as_pandas

import astropy.time as astro_time


class InvalidTrigger(RuntimeError):
    pass


class InvalidUTC(RuntimeError):
    pass


class SwiftGRBCatalog(VirtualObservatoryCatalog):
    def __init__(self, update=False):
        """
        The Swift GRB catalog. Search for GRBs  by trigger
        number, location, spectral parameters, T90, and date range.

        :param update: force update the XML VO table
        """

        super(SwiftGRBCatalog, self).__init__('swiftgrb',
                                              threeML_config['catalogs']['Swift']['Swift GRB catalog'],
                                              'Swift GRB catalog')

        self._vo_dataframe, self._all_table = get_heasarc_table_as_pandas('swiftgrb',
                                                                          update=update,
                                                                          cache_time_days=1.)

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

        obs_inst_ = map(numpy.unique, [np.asarray(self._vo_dataframe.other_obs),
                                       np.asarray(self._vo_dataframe.other_obs2),
                                       np.asarray(self._vo_dataframe.other_obs3),
                                       np.asarray(self._vo_dataframe.other_obs4)])

        self._other_observings_instruments = filter(lambda x: x != '', np.unique(numpy.concatenate(obs_inst_)))

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

        for trigger in trigger_names:

            assert_string = "The trigger %s is not valid. Must be in the form %s" % (trigger,
                                                                                     ', or '.join(
                                                                                             _valid_trigger_args))

            trigger = trigger.upper()

            search = re.search('(GRB|grb)? ?(\d{6}[a-zA-Z]?)', trigger)

            assert search is not None, assert_string

            assert search.group(2) is not None, assert_string

            valid_names.append("GRB %s" % search.group(2))

        n_entries = self._vo_dataframe.shape[0]

        idx = np.zeros(n_entries, dtype=bool)

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

        time = astro_time.Time(self._vo_dataframe.trigger_time, format='isot', scale='utc')

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
        Return GRBs also observed by the requested intrument

        """

        assert instrument in self._other_observings_instruments, "Other instrument choices include %s" % (
            ' ,'.join(self._other_observings_instruments))

        idx = np.zeros(self._vo_dataframe.shape[0])

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
        Parses the Fermi-GBM gcn to obtain the trigger number

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

    def get_other_observation_information(self):
        """
        Return the detectors used for spectral analysis as well as their background
        intervals. Peak flux and fluence intervals are also returned as well as best fit models

        :return: detector information dictionary
        """

        assert self._last_query_results is not None, "You have to run a query before getting detector information"

        # Loop over the table and build a source for each entry
        sources = {}

        for name, row in self._last_query_results.T.iteritems():

            # First we want to get the the detectors used in the SCAT file

            obs_instrument = {}

            for obs in ['xrt', 'uvot', 'bat', 'opt', 'radio']:

                obs_detection = "%s_detection" % obs
                print obs_detection

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

                    gcn_number = filter(lambda x: x != '', row[obs_ref].split('.'))[1]

                    gcn = "https://gcn.gsfc.nasa.gov/gcn3/%s.gcn3" % gcn_number

                    info = {'GCN': gcn, 'observed': detect}


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

        :return: detector information dictionary
        """

        assert self._last_query_results is not None, "You have to run a query before getting detector information"

        # Loop over the table and build a source for each entry
        sources = {}

        for name, row in self._last_query_results.T.iteritems():

            # First we want to get the the detectors used in the SCAT file

            obs_instrument = {}

            for obs in range(1, 5):

                if obs == 1:

                    obs_base = "other_obs"

                else:

                    obs_base = "other_obs%d" % obs

                obs_ref = "%s_ref" % obs_base

                obs = row[obs_base]

                if obs == '':

                    observed = False


                else:

                    observed = True

                if observed:

                    gcn_number = filter(lambda x: x != '', row[obs_ref].split('.'))[1]

                    gcn = "https://gcn.gsfc.nasa.gov/gcn3/%s.gcn3" % gcn_number

                    # just for Fermi GBM. can be expanded

                    if obs == 'Fermi-GBM':

                        info = {'GCN': gcn, 'trigger number': self._get_fermiGBM_trigger_number_from_gcn(gcn)}


                    else:

                        info = {'GCN': gcn, 'trigger number': None}

                    obs_instrument[obs] = info

            sources[name] = obs_instrument

        sources = pd.concat(map(pd.DataFrame, sources.values()), keys=sources.keys())

        return sources
