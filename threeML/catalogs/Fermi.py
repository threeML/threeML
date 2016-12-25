import numpy
from VirtualObservatoryCatalog import VirtualObservatoryCatalog
from astropy.time import Time

from astromodels import *
from astromodels.utils.angular_distance import angular_distance

from threeML.exceptions.custom_exceptions import custom_warnings
from threeML.config.config import threeML_config


class InvalidTrigger(RuntimeError):
    pass


class FermiGBMBurstCatalog(VirtualObservatoryCatalog):
    def __init__(self):

        super(FermiGBMBurstCatalog, self).__init__('fermigbrst',
                                                   threeML_config['catalogs']['Fermi']['GBM burst catalog'],
                                                   'Fermi/GBM burst catalog')

        self._gbm_detector_lookup = np.array(['n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6',
                                              'n7', 'n8', 'n9', 'na', 'nb', 'b0', 'b1'])

        self._available_models = ('band', 'compt', 'pl', 'sbpl')

        self._grabbed_all_data = False

    def apply_format(self, table):

        new_table = table['name',
                          'ra', 'dec',
                          'trigger_time',
                          't90',
                          'Search_Offset',
        ]

        new_table['ra'].format = '5.3f'
        new_table['dec'].format = '5.3f'

        return new_table.group_by('trigger_time')

    def search_trigger_name(self, *trigger_names):
        """
        Find the information on the given trigger names.
        First run will be slow. Subsequent runs are very fast

        :param trigger_names: trigger numbers (str) e.g. '080916009' or 'bn080916009' or 'GRB080916009'
        :return:
        """

        # If we have not downloaded the entire table
        # then we must do so.

        # check the trigger names

        _valid_trigger_args = ['080916008', 'bn080916009', 'GRB080916009']

        for trigger in trigger_names:

            assert type(trigger) == str, "The trigger %s is not valid. Must be in the form %s" % (trigger,
                                                                                                  ', or '.join(
                                                                                                          _valid_trigger_args))

            test = trigger.split('bn')

            assert len(test) == 2, "The trigger %s is not valid. Must be in the form %s" % (test,
                                                                                            ', or '.join(
                                                                                                _valid_trigger_args))

            trigger = test[-1]

            assert len(trigger) == 9, "The trigger %s is not valid. Must be in the form %s" % (trigger,
                                                                                               ', or '.join(
                                                                                                   _valid_trigger_args))

            for trial in trigger:

                try:

                    int(trial)

                except(ValueError):

                    raise InvalidTrigger(
                            "The trigger %s is not valid. Must be in the form %s" % (trigger,
                                                                                     ', or '.join(_valid_trigger_args)))


        if not self._grabbed_all_data:

            self._all_table = self.cone_search(0, 0, 360)

            self._completed_table = self._last_query_results

            self._grabbed_all_data = True

        n_entries = self._completed_table.shape[0]

        idx = np.zeros(n_entries, dtype=bool)

        for name in trigger_names:

            condition = self._completed_table.trigger_name == name

            idx = np.logical_or(idx, condition)

        self._last_query_results = self._completed_table[idx]

        return self._all_table[np.asarray(idx)]

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



        if not self._grabbed_all_data:

            self._all_table = self.cone_search(0, 0, 360)

            self._completed_table = self._last_query_results

            self._grabbed_all_data = True

        n_entries = self._completed_table.shape[0]

        # Create a dummy index first

        idx = np.ones(n_entries, dtype=bool)

        # find the entries greater
        if t90_greater is not None:

            idx_tmp = self._completed_table.t90 >= t90_greater

            idx = np.logical_and(idx, idx_tmp)

        # find the entries less
        if t90_less is not None:

            idx_tmp = self._completed_table.t90 <= t90_less

            idx = np.logical_and(idx, idx_tmp)

        # save this look up
        self._last_query_results = self._completed_table[idx]

        return self._all_table[np.asarray(idx)]

    def search_mjd(self, mjd_start, mjd_stop):
        """
        search for triggers in a range of MJD

        :param mjd_start: start of MJD range
        :param mjd_stop:  stop of MJD range
        :return: table of results
        """

        assert mjd_start < mjd_stop, "start must come before stop"

        if not self._grabbed_all_data:

            self._all_table = self.cone_search(0, 0, 360)

            self._completed_table = self._last_query_results

            self._grabbed_all_data = True

        n_entries = self._completed_table.shape[0]

        idx = np.logical_and(mjd_start <= self._completed_table.trigger_time,
                             self._completed_table.trigger_time <= mjd_stop)

        # save this look up
        self._last_query_results = self._completed_table[idx]

        return self._all_table[np.asarray(idx)]

    def search_utc(self, utc_start, utc_stop):
        """
        Search for triggers in a range of UTC values. UTC time must be specified
         in the as follows '1999-01-01T00:00:00.123456789' '2010-01-01T00:00:00'
        :param utc_start: start of UTC interval
        :param utc_stop: stop of UTC interval
        :return:
        """

        # use astropy to read the UTC format
        utc_start, utc_stop = Time([utc_start, utc_stop], format='isot', scale='utc')

        # convert the UTC format to MJD and use the MJD search
        return self.search_mjd(utc_start.mjd, utc_stop.mjd)



    def get_detector_information(self):
        """
        Return the detectors used for spectral analysis as well as their background
        intervals. Peak flux and fluence intervals are also returned

        :return: detector information dictionary
        """

        assert self._last_query_results is not None, "You have to run a query before getting detector information"

        # Loop over the table and build a source for each entry
        sources = {}

        for name, row in self._last_query_results.T.iteritems():

            # First we want to get the the detectors used in the SCAT file

            idx = np.array(map(int, row['scat_detector_mask']), dtype=bool)
            detector_selection = self._gbm_detector_lookup[idx]

            # Now we want to know the background intervals

            lo_start = row['back_interval_low_start']
            lo_stop = row['back_interval_low_stop']
            hi_start = row['back_interval_high_start']
            hi_stop = row['back_interval_high_stop']

            # the GBM plugin accepts these as strings

            pre_bkg = "%f-%f" % (lo_start, lo_stop)
            post_bkg = "%f-%f" % (hi_start, hi_stop)
            full_bkg = "%s,%s" % (pre_bkg, post_bkg)

            background_dict = {'pre': pre_bkg, 'post': post_bkg, 'full': full_bkg}

            # now we want the fluence interval and peak flux intervals

            # first the fluence

            start_flu = row['t90_start']
            stop_flu = row['t90_start'] + row['t90']

            interval_fluence = "%f-%f" % (start_flu, stop_flu)

            # peak flux

            start_pk = row['pflx_spectrum_start']
            stop_pk = row['pflx_spectrum_stop']

            interval_pk = "%f-%f" % (start_pk, stop_pk)

            # build the dictionary
            spectrum_dict = {'fluence': interval_fluence, 'peak': interval_pk}

            trigger = row['trigger_name']

            sources[name] = {'source': spectrum_dict, 'background': background_dict, 'trigger': trigger}

        return sources

    def get_duration_information(self):
        """
        Return the T90 and T50 information

        :return: duration dictionary
        """

        assert self._last_query_results is not None, "You have to run a query before getting detector information"

        # Loop over the table and build a source for each entry
        sources = {}

        for name, row in self._last_query_results.T.iteritems():

            # T90
            start_t90 = row['t90_start']
            t90 = row['t90']
            t90_err = row['t90_error']

            # T50
            start_t50 = row['t50_start']
            t50 = row['t50']
            t50_err = row['t50_error']

            sources[name] = {'T90': {'value': t90, 'err': t90_err, 'start': start_t90},
                             'T50': {'value': t50, 'err': t50_err, 'start': start_t50}}

        return sources

    def get_model(self, model='band', interval='fluence'):
        """
        Return the fitted model from the Fermi GBM catalog in 3ML Model form.
        You can choose band, compt, pl, or sbpl models corresponding to the models
        fitted in the GBM catalog. The interval for the fit can be the 'fluence' or
        'peak' interval

        :param model: one of 'band' (default), 'compt', 'pl', 'sbpl'
        :param interval: 'peak' or 'fluence' (default)
        :return: a dictionary of 3ML likelihood models that can be fitted
        """

        # check the model name and the interval selection
        model = model.lower()

        assert model in self._available_models, 'model is not in catalog. available choices are %s' % (', ').join(
                self._available_models)

        available_intervals = {'fluence': 'flnc', 'peak': 'plfx'}

        assert interval in available_intervals.keys(), 'interval not recognized. choices are %s' % (
            ' ,'.join(available_intervals.keys()))

        sources = {}
        lh_model = None

        for name, row in self._last_query_results.T.iteritems():

            ra = row['ra']
            dec = row['dec']

            # get the proper 3ML model

            if model == 'band':

                lh_model = self._build_band(name, ra, dec, row, available_intervals[interval])

            if model == 'compt':

                lh_model = self._build_cpl(name, ra, dec, row, available_intervals[interval])

            if model == 'pl':

                lh_model = self._build_powerlaw(name, ra, dec, row, available_intervals[interval])

            if model == 'sbpl':

                lh_model = self._build_sbpl(name, ra, dec, row, available_intervals[interval])

            # the assertion above should never let us get here
            if lh_model is None:

                raise RuntimeError("We should never get here. This is a bug")

            # return the model

            sources[name] = lh_model

        return sources

    @staticmethod
    def _build_band(name, ra, dec, row, interval):
        """
        builds a band function from the Fermi GBM catalog

        :param name: GRB name
        :param ra: GRB ra
        :param dec: GRB de
        :param row: VO table row
        :param interval: interval type for parameters
        :return: 3ML likelihood model
        """

        # construct the primary string

        primary_string = "%s_band_" % interval

        # get the parameters
        epeak = row[primary_string + 'epeak']
        alpha = row[primary_string + 'alpha']
        beta = row[primary_string + 'beta']
        amp = row[primary_string + 'ampl']

        band = Band()

        band.K = amp
        band.xp = epeak

        # The GBM catalog has some extreme alpha values

        if alpha < band.alpha.min_value:

            band.alpha.min_value = alpha

        if alpha > band.alpha.max_value:

            band.alpha.max_value = alpha

        band.alpha = alpha

        # The GBM catalog has some extreme beta values

        if beta < band.beta.min_value:

            band.beta.min_value = beta

        if beta > band.beta.max_value:

            band.beta.max_value = beta

        band.beta = beta

        # build the model
        ps = PointSource(name, ra, dec, spectral_shape=band)

        model = Model(ps)

        return model

    @staticmethod
    def _build_cpl(name, ra, dec, row, interval):
        """
        builds a cpl function from the Fermi GBM catalog

        :param name: GRB name
        :param ra: GRB ra
        :param dec: GRB de
        :param row: VO table row
        :param interval: interval type for parameters
        :return: 3ML likelihood model
        """

        # need to correct epeak
        primary_string = "%s_comp_" % interval

        epeak = row[primary_string + 'epeak']
        index = row[primary_string + 'index']
        pivot = row[primary_string + 'pivot']
        amp = row[primary_string + 'ampl']

        cpl = Cutoff_powerlaw()

        cpl.K = amp
        cpl.xc = epeak / (2 - index)
        cpl.piv = pivot
        cpl.index = index

        ps = PointSource(name, ra, dec, spectral_shape=cpl)

        model = Model(ps)

        return model

    @staticmethod
    def _build_powerlaw(name, ra, dec, row, interval):
        """
        builds a pl function from the Fermi GBM catalog

        :param name: GRB name
        :param ra: GRB ra
        :param dec: GRB de
        :param row: VO table row
        :param interval: interval type for parameters
        :return: 3ML likelihood model
        """

        primary_string = "%s_plaw_" % interval

        index = row[primary_string + 'index']
        pivot = row[primary_string + 'pivot']
        amp = row[primary_string + 'ampl']

        pl = Powerlaw()

        pl.K = amp
        pl.piv = pivot
        pl.index = index

        ps = PointSource(name, ra, dec, spectral_shape=pl)

        model = Model(ps)

        return model

    @staticmethod
    def _build_sbpl(name, ra, dec, row, interval):
        """
        builds a sbpl function from the Fermi GBM catalog

        :param name: GRB name
        :param ra: GRB ra
        :param dec: GRB de
        :param row: VO table row
        :param interval: interval type for parameters
        :return: 3ML likelihood model
        """

        primary_string = "%s_sbpl_" % interval

        alpha = row[primary_string + 'indx1']
        beta = row[primary_string + 'indx2']
        amp = row[primary_string + 'ampl']
        break_scale = row[primary_string + 'brksc']
        break_energy = row[primary_string + 'brken']
        pivot = row[primary_string + 'pivot']

        sbpl = SmoothlyBrokenPowerLaw()

        sbpl.K = amp
        sbpl.pivot = pivot

        # The GBM catalog has some extreme alpha values

        if alpha < sbpl.alpha.min_value:

            sbpl.alpha.min_value = alpha

        if alpha > sbpl.alpha.max_value:

            sbpl.alpha.max_value = alpha

        sbpl.alpha = alpha

        # The GBM catalog has some extreme beta values

        if beta < sbpl.beta.min_value:

            sbpl.beta.min_value = beta

        if beta > sbpl.beta.max_value:

            sbpl.beta.max_value = beta

        sbpl.beta = beta
        sbpl.break_scale = break_scale
        sbpl.break_energy = break_energy

        sbpl.break_scale.free = True

        ps = PointSource(name, ra, dec, spectral_shape=sbpl)

        model = Model(ps)

        return model





#########

threefgl_types = {
    'agn': 'other non-blazar active galaxy',
    'bcu': 'active galaxy of uncertain type',
    'bin': 'binary',
    'bll': 'BL Lac type of blazar',
    'css': 'compact steep spectrum quasar',
    'fsrq': 'FSRQ type of blazar',
    'gal': 'normal galaxy (or part)',
    'glc': 'globular cluster',
    'hmb': 'high-mass binary',
    'nlsy1': 'narrow line Seyfert 1',
    'nov': 'nova',
    'PSR': 'pulsar, identified by pulsations',
    'psr': 'pulsar, no pulsations seen in LAT yet',
    'pwn': 'pulsar wind nebula',
    'rdg': 'radio galaxy',
    'sbg': 'starburst galaxy',
    'sey': 'Seyfert galaxy',
    'sfr': 'star-forming region',
    'snr': 'supernova remnant',
    'spp': 'special case - potential association with SNR or PWN',
    'ssrq': 'soft spectrum radio quasar',
    '': 'unknown'
}


def _sanitize_3fgl_name(fgl_name):
    swap = fgl_name.replace(" ", "_").replace("+", "p").replace("-", "m").replace(".", "d")

    if swap[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
        swap = "_%s" % swap

    return swap


def _get_point_source_from_3fgl(fgl_name, catalog_entry, fix=False):
    """
    Translate a spectrum from the 3FGL into an astromodels spectrum
    """

    name = _sanitize_3fgl_name(fgl_name)

    spectrum_type = catalog_entry['spectrum_type']
    ra = float(catalog_entry['ra'])
    dec = float(catalog_entry['dec'])

    if spectrum_type == 'PowerLaw':

        this_spectrum = Powerlaw()

        this_source = PointSource(name, ra=ra, dec=dec, spectral_shape=this_spectrum)

        this_spectrum.index = float(catalog_entry['powerlaw_index']) * -1
        this_spectrum.index.fix = fix
        this_spectrum.K = float(catalog_entry['flux_density']) / (u.cm ** 2 * u.s * u.MeV)
        this_spectrum.K.fix = fix
        this_spectrum.K.bounds = (this_spectrum.K.value / 1000.0, this_spectrum.K.value * 1000)
        this_spectrum.piv = float(catalog_entry['pivot_energy']) * u.MeV

    elif spectrum_type == 'LogParabola':

        this_spectrum = Log_parabola()

        this_source = PointSource(name, ra=ra, dec=dec, spectral_shape=this_spectrum)

        this_spectrum.alpha = float(catalog_entry['spectral_index']) * -1
        this_spectrum.alpha.fix = fix
        this_spectrum.beta = float(catalog_entry['beta'])
        this_spectrum.beta.fix = fix
        this_spectrum.piv = float(catalog_entry['pivot_energy']) * u.MeV
        this_spectrum.K = float(catalog_entry['flux_density']) / (u.cm ** 2 * u.s * u.MeV)
        this_spectrum.K.fix = fix
        this_spectrum.K.bounds = (this_spectrum.K.value / 1000.0, this_spectrum.K.value * 1000)

    elif spectrum_type == 'PLExpCutoff':

        this_spectrum = Cutoff_powerlaw()

        this_source = PointSource(name, ra=ra, dec=dec, spectral_shape=this_spectrum)

        this_spectrum.index = float(catalog_entry['spectral_index']) * -1
        this_spectrum.index.fix = fix
        this_spectrum.piv = float(catalog_entry['pivot_energy']) * u.MeV
        this_spectrum.K = float(catalog_entry['flux_density']) / (u.cm ** 2 * u.s * u.MeV)
        this_spectrum.K.fix = fix
        this_spectrum.K.bounds = (this_spectrum.K.value / 1000.0, this_spectrum.K.value * 1000)
        this_spectrum.xc = float(catalog_entry['cutoff']) * u.MeV
        this_spectrum.xc.fix = fix

    elif spectrum_type == 'PLSuperExpCutoff':

        this_spectrum = Super_cutoff_powerlaw()

        this_source = PointSource(name, ra=ra, dec=dec, spectral_shape=this_spectrum)

        this_spectrum.index = float(catalog_entry['spectral_index']) * -1
        this_spectrum.index.fix = fix
        this_spectrum.gamma = float(catalog_entry['exp_index'])
        this_spectrum.gamma.fix = fix
        this_spectrum.piv = float(catalog_entry['pivot_energy']) * u.MeV
        this_spectrum.K = float(catalog_entry['flux_density']) / (u.cm ** 2 * u.s * u.MeV)
        this_spectrum.K.fix = fix
        this_spectrum.K.bounds = (this_spectrum.K.value / 1000.0, this_spectrum.K.value * 1000)
        this_spectrum.xc = float(catalog_entry['cutoff']) * u.MeV
        this_spectrum.xc.fix = fix

    else:

        raise NotImplementedError("Spectrum type %s is not a valid 3FGL type" % spectrum_type)

    return this_source


class ModelFrom3FGL(Model):
    def __init__(self, ra_center, dec_center, *sources):

        self._ra_center = float(ra_center)
        self._dec_center = float(dec_center)

        super(ModelFrom3FGL, self).__init__(*sources)

    def free_point_sources_within_radius(self, radius, normalization_only=True):
        """
        Free the parameters for the point sources within the given radius of the center of the search cone

        :param radius: radius in degree
        :param normalization_only: if True, frees only the normalization of the source (default: True)
        :return: none
        """
        self._free_or_fix(True, radius, normalization_only)

    def fix_point_sources_within_radius(self, radius, normalization_only=True):
        """
        Fixes the parameters for the point sources within the given radius of the center of the search cone

        :param radius: radius in degree
        :param normalization_only: if True, fixes only the normalization of the source (default: True)
        :return: none
        """
        self._free_or_fix(False, radius, normalization_only)

    def _free_or_fix(self, free, radius, normalization_only):

        for src_name in self.point_sources:

            src = self.point_sources[src_name]

            this_d = angular_distance(self._ra_center, self._dec_center, src.position.ra.value, src.position.dec.value)

            if this_d <= radius:

                if normalization_only:

                    src.spectrum.main.shape.K.free = free

                else:

                    for par in src.spectrum.main.parameters:

                        src.spectrum.main.parameters[par].free = free


class FermiLATSourceCatalog(VirtualObservatoryCatalog):
    def __init__(self):

        super(FermiLATSourceCatalog, self).__init__('fermilpsc',
                                                    threeML_config['catalogs']['Fermi']['LAT FGL'],
                                               'Fermi/LAT source catalog')

    def apply_format(self, table):

        def translate(key):
            if (key.lower() == 'psr'):
                return threefgl_types[key]
            else:
                return threefgl_types[key.lower()]

        # Translate the 3 letter code to a more informative category, according
        # to the dictionary above

        table['source_type'] = numpy.array(map(translate, table['source_type']))

        new_table = table['name',
                          'source_type',
                          'ra', 'dec',
                          'assoc_name_1',
                          'tevcat_assoc',
                          'Search_Offset']

        return new_table.group_by('Search_Offset')

    def get_model(self, use_association_name=True):

        assert self._last_query_results is not None, "You have to run a query before getting a model"

        # Loop over the table and build a source for each entry
        sources = []

        for name, row in self._last_query_results.T.iteritems():

            if name[-1] == 'e':
                # Extended source
                custom_warnings.warn("Source %s is extended, support for extended source is not here yet. I will ignore"
                                     "it" % name)

            # If there is an association and use_association is True, use that name, otherwise the 3FGL name
            if row['assoc_name_1'] != '' and use_association_name:

                this_name = row['assoc_name_1']

                # The crab is the only source which is present more than once in the 3FGL

                if this_name == "Crab":

                    if name[-1]=='i':

                        this_name = "Crab_IC"

                    elif name[-1]=="s":

                        this_name = "Crab_synch"

                    else:

                        this_name = "Crab_pulsar"

            else:

                this_name = name

            # By default all sources are fixed. The user will free the one he/she will need

            this_source = _get_point_source_from_3fgl(this_name, row, fix=True)

            sources.append(this_source)

        return ModelFrom3FGL(self.ra_center, self.dec_center, *sources)


class FermiLLEBurstCatalog(VirtualObservatoryCatalog):
    def __init__(self):
        super(FermiLLEBurstCatalog, self).__init__('fermille',
                                                   threeML_config['catalogs']['Fermi']['LLE catalog'],
                                              'Fermi/LLE catalog')

        self._grabbed_all_data = False


    def apply_format(self, table):
        new_table = table['name',
                          'ra', 'dec',
                          'trigger_time',
                          'trigger_type',
                          'Search_Offset',
        ]

        new_table['ra'].format = '5.3f'
        new_table['dec'].format = '5.3f'

        return new_table.group_by('trigger_time')

    def search_trigger_name(self, *trigger_names):
        """
        Find the information on the given trigger names.
        First run will be slow. Subsequent runs are very fast

        :param trigger_names: trigger numbers (str) e.g. '080916009' or 'bn080916009' or 'GRB080916009'
        :return:
        """

        # If we have not downloaded the entire table
        # then we must do so.

        # check the trigger names

        _valid_trigger_args = ['bn080916009']



        for trigger in trigger_names:

            assert type(trigger) == str, "The trigger %s is not valid. Must be in the form %s" % (trigger,
                                                                                                  ', or '.join(
                                                                                                          _valid_trigger_args))

            test = trigger.split('bn')

            assert len(test) == 2, "The trigger %s is not valid. Must be in the form %s" % (test,
                                                                                            ', or '.join(
                                                                                                    _valid_trigger_args))

            trigger = test[-1]

            assert len(trigger) == 9, "The trigger %s is not valid. Must be in the form %s" % (trigger,
                                                                                               ', or '.join(
                                                                                                       _valid_trigger_args))

            for trial in trigger:

                try:

                    int(trial)

                except(ValueError):

                    raise InvalidTrigger(
                            "The trigger %s is not valid. Must be in the form %s" % (trigger,
                                                                                     ', or '.join(_valid_trigger_args)))

        if not self._grabbed_all_data:

            self._all_table = self.cone_search(0, 0, 360)

            self._completed_table = self._last_query_results

            self._grabbed_all_data = True

        n_entries = self._completed_table.shape[0]

        idx = np.zeros(n_entries, dtype=bool)

        for name in trigger_names:

            condition = self._completed_table.trigger_name == name

            idx = np.logical_or(idx, condition)

        self._last_query_results = self._completed_table[idx]

        return self._all_table[np.asarray(idx)]

    def search_mjd(self, mjd_start, mjd_stop):
        """
        search for triggers in a range of MJD

        :param mjd_start: start of MJD range
        :param mjd_stop:  stop of MJD range
        :return: table of results
        """

        assert mjd_start < mjd_stop, "start must come before stop"

        if not self._grabbed_all_data:

            self._all_table = self.cone_search(0, 0, 360)

            self._completed_table = self._last_query_results

            self._grabbed_all_data = True

        n_entries = self._completed_table.shape[0]

        idx = np.logical_and(mjd_start <= self._completed_table.trigger_time,
                             self._completed_table.trigger_time <= mjd_stop)

        # save this look up
        self._last_query_results = self._completed_table[idx]

        return self._all_table[np.asarray(idx)]

    def search_utc(self, utc_start, utc_stop):
        """
        Search for triggers in a range of UTC values. UTC time must be specified
         in the as follows '1999-01-01T00:00:00.123456789' '2010-01-01T00:00:00'
        :param utc_start: start of UTC interval
        :param utc_stop: stop of UTC interval
        :return:
        """

        # use astropy to read the UTC format
        utc_start, utc_stop = Time([utc_start, utc_stop], format='isot', scale='utc')

        # convert the UTC format to MJD and use the MJD search
        return self.search_mjd(utc_start.mjd, utc_stop.mjd)
