import numpy
from VirtualObservatoryCatalog import VirtualObservatoryCatalog

from astromodels import *
from astromodels.utils.angular_distance import angular_distance

from threeML.exceptions.custom_exceptions import custom_warnings
from threeML.config.config import threeML_config

class FermiGBMBurstCatalog(VirtualObservatoryCatalog):

    def __init__(self):

        super(FermiGBMBurstCatalog, self).__init__('fermigbrst',
                                                   threeML_config['catalogs']['Fermi']['GBM burst catalog'],
                                              'Fermi/GBM burst catalog')

    def apply_format(self, table):

        new_table = table['name',
                          'ra', 'dec',
                          'trigger_time',
                          'Search_Offset']

        new_table['ra'].format = '5.3f'
        new_table['dec'].format = '5.3f'

        return new_table.group_by('Search_Offset')


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
