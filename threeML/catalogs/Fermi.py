from __future__ import division

import re
from builtins import map, str

import numpy
from astromodels import *
from astromodels.utils.angular_distance import angular_distance
from past.utils import old_div

from threeML.config.config import threeML_config
from threeML.exceptions.custom_exceptions import custom_warnings
from threeML.io.dict_with_pretty_print import DictWithPrettyPrint
from threeML.io.get_heasarc_table_as_pandas import get_heasarc_table_as_pandas
from threeML.io.logging import setup_logger

from .VirtualObservatoryCatalog import VirtualObservatoryCatalog

log = setup_logger(__name__)

_trigger_name_match = re.compile("^GRB\d{9}$")
_3FGL_name_match = re.compile("^3FGL J\d{4}.\d(\+|-)\d{4}\D?$")


def _gbm_and_lle_valid_source_check(source):
    """
    checks if source name is valid for both GBM and LLE data

    :param source: source name
    :return: bool
    """

    warn_string = (
        "The trigger %s is not valid. Must be in the form GRB080916009" % source
    )

    match = _trigger_name_match.match(source)

    if match is None:

        log.warning(warn_string)

        answer = False

    else:

        answer = True

    return answer


class FermiGBMBurstCatalog(VirtualObservatoryCatalog):
    def __init__(self, update=False):
        """
        The Fermi-LAT GBM GRB catalog. Search for GRBs  by trigger
        number, location, spectral parameters, T90, and date range.

        :param update: force update the XML VO table
        """

        self._update = update

        super(FermiGBMBurstCatalog, self).__init__(
            "fermigbrst",
            threeML_config["catalogs"]["Fermi"]["GBM burst catalog"],
            "Fermi-LAT/GBM burst catalog",
        )

        self._gbm_detector_lookup = np.array(
            [
                "n0",
                "n1",
                "n2",
                "n3",
                "n4",
                "n5",
                "n6",
                "n7",
                "n8",
                "n9",
                "na",
                "nb",
                "b0",
                "b1",
            ]
        )

        self._available_models = ("band", "comp", "plaw", "sbpl")

    def _get_vo_table_from_source(self):

        self._vo_dataframe = get_heasarc_table_as_pandas(
            "fermigbrst", update=self._update, cache_time_days=1.0
        )

    def apply_format(self, table):
        new_table = table[
            "name",
            "ra",
            "dec",
            "trigger_time",
            "t90",
        ]

        new_table["ra"].format = "5.3f"
        new_table["dec"].format = "5.3f"

        return new_table.group_by("trigger_time")

    def _source_is_valid(self, source):

        return _gbm_and_lle_valid_source_check(source)

    def get_detector_information(self):
        """
        Return the detectors used for spectral analysis as well as their background
        intervals. Peak flux and fluence intervals are also returned as well as best fit models

        :return: detector information dictionary
        """

        assert (
            self._last_query_results is not None
        ), "You have to run a query before getting detector information"

        # Loop over the table and build a source for each entry
        sources = {}

        for name, row in self._last_query_results.T.items():
            # First we want to get the the detectors used in the SCAT file

            idx = np.array(list(map(int, row["scat_detector_mask"])), dtype=bool)
            detector_selection = self._gbm_detector_lookup[idx]

            # get the location

            ra = row["ra"]
            dec = row["dec"]

            # Now we want to know the background intervals

            lo_start = row["back_interval_low_start"]
            lo_stop = row["back_interval_low_stop"]
            hi_start = row["back_interval_high_start"]
            hi_stop = row["back_interval_high_stop"]

            # the GBM plugin accepts these as strings

            pre_bkg = "%f-%f" % (lo_start, lo_stop)
            post_bkg = "%f-%f" % (hi_start, hi_stop)
            full_bkg = "%s,%s" % (pre_bkg, post_bkg)

            background_dict = {"pre": pre_bkg, "post": post_bkg, "full": full_bkg}

            # now we want the fluence interval and peak flux intervals

            # first the fluence

            start_flu = row["t90_start"]
            stop_flu = row["t90_start"] + row["t90"]

            interval_fluence = "%f-%f" % (start_flu, stop_flu)

            # peak flux

            start_pk = row["pflx_spectrum_start"]
            stop_pk = row["pflx_spectrum_stop"]

            interval_pk = "%f-%f" % (start_pk, stop_pk)

            # build the dictionary
            spectrum_dict = {"fluence": interval_fluence, "peak": interval_pk}

            trigger = row["trigger_name"]

            # get the best fit model in the fluence and peak intervals

            best_fit_peak = row["pflx_best_fitting_model"].split("_")[-1]

            best_fit_fluence = row["flnc_best_fitting_model"].split("_")[-1]

            best_dict = {"fluence": best_fit_fluence, "peak": best_fit_peak}

            sources[name] = {
                "source": spectrum_dict,
                "background": background_dict,
                "trigger": trigger,
                "detectors": detector_selection,
                "best fit model": best_dict,
                "ra": ra,
                "dec": dec,
            }

        return DictWithPrettyPrint(sources)

    def get_model(self, model="band", interval="fluence"):
        """
        Return the fitted model from the Fermi-LAT GBM catalog in 3ML Model form.
        You can choose band, comp, plaw, or sbpl models corresponding to the models
        fitted in the GBM catalog. The interval for the fit can be the 'fluence' or
        'peak' interval

        :param model: one of 'band' (default), 'comp', 'plaw', 'sbpl'
        :param interval: 'peak' or 'fluence' (default)
        :return: a dictionary of 3ML likelihood models that can be fitted
        """

        # check the model name and the interval selection
        model = model.lower()

        assert (
            model in self._available_models
        ), "model is not in catalog. available choices are %s" % (", ").join(
            self._available_models
        )

        available_intervals = {"fluence": "flnc", "peak": "pflx"}

        assert interval in list(
            available_intervals.keys()
        ), "interval not recognized. choices are %s" % (
            " ,".join(list(available_intervals.keys()))
        )

        sources = {}
        lh_model = None

        for name, row in self._last_query_results.T.items():

            ra = row["ra"]
            dec = row["dec"]

            # name = str(name, 'utf-8')

            # get the proper 3ML model

            if model == "band":
                lh_model, shape = self._build_band(
                    name, ra, dec, row, available_intervals[interval]
                )

            if model == "comp":
                lh_model, shape = self._build_cpl(
                    name, ra, dec, row, available_intervals[interval]
                )

            if model == "plaw":
                lh_model, shape = self._build_powerlaw(
                    name, ra, dec, row, available_intervals[interval]
                )

            if model == "sbpl":
                lh_model, shape = self._build_sbpl(
                    name, ra, dec, row, available_intervals[interval]
                )

            # the assertion above should never let us get here
            if lh_model is None:
                raise RuntimeError("We should never get here. This is a bug")

            # return the model

            sources[name] = lh_model

        return sources

    @staticmethod
    def _build_band(name, ra, dec, row, interval):
        """
        builds a band function from the Fermi-LAT GBM catalog

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
        epeak = row[primary_string + "epeak"]
        alpha = row[primary_string + "alpha"]
        beta = row[primary_string + "beta"]
        amp = row[primary_string + "ampl"]

        band = Band()

        if amp < 0.0:
            amp = 0.0

        band.K = amp

        if epeak < band.xp.min_value:
            band.xp.min_value = epeak

        band.xp = epeak

        # The GBM catalog has some extreme alpha values

        if alpha < band.alpha.min_value:

            band.alpha.min_value = alpha

        elif alpha > band.alpha.max_value:

            band.alpha.max_value = alpha

        band.alpha = alpha

        # The GBM catalog has some extreme beta values

        if beta < band.beta.min_value:

            band.beta.min_value = beta

        elif beta > band.beta.max_value:

            band.beta.max_value = beta

        band.beta = beta

        # build the model
        ps = PointSource(name, ra, dec, spectral_shape=band)

        model = Model(ps)

        return model, band

    @staticmethod
    def _build_cpl(name, ra, dec, row, interval):
        """
        builds a cpl function from the Fermi-LAT GBM catalog

        :param name: GRB name
        :param ra: GRB ra
        :param dec: GRB de
        :param row: VO table row
        :param interval: interval type for parameters
        :return: 3ML likelihood model
        """

        primary_string = "%s_comp_" % interval

        epeak = row[primary_string + "epeak"]
        index = row[primary_string + "index"]
        pivot = row[primary_string + "pivot"]
        amp = row[primary_string + "ampl"]

        # need to correct epeak to e cut
        ecut = old_div(epeak, (2 - index))

        cpl = Cutoff_powerlaw()

        if amp < 0.0:
            amp = 0.0

        cpl.K = amp

        if ecut < cpl.xc.min_value:
            cpl.xc.min_value = ecut

        cpl.xc = ecut

        cpl.piv = pivot

        if index < cpl.index.min_value:

            cpl.index.min_value = index

        elif index > cpl.index.max_value:

            cpl.index.max_value = index

        cpl.index = index

        ps = PointSource(name, ra, dec, spectral_shape=cpl)

        model = Model(ps)

        return model, cpl

    @staticmethod
    def _build_powerlaw(name, ra, dec, row, interval):
        """
        builds a pl function from the Fermi-LAT GBM catalog

        :param name: GRB name
        :param ra: GRB ra
        :param dec: GRB de
        :param row: VO table row
        :param interval: interval type for parameters
        :return: 3ML likelihood model
        """

        primary_string = "%s_plaw_" % interval

        index = row[primary_string + "index"]
        pivot = row[primary_string + "pivot"]
        amp = row[primary_string + "ampl"]

        pl = Powerlaw()

        if amp < 0.0:
            amp = 0.0

        pl.K = amp
        pl.piv = pivot
        pl.index = index

        ps = PointSource(name, ra, dec, spectral_shape=pl)

        model = Model(ps)

        return model, pl

    @staticmethod
    def _build_sbpl(name, ra, dec, row, interval):
        """
        builds a sbpl function from the Fermi-LAT GBM catalog

        :param name: GRB name
        :param ra: GRB ra
        :param dec: GRB de
        :param row: VO table row
        :param interval: interval type for parameters
        :return: 3ML likelihood model
        """

        primary_string = "%s_sbpl_" % interval

        alpha = row[primary_string + "indx1"]
        beta = row[primary_string + "indx2"]
        amp = row[primary_string + "ampl"]
        break_scale = row[primary_string + "brksc"]
        break_energy = row[primary_string + "brken"]
        pivot = row[primary_string + "pivot"]

        sbpl = SmoothlyBrokenPowerLaw()

        if amp < 0.0:
            amp = 0.0

        sbpl.K = amp
        sbpl.pivot = pivot

        # The GBM catalog has some extreme alpha values

        if break_energy < sbpl.break_energy.min_value:
            sbpl.break_energy.min_value = break_energy

        sbpl.break_energy = break_energy

        if alpha < sbpl.alpha.min_value:

            sbpl.alpha.min_value = alpha

        elif alpha > sbpl.alpha.max_value:

            sbpl.alpha.max_value = alpha

        sbpl.alpha = alpha

        # The GBM catalog has some extreme beta values

        if beta < sbpl.beta.min_value:

            sbpl.beta.min_value = beta

        elif beta > sbpl.beta.max_value:

            sbpl.beta.max_value = beta

        sbpl.beta = beta
        sbpl.break_scale = break_scale

        sbpl.break_scale.free = True

        ps = PointSource(name, ra, dec, spectral_shape=sbpl)

        model = Model(ps)

        return model, sbpl


#########

threefgl_types = {
    "agn": "other non-blazar active galaxy",
    "bcu": "active galaxy of uncertain type",
    "bin": "binary",
    "bll": "BL Lac type of blazar",
    "css": "compact steep spectrum quasar",
    "fsrq": "FSRQ type of blazar",
    "gal": "normal galaxy (or part)",
    "glc": "globular cluster",
    "hmb": "high-mass binary",
    "nlsy1": "narrow line Seyfert 1",
    "nov": "nova",
    "PSR": "pulsar, identified by pulsations",
    "psr": "pulsar, no pulsations seen in LAT yet",
    "pwn": "pulsar wind nebula",
    "rdg": "radio galaxy",
    "sbg": "starburst galaxy",
    "sey": "Seyfert galaxy",
    "sfr": "star-forming region",
    "snr": "supernova remnant",
    "spp": "special case - potential association with SNR or PWN",
    "ssrq": "soft spectrum radio quasar",
    "unk": "unknown",
    "": "unknown",
}


def _sanitize_3fgl_name(fgl_name):
    swap = (
        fgl_name.replace(" ", "_").replace("+", "p").replace("-", "m").replace(".", "d")
    )

    if swap[0] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
        swap = "_%s" % swap

    return swap


def _get_point_source_from_3fgl(fgl_name, catalog_entry, fix=False):
    """
    Translate a spectrum from the 3FGL into an astromodels spectrum
    """

    name = _sanitize_3fgl_name(fgl_name)

    spectrum_type = catalog_entry["spectrum_type"]
    ra = float(catalog_entry["ra"])
    dec = float(catalog_entry["dec"])

    if spectrum_type == "PowerLaw":

        this_spectrum = Powerlaw()

        this_source = PointSource(name, ra=ra, dec=dec, spectral_shape=this_spectrum)

        this_spectrum.index = float(catalog_entry["pl_index"]) * -1
        this_spectrum.index.fix = fix
        this_spectrum.K = float(catalog_entry["pl_flux_density"]) / (
            u.cm ** 2 * u.s * u.MeV
        )
        this_spectrum.K.fix = fix
        this_spectrum.K.bounds = (
            this_spectrum.K.value / 1000.0,
            this_spectrum.K.value * 1000,
        )
        this_spectrum.piv = float(catalog_entry["pivot_energy"]) * u.MeV

    elif spectrum_type == "LogParabola":

        this_spectrum = Log_parabola()

        this_source = PointSource(name, ra=ra, dec=dec, spectral_shape=this_spectrum)

        this_spectrum.alpha = float(catalog_entry["lp_index"]) * -1
        this_spectrum.alpha.fix = fix
        this_spectrum.beta = float(catalog_entry["lp_beta"])
        this_spectrum.beta.fix = fix
        this_spectrum.piv = float(catalog_entry["pivot_energy"]) * u.MeV
        this_spectrum.K = float(catalog_entry["lp_flux_density"]) / (
            u.cm ** 2 * u.s * u.MeV
        )
        this_spectrum.K.fix = fix
        this_spectrum.K.bounds = (
            this_spectrum.K.value / 1000.0,
            this_spectrum.K.value * 1000,
        )

    elif spectrum_type == "PLExpCutoff":

        this_spectrum = Cutoff_powerlaw()

        this_source = PointSource(name, ra=ra, dec=dec, spectral_shape=this_spectrum)

        this_spectrum.index = float(catalog_entry["plec_index"]) * -1
        this_spectrum.index.fix = fix
        this_spectrum.piv = float(catalog_entry["pivot_energy"]) * u.MeV
        this_spectrum.K = float(catalog_entry["plec_flux_density"]) / (
            u.cm ** 2 * u.s * u.MeV
        )
        this_spectrum.K.fix = fix
        this_spectrum.K.bounds = (
            this_spectrum.K.value / 1000.0,
            this_spectrum.K.value * 1000,
        )
        this_spectrum.xc = float(catalog_entry["cutoff"]) * u.MeV
        this_spectrum.xc.fix = fix

    elif spectrum_type in ["PLSuperExpCutoff", "PLSuperExpCutoff2"]:
        # This is the new definition, from the 4FGL catalog.
        # Note that in version 19 of the 4FGL, cutoff spectra are designated as PLSuperExpCutoff
        # rather than PLSuperExpCutoff2 as in version , but the same parametrization is used.
        this_spectrum = Super_cutoff_powerlaw()

        this_source = PointSource(name, ra=ra, dec=dec, spectral_shape=this_spectrum)
        a = float(catalog_entry["plec_exp_factor"])
        E0 = float(catalog_entry["pivot_energy"])
        b = float(catalog_entry["plec_exp_index"])
        conv = math.exp(a * E0 ** b)
        this_spectrum.index = float(catalog_entry["plec_index"]) * -1
        this_spectrum.index.fix = fix
        this_spectrum.gamma = b
        this_spectrum.gamma.fix = fix
        this_spectrum.piv = E0 * u.MeV
        this_spectrum.K = (
            conv * float(catalog_entry["plec_flux_density"]) / (u.cm ** 2 * u.s * u.MeV)
        )
        this_spectrum.K.fix = fix
        this_spectrum.K.bounds = (
            this_spectrum.K.value / 1000.0,
            this_spectrum.K.value * 1000,
        )
        this_spectrum.xc = a ** (old_div(-1.0, b)) * u.MeV
        this_spectrum.xc.fix = fix

    else:

        raise NotImplementedError(
            "Spectrum type %s is not a valid 4FGL type" % spectrum_type
        )

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

            this_d = angular_distance(
                self._ra_center,
                self._dec_center,
                src.position.ra.value,
                src.position.dec.value,
            )

            if this_d <= radius:

                if normalization_only:

                    src.spectrum.main.shape.K.free = free

                else:

                    for par in src.spectrum.main.shape.parameters:
                        src.spectrum.main.shape.parameters[par].free = free


class FermiLATSourceCatalog(VirtualObservatoryCatalog):
    def __init__(self, update=False):

        self._update = update

        super(FermiLATSourceCatalog, self).__init__(
            "fermilpsc",
            threeML_config["catalogs"]["Fermi"]["LAT FGL"],
            "Fermi-LAT/LAT source catalog",
        )

    def _get_vo_table_from_source(self):

        self._vo_dataframe = get_heasarc_table_as_pandas(
            "fermilpsc", update=self._update, cache_time_days=10.0
        )

    def _source_is_valid(self, source):
        """
        checks if source name is valid for the 3FGL catalog

        :param source: source name
        :return: bool
        """

        warn_string = (
            "The trigger %s is not valid. Must be in the form '3FGL J0000.0+0000'"
            % source
        )

        match = _3FGL_name_match.match(source)

        if match is None:

            log.warning(warn_string)

            answer = False

        else:

            answer = True

        return answer

    def apply_format(self, table):
        def translate(key):
            if key.lower() == "psr":
                return threefgl_types[key]
            if key.lower() in list(threefgl_types.keys()):
                return threefgl_types[key.lower()]
            return "unknown"

        # Translate the 3 letter code to a more informative category, according
        # to the dictionary above

        table["source_type"] = numpy.array(list(map(translate, table["source_type"])))

        try:

            new_table = table[
                "name",
                "source_type",
                "ra",
                "dec",
                "assoc_name",
                "tevcat_assoc",
                "Search_Offset",
            ]

            return new_table.group_by("Search_Offset")

        # we may have not done a cone search!
        except (ValueError):

            new_table = table[
                "name", "source_type", "ra", "dec", "assoc_name", "tevcat_assoc"
            ]

            return new_table.group_by("name")

    def get_model(self, use_association_name=True):

        assert (
            self._last_query_results is not None
        ), "You have to run a query before getting a model"

        # Loop over the table and build a source for each entry
        sources = []
        source_names = []
        for name, row in self._last_query_results.T.items():
            if name[-1] == "e":
                # Extended source
                log.warning(
                    "Source %s is extended, support for extended source is not here yet. I will ignore"
                    "it" % name
                )

            # If there is an association and use_association is True, use that name, otherwise the 3FGL name
            if row["assoc_name"] != "" and use_association_name:

                this_name = row["assoc_name"]

                # The crab is the only source which is present more than once in the 3FGL

                if this_name == "Crab Nebula":

                    if name[-1] == "i":

                        this_name = "Crab_IC"

                    elif name[-1] == "s":

                        this_name = "Crab_synch"

                    else:

                        this_name = "Crab_pulsar"
            else:

                this_name = name

            # in the 4FGL name there are more sources with the same name: this nwill avod any duplicates:
            i = 1
            while this_name in source_names:
                this_name += str(i)
                i += 1
                pass
            # By default all sources are fixed. The user will free the one he/she will need

            source_names.append(this_name)

            this_source = _get_point_source_from_3fgl(this_name, row, fix=True)

            sources.append(this_source)

        return ModelFrom3FGL(self.ra_center, self.dec_center, *sources)


class FermiLLEBurstCatalog(VirtualObservatoryCatalog):
    def __init__(self, update=False):
        """
        The Fermi-LAT LAT Low-Energy (LLE) trigger catalog. Search for GRBs and solar flares by trigger
        number, location, trigger type and date range.

        :param update: force update the XML VO table
        """

        self._update = update

        super(FermiLLEBurstCatalog, self).__init__(
            "fermille",
            threeML_config["catalogs"]["Fermi"]["LLE catalog"],
            "Fermi-LAT/LLE catalog",
        )

    def apply_format(self, table):
        new_table = table["name", "ra", "dec", "trigger_time", "trigger_type"]

        # Remove rows with masked elements in trigger_time column
        if new_table.masked:
            new_table = new_table[~new_table["trigger_time"].mask]

        new_table["ra"].format = "5.3f"
        new_table["dec"].format = "5.3f"

        return new_table.group_by("trigger_time")

    def _get_vo_table_from_source(self):

        self._vo_dataframe = get_heasarc_table_as_pandas(
            "fermille", update=self._update, cache_time_days=5.0
        )

    def _source_is_valid(self, source):

        return _gbm_and_lle_valid_source_check(source)
