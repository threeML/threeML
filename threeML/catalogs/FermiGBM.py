from __future__ import division

import re
from builtins import map, str

import astromodels
import numpy
from past.utils import old_div

from threeML.config.config import threeML_config
from threeML.io.dict_with_pretty_print import DictWithPrettyPrint
from threeML.io.get_heasarc_table_as_pandas import get_heasarc_table_as_pandas
from threeML.io.logging import setup_logger

from .catalog_utils import _gbm_and_lle_valid_source_check
from .VirtualObservatoryCatalog import VirtualObservatoryCatalog

log = setup_logger(__name__)


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
            threeML_config["catalogs"]["Fermi"]["catalogs"]["GBM burst catalog"].url,
            "Fermi-LAT/GBM burst catalog",
        )

        self._gbm_detector_lookup = numpy.array(
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

            idx = numpy.array(list(map(int, row["scat_detector_mask"])), dtype=bool)
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

            background_dict = {
                "pre": pre_bkg,
                "post": post_bkg,
                "full": full_bkg,
            }

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

        band = astromodels.Band()

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
        ps = astromodels.PointSource(name, ra, dec, spectral_shape=band)

        model = astromodels.Model(ps)

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

        cpl = astromodels.Cutoff_powerlaw()

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

        ps = astromodels.PointSource(name, ra, dec, spectral_shape=cpl)

        model = astromodels.Model(ps)

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

        pl = astromodels.Powerlaw()

        if amp < 0.0:
            amp = 0.0

        pl.K = amp
        pl.piv = pivot
        pl.index = index

        ps = astromodels.PointSource(name, ra, dec, spectral_shape=pl)

        model = astromodels.Model(ps)

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

        sbpl = astromodels.SmoothlyBrokenPowerLaw()

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

        ps = astromodels.PointSource(name, ra, dec, spectral_shape=sbpl)

        model = astromodels.Model(ps)

        return model, sbpl


class FermiGBMTriggerCatalog(VirtualObservatoryCatalog):
    def __init__(self, update=False):
        """
        The Fermi-GBM trigger catalog.

        :param update: force update the XML VO table
        """

        self._update = update

        super(FermiGBMTriggerCatalog, self).__init__(
            "fermigtrig",
            threeML_config["catalogs"]["Fermi"]["catalogs"]["GBM trigger catalog"].url,
            "Fermi-GBM trigger catalog",
        )

    def _get_vo_table_from_source(self):

        self._vo_dataframe = get_heasarc_table_as_pandas(
            "fermigtrig", update=self._update, cache_time_days=1.0
        )

    def apply_format(self, table):
        new_table = table[
            "name",
            "trigger_type",
            "ra",
            "dec",
            "trigger_time",
            "localization_source",
        ]

        new_table["ra"].format = "5.3f"
        new_table["dec"].format = "5.3f"

        return new_table.group_by("trigger_time")

    def _source_is_valid(self, source):

        return _gbm_and_lle_valid_source_check(source)
