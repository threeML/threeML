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
from .catalog_utils import _gbm_and_lle_valid_source_check


log = setup_logger(__name__)


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
            threeML_config["catalogs"]["Fermi"]["catalogs"]["LLE catalog"].url,
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
