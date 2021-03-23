from __future__ import division

import re
from builtins import map, str

import numpy
from astromodels import *

from threeML.config.config import threeML_config
from threeML.io.dict_with_pretty_print import DictWithPrettyPrint
from threeML.io.get_heasarc_table_as_pandas import get_heasarc_table_as_pandas
from threeML.io.logging import setup_logger

from .VirtualObservatoryCatalog import VirtualObservatoryCatalog
from .catalog_utils import _get_point_source_from_fgl, _get_extended_source_from_fgl, ModelFromFGL
try:
    from fermipy.catalog import Catalog
    have_fermipy = True
except:
    have_fermipy = False


log = setup_logger(__name__)

_FGL_name_match = re.compile("^[34]FGL J\d{4}.\d(\+|-)\d{4}\D?$")

class FermiLATSourceCatalog(VirtualObservatoryCatalog):
    def __init__(self, update=False):

        self._update = update

        super(FermiLATSourceCatalog, self).__init__(
            "fermilpsc",
            threeML_config["catalogs"]["Fermi"]["catalogs"]["LAT FGL"].url,
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
            "The trigger %s is not valid. Must be in the form 'nFGL J0000.0+0000'"
            % source
        )

        match = _FGL_name_match.match(source)

        if match is None:

            log.warning(warn_string)

            answer = False

        else:

            answer = True

        return answer

        table["short_source_type"] = table["source_type"]
        table["source_type"] = numpy.array(list(map(translate, table["short_source_type"])))

        if "Search_Offset" in table.columns:

            new_table = table[
                "name",
                "source_type",
                "short_source_type",
                "ra",
                "dec",
                "assoc_name",
                "tevcat_assoc",
                "Search_Offset",
            ]

            return new_table.group_by("Search_Offset")

        # we may have not done a cone search!
        else:

            new_table = table[
                "name", "source_type", "short_source_type" "ra", "dec", "assoc_name", "tevcat_assoc"
            ]

            return new_table.group_by("name")



    def apply_format(self, table):

        fgl_types = {
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
        # Translate the 3 letter code to a more informative category, according
        # to the dictionary above
        def translate(key):
            if isinstance(key, bytes):
                key = key.decode("ascii")
            if key.lower() == "psr":
                return fgl_types[key]
            if key.lower() in list(fgl_types.keys()):
                return fgl_types[key.lower()]
            return key


    def get_model(self, use_association_name=True):

        assert (
            self._last_query_results is not None
        ), "You have to run a query before getting a model"

        # Loop over the table and build a source for each entry
        sources = []
        source_names = []
        for name, row in self._last_query_results.T.items():
                
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

            if row["extended_source_name"] is not None and row["extended_source_name"] != "":
        
                if "Spatial_Function" in row:
                    
                    this_source = _get_extended_source_from_fgl(this_name, row, fix=True)

                else:
                
                    log.warning(
                        "Source %s is extended, but not morphology information is available. "
                        "I will provide a point source instead" % name
                    )
                    this_source = _get_point_source_from_fgl(this_name, row, fix=True)

            else:
    
                this_source = _get_point_source_from_fgl(this_name, row, fix=True)

            sources.append(this_source)

        return ModelFromFGL(self.ra_center, self.dec_center, *sources)


class FermiPySourceCatalog(FermiLATSourceCatalog):

    def __init__(self, catalog_name = "4FGL", update=False):

        self._update = update

        super(FermiPySourceCatalog, self).__init__(update)

    def _get_vo_table_from_source(self):

        if not have_fermipy:
            
            log.error("Must have fermipy installed to use FermiPySourceCatalog")
            self._vo_dataframe = None
            
        else:
        
            try:
                
                self._fermipy_catalog = Catalog.create(catalog_name)
            
            except:
            
                log.error(f"Catalog {catalog_name} not available in fermipy")
                        
            self._astropy_table = self._fermipy_catalog.table

            #remove multi-dimensional columns
            for column in self._astropy_table.columns:
            
                if (("_History" in column) or ("_Band" in column)) or (column == "param_values"):
                
                    self._astropy_table.remove_column(column)
            
            self._vo_dataframe = self._table.to_pandas()

