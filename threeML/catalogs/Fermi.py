import warnings

from .catalog_utils import ModelFromFGL as ModelFrom3FGL
from .FermiGBM import FermiGBMBurstCatalog, FermiGBMTriggerCatalog
from .FermiLAT import FermiLATSourceCatalog, FermiPySourceCatalog
from .FermiLLE import FermiLLEBurstCatalog

warnings.warn(
    "Importing from 'threeML.catalogs.Fermi' is deprecated. "
    + "Use 'from threeML.catalogs'",
    DeprecationWarning,
    stacklevel=2,
)
__all__ = [
    "ModelFrom3FGL",
    "FermiGBMBurstCatalog",
    "FermiGBMTriggerCatalog",
    "FermiLATSourceCatalog",
    "FermiPySourceCatalog",
    "FermiLLEBurstCatalog",
]
