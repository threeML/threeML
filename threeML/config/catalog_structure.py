from dataclasses import dataclass, field
from enum import Enum, Flag
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
from omegaconf import II, MISSING, SI, OmegaConf


@dataclass(frozen=True)
class PublicDataServer:
    public_ftp_location: Optional[str] = None
    public_http_location: str = MISSING
    query_form: Optional[str] = None


@dataclass(frozen=True)
class CatalogServer:
    url: str = MISSING


@dataclass
class InstrumentCatalog:
    catalogs: Dict[str, CatalogServer] = MISSING


@dataclass(frozen=True)
class Catalogs:
    Fermi: InstrumentCatalog = InstrumentCatalog({"LAT FGL": CatalogServer("https://heasarc.gsfc.nasa.gov/cgi-bin/vo/cone/coneGet.pl?table=fermilpsc&"),
                                                  "GBM burst catalog": CatalogServer("https://heasarc.gsfc.nasa.gov/cgi-bin/vo/cone/coneGet.pl?table=fermigbrst&"),
                                                  "GBM trigger catalog": CatalogServer("https://heasarc.gsfc.nasa.gov/cgi-bin/vo/cone/coneGet.pl?table=fermigtrig&"),
                                                  "LLE catalog": CatalogServer("https://heasarc.gsfc.nasa.gov/cgi-bin/vo/cone/coneGet.pl?table=fermille&")
                                                  })

    Swift: InstrumentCatalog = InstrumentCatalog({"Swift GRB catalog": CatalogServer(
        "https://heasarc.gsfc.nasa.gov/cgi-bin/vo/cone/coneGet.pl?table=swiftgrb&")})
