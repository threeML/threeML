from dataclasses import dataclass, field
from typing import Dict, Optional

from omegaconf import MISSING

heasarc_url = "https://heasarc.gsfc.nasa.gov/cgi-bin/vo/cone/coneGet.pl?table"


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
    Fermi: InstrumentCatalog = field(
        default_factory=lambda: InstrumentCatalog(
            {
                "LAT FGL": CatalogServer(heasarc_url + "=fermilpsc&"),
                "GBM burst catalog": CatalogServer(heasarc_url + "=fermigbrst&"),
                "GBM trigger catalog": CatalogServer(heasarc_url + "=fermigtrig&"),
                "LLE catalog": CatalogServer(heasarc_url + "=fermille&"),
            }
        )
    )

    Swift: InstrumentCatalog = field(
        default_factory=lambda: InstrumentCatalog(
            {"Swift GRB catalog": CatalogServer(heasarc_url + "=swiftgrb&")}
        )
    )
