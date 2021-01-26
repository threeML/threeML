from dataclasses import dataclass
from enum import Enum, Flag
from typing import Any, Dict, List, Optional

from omegaconf import II, MISSING, SI, OmegaConf


class Switch(Flag):
    on = True
    off = False
    ON = True
    OFF = False
    On = True
    Off = False


# logging

@dataclass
class Logging:

    path: str = "~/.threeml/log"
    developer: Switch = Switch.off
    usr: Switch = Switch.on
    console: Switch = Switch.on


@dataclass
class Parallel:
    profile_name: str = "default"
    use_parallel: bool = False


@dataclass
class Interface:
    progress_bars: Switch = Switch.on
    show_progress_bars: bool = II("progress_bars")
    multi_progress_color: Switch = Switch.on
    multi_progress_cmap: str = "viridis"
    progress_bar_color: str = "#9C04FF"


@dataclass(frozen=True)
class PublicDataServer:
    public_ftp_Location: Optional[str] = None
    public_http_Location: str = MISSING
    query_form: Optional[str] = None


@dataclass(frozen=True)
class CatalogServer:
    url: str = MISSING


@dataclass
class InstrumentCatalogs:
    catalogs: Dict[str, CatalogServer]


@dataclass
class BinnedSpectrumPlot:
    data_cmap: str = Set1
    model_cmap: str = Set1
    step: bool = False


@dataclass
class LightCurve:
    light_curve_color: str = "#34495E"
    selection_color: str = "#85929E"
    background_color: str = "#C0392B"
    background_selection_color: str = "#E74C3C"


class Sampler(Enum):
    emcee = 1
    pymultinest = 2
    zeus = 3
    dynesty = 4
    ultranest = 5


_sampler_default = (



)

    
class Optimizer(Enum):
    minuit = 1
    
    


@dataclass
class BayesianDefault:
    default_sampler: Sampler.emcee
