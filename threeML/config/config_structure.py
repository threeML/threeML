import logging
from dataclasses import dataclass, field
from enum import Enum, Flag, IntEnum
from typing import Any, Dict, List, Optional

from omegaconf import II, MISSING, SI, OmegaConf

from .catalog_structure import Catalogs, PublicDataServer
from .fitting_structure import BayesianDefault, MLEDefault
from .plotting_structure import ModelPlotting
from .plugin_structure import Plugins, TimeSeries


# logging
class LoggingLevel(IntEnum):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


@dataclass
class Logging:

    path: str = "~/.threeml/log"
    developer: bool = 'off'
    usr: bool = 'on'
    console: bool = 'on'
    level: LoggingLevel = LoggingLevel.INFO
    startup_warning: bool = 'on'


@dataclass
class Parallel:
    profile_name: str = "default"
    use_parallel: bool = False


@dataclass
class Interface:
    progress_bars: bool = 'on'
    multi_progress_color: bool = 'on'
    multi_progress_cmap: str = "viridis"
    progress_bar_color: str = "#9C04FF"


@dataclass
class Config:
    logging: Logging = Logging()
    parallel: Parallel = Parallel()
    interface: Interface = Interface()
    plugins: Plugins = Plugins()
    time_series: TimeSeries = TimeSeries()
    mle: MLEDefault = MLEDefault()
    bayesian: BayesianDefault = BayesianDefault()

    model_plot: ModelPlotting = ModelPlotting()

    LAT: PublicDataServer = PublicDataServer(public_ftp_Location="ftp://heasarc.nasa.gov/fermi/data",
                                             public_http_Location="https://heasarc.gsfc.nasa.gov/FTP/fermi/data/lat",
                                             query_form="https://fermi.gsfc.nasa.gov/cgi-bin/ssc/LAT/LATDataQuery.cgi")
    GBM: PublicDataServer = PublicDataServer(public_ftp_Location="ftp://heasarc.nasa.gov/fermi/data",
                                             public_http_Location="https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm")
    catalogs: Catalogs = Catalogs()
