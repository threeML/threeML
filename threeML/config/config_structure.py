import logging
from dataclasses import dataclass
from enum import IntEnum

from .catalog_structure import Catalogs, PublicDataServer
from .fitting_structure import BayesianDefault, MLEDefault
from .plotting_structure import GenericPlotting, ModelPlotting
from .plugin_structure import Plugins, TimeSeries
from .point_source_structure import PointSourceDefaults

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
    startup_warnings: bool = 'on'


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
    plotting: GenericPlotting = GenericPlotting()
    model_plot: ModelPlotting = ModelPlotting()
    point_source: PointSourceDefaults = PointSourceDefaults()
    
    LAT: PublicDataServer = PublicDataServer(public_ftp_location="ftp://heasarc.nasa.gov/fermi/data",
                                             public_http_location="https://heasarc.gsfc.nasa.gov/FTP/fermi/data/lat",
                                             query_form="https://fermi.gsfc.nasa.gov/cgi-bin/ssc/LAT/LATDataQuery.cgi")
    GBM: PublicDataServer = PublicDataServer(public_ftp_location="ftp://heasarc.nasa.gov/fermi/data",
                                             public_http_location="https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm")
    catalogs: Catalogs = Catalogs()
