from dataclasses import dataclass
from enum import Flag
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
