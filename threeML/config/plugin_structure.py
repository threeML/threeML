from dataclasses import dataclass, field
from enum import Enum, Flag
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
from omegaconf import II, MISSING, SI, OmegaConf

from .plotting_structure import BinnedSpectrumPlot, DataHistPlot, MPLCmap


@dataclass
class OGIP:
    fit_plot: BinnedSpectrumPlot = BinnedSpectrumPlot()
    data_plot: DataHistPlot = DataHistPlot()


@dataclass
class Photo:
    fit_plot: BinnedSpectrumPlot = BinnedSpectrumPlot()


@dataclass
class Plugins:
    ogip: OGIP = OGIP()
    photo: Photo = Photo()


@dataclass
class TimeSeries:
    light_curve_color: str = "#34495E"
    selection_color: str = "#85929E"
    background_color: str = "#C0392B"
    background_selection_color: str = "#E74C3C"
