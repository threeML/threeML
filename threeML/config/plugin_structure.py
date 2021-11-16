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
    response_cmap: MPLCmap = MPLCmap.viridis
    response_zero_color: str = "k"


@dataclass
class Photo:
    fit_plot: BinnedSpectrumPlot = BinnedSpectrumPlot()


@dataclass
class Plugins:
    ogip: OGIP = OGIP()
    photo: Photo = Photo()


@dataclass
class TimeSeriesFit:
    fit_poly: bool = True
    unbinned: bool = False
    bayes: bool = False


@dataclass
class TimeSeries:
    light_curve_color: str = "#05716c"
    selection_color: str = "#1fbfb8"
    background_color: str = "#C0392B"
    background_selection_color: str = "#E74C3C"
    fit: TimeSeriesFit = TimeSeriesFit()
