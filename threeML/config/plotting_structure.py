from dataclasses import dataclass, field
from enum import Enum, Flag
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
from omegaconf import II, MISSING, SI, OmegaConf

# type checking matplotlib colormaps
MPLCmap = Enum("MPLCmap", zip(plt.colormaps(), plt.colormaps()))


@dataclass
class BinnedSpectrumPlot:
    data_cmap: MPLCmap = MPLCmap.Set1
    model_cmap: MPLCmap = MPLCmap.Set1
    step: bool = False


@dataclass
class DataHistPlot:
    counts_color: str = '#31FE6F'
    background_color: str = '#377eb8'



@dataclass
class PlotStyle:
    linestyle: Optional[str] = '-'
    lineswidth: Optional[float] = 1.7


@dataclass
class ContourStyle:
    alpha: float = 0.4


class LegendLoc(Enum):
    best = 'best'
    lower_left = 'lower left'
    lower_right = 'lower right'
    upper_left = 'upper left'
    upper_right = 'upper right'

@dataclass
class LegendStyle:
    loc: LegendLoc = LegendLoc.best
    fancybox: bool = True
    shadow: bool = True


@dataclass
class PointSourcePlot:

    fit_cmap: MPLCmap = MPLCmap.Set1
    contour_cmap: MPLCmap = MPLCmap.Set1
    bayes_cmap: MPLCmap = MPLCmap.Set1
    plot_style: PlotStyle = PlotStyle()
    contour_style: ContourStyle = ContourStyle()
    legend_style: LegendStyle = LegendStyle()


@dataclass
class ModelPlotting:

    point_source_plot: PointSourcePlot = PointSourcePlot()
