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
    linewidth: Optional[float] = 1.7


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
    show_legend: bool = True
    legend_style: LegendStyle = LegendStyle()
    flux_unit: str = "1/(keV s cm2)"
    emin: float = 10.
    emax: float = 1e4
    num_ene: int = 100
    ene_unit: str = "keV"


@dataclass
class ResidualPlot:
    linewidth: float = 1
    marker: str = "."
    size: float = 3


@dataclass
class GenericPlotting:

    residual_plot: ResidualPlot = ResidualPlot()


@dataclass
class ModelPlotting:

    point_source_plot: PointSourcePlot = PointSourcePlot()
