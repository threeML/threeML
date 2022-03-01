from dataclasses import dataclass, field
from enum import Enum, Flag
from typing import Any, Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import II, MISSING, SI, OmegaConf

# type checking matplotlib colormaps
MPLCmap = Enum("MPLCmap", zip(plt.colormaps(), plt.colormaps()))


@dataclass
class BinnedSpectrumPlot:
    data_cmap: MPLCmap = MPLCmap.Set1
    model_cmap: MPLCmap = MPLCmap.Set1
    background_cmap: MPLCmap = MPLCmap.Set1
    n_colors: int = 5
    step: bool = False
    show_legend: bool = True
    show_residuals: bool = True
    data_color: Optional[str] = None
    model_color: Optional[str] = None
    background_color: Optional[str] = None
    show_background: bool = False
    data_mpl_kwargs: Optional[Dict[str, Any]] = None
    model_mpl_kwargs: Optional[Dict[str, Any]] = None
    background_mpl_kwargs: Optional[Dict[str, Any]] = None


@dataclass
class DataHistPlot:
    counts_color: str = "#500472"
    background_color: str = "#79cbb8"
    warn_channels_color: str = "#C79BFE"
    bad_channels_color: str = "#FE3131"
    masked_channels_color: str = "#566573"


@dataclass
class PlotStyle:
    linestyle: Optional[str] = '-'
    linewidth: Optional[float] = 1.7


@dataclass
class ContourStyle:
    alpha: float = 0.4


@dataclass
class CornerStyle:
    show_titles: bool = True
    smooth: float = 0.9
    title_fmt: str = ".2g"
    bins: int = 25
    quantiles: List[float] =  field(default_factory= lambda:[0.16, 0.50, 0.84])
    fill_contours: bool = True
    cmap: MPLCmap = MPLCmap.viridis
    extremes: str = "white"
    contourf_kwargs:  Dict[str, Any] = field(
        default_factory=lambda: {"colors": None, "extend": "both"})
    levels: List[float] = field(default_factory= lambda:[0.99, 0.865,0.393])

    
# class LegendLoc(Enum):
#     best = 'best'
#     lower_left = 'lower left'
#     lower_right = 'lower right'
#     upper_left = 'upper left'
#     upper_right = 'upper right'


@dataclass
class LegendStyle:
    loc: str = "best"
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
    legend_font_size: float = 6.94

@dataclass
class GenericPlotting:

    mplstyle: str = "threeml.mplstyle"
    residual_plot: ResidualPlot = ResidualPlot()


@dataclass
class ModelPlotting:

    point_source_plot: PointSourcePlot = PointSourcePlot()
