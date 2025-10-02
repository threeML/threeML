from dataclasses import dataclass, field

from .plotting_structure import (
    BinnedSpectrumPlot,
    DataHistPlot,
    FermiSpectrumPlot,
    MPLCmap,
)


@dataclass
class OGIP:
    fit_plot: BinnedSpectrumPlot = field(default_factory=lambda: BinnedSpectrumPlot())
    data_plot: DataHistPlot = field(default_factory=lambda: DataHistPlot())
    response_cmap: MPLCmap = MPLCmap.viridis
    response_zero_color: str = "k"


@dataclass
class Fermipy:
    fit_plot: FermiSpectrumPlot = field(default_factory=lambda: FermiSpectrumPlot())


#    data_plot: DataHistPlot = DataHistPlot()


@dataclass
class Photo:
    fit_plot: BinnedSpectrumPlot = field(default_factory=lambda: BinnedSpectrumPlot())


@dataclass
class Plugins:
    ogip: OGIP = field(default_factory=lambda: OGIP())
    photo: Photo = field(default_factory=lambda: Photo())
    fermipy: Fermipy = field(default_factory=lambda: Fermipy())


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
    fit: TimeSeriesFit = field(default_factory=lambda: TimeSeriesFit())
