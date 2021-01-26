from dataclasses import dataclass, field
from enum import Enum, Flag
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
from omegaconf import II, MISSING, SI, OmegaConf
from .plotting_structure import MPLCmap

class Sampler(Enum):
    emcee = "emcee"
    multinest = "multinest"
    zeus = "zeus"
    dynesty = "dynesty"
    ultranest = "ultranest"


_sampler_default = {'emcee': {'n_burnin': 1}}


class Optimizer(Enum):
    minuit = "minuit"
    scipy = "scipy"
    ROOT = "ROOT"


@dataclass
class BayesianDefault:
    default_sampler: Sampler = Sampler.emcee
    default_setup: Optional[Dict[str, Any]] = field(
        default_factory=lambda: {'n_burnin': 1})


@dataclass
class MLEDefault:
    default_minimizer: Optimizer = Optimizer.minuit
    default_algorithm: Optional[str] = None
    default_callback: Optional[str] = None
    contour_cmap: str = MPLCmap.Pastel1
    contour_background: str = 'white'
    contour_level_1: str = '#76E36C'
    contour_level_2: str = '#FF5239'
    contour_level_3: str = '#64CAFE'
    profile_color: str = 'k'

    profile_level_1: str = '#76E36C'
    profile_level_2: str = '#FF5239'
    profile_level_3: str = '#64CAFE'
