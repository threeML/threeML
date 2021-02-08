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
    dynesty_nested = "dynesty_nested"
    dynesty_dynamic = "dynesty_dynamic"
    ultranest = "ultranest"
    autoemcee = "autoemcee"


_sampler_default = {'emcee': {'n_burnin': 1}}


class Optimizer(Enum):
    minuit = "minuit"
    scipy = "scipy"
    ROOT = "ROOT"


@dataclass
class BayesianDefault:
    default_sampler: Sampler = Sampler.emcee
    default_setup: Optional[Dict[str, Any]] = field(
        default_factory=lambda: {'n_burnin': 250,
                            'n_iterations': 500,
                            "n_walkers": 50,
                            "seed": 5123})


@dataclass
class MLEDefault:
    default_minimizer: Optimizer = Optimizer.minuit
    default_minimizer_algorithm: Optional[str] = None
    default_minimizer_callback: Optional[str] = None
    contour_cmap: MPLCmap = MPLCmap.Pastel1
    contour_background: str = 'white'
    contour_level_1: str = '#ffa372'
    contour_level_2: str = '#ed6663'
    contour_level_3: str = '#0f4c81'
    profile_color: str = 'k'

    profile_level_1: str = '#ffa372'
    profile_level_2: str = '#ed6663'
    profile_level_3: str = '#0f4c81'
