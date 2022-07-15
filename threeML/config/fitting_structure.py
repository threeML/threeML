from dataclasses import dataclass, field
from enum import Enum, Flag
from typing import Any, Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from omegaconf import II, MISSING, SI, OmegaConf

from .plotting_structure import CornerStyle, MPLCmap


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

    use_median_fit: bool = False

    default_sampler: Sampler = Sampler.emcee

    emcee_setup: Optional[Dict[str, Any]] = field(
        default_factory=lambda: {'n_burnin': None,
                            'n_iterations': 500,
                            "n_walkers": 50,
                            "seed": 5123})
    
    multinest_setup: Optional[Dict[str, Any]] = field(
        default_factory=lambda: {'n_live_points': 400,
                            'chain_name': "chains/fit-",
                            "resume": False,
                            "importance_nested_sampling": False,
                            "auto_clean": False,


                            })

    ultranest_setup: Optional[Dict[str, Any]] = field(
        default_factory=lambda: { "min_num_live_points":400,
                             "dlogz":0.5,
                             "dKL": 0.5,
                             "frac_remain": 0.01,
                             "Lepsilon": 0.001,
                             "min_ess": 400,
                             "update_interval_volume_fraction":0.8,
                             "cluster_num_live_points":40,
                             "use_mlfriends": True,
                             "resume": 'overwrite' }
    )

    zeus_setup: Optional[Dict[str, Any]] = field(
        default_factory=lambda: {'n_burnin': None,
                            'n_iterations': 500,
                            "n_walkers": 50,
                            "seed": 5123})


    
    dynesty_nested_setup: Optional[Dict[str, Any]] = field(
        default_factory=lambda: { "n_live_points": 400,
                             "maxiter": None,
                             "maxcall": None,
                             "dlogz": None,
                             "logl_max": np.inf,
                             "n_effective": None,
                             "add_live": True,
                             "print_func": None,
                             "save_bounds":True,
                             "bound":"multi",
                             "wrapped_params": None,
                             "sample": "auto",
                             "periodic": None,
                             "reflective": None,
                             "update_interval": None,
                             "first_update": None,
                             "npdim": None,
                             "rstate": None,
                             "use_pool": None,
                             "live_points": None,
                             "logl_args": None,
                             "logl_kwargs": None,
                             "ptform_args": None,
                             "ptform_kwargs": None,
                             "gradient": None,
                             "grad_args": None,
                             "grad_kwargs": None,
                             "compute_jac": False,
                             "enlarge": None,
                             "bootstrap": 0,
                             "vol_dec": 0.5,
                             "vol_check": 2.0,
                             "walks": 25,
                             "facc": 0.5,
                             "slices": 5,
                             "fmove": 0.9,
                             "max_move": 100,
                             "update_func": None,
                             
                             
        })

    dynesty_dynmaic_setup: Optional[Dict[str, Any]] = field(
        default_factory=lambda: {
            "nlive_init": 500,
            "maxiter_init": None,
            "maxcall_init": None,
        "dlogz_init": 0.01,
            "logl_max_init": np.inf,
            "n_effective_init": np.inf,
            "nlive_batch": 500,
            "wt_function": None,
            "wt_kwargs": None,
            "maxiter_batch": None,
            "maxcall_batch": None,
            "maxiter": None,
            "maxcall": None,
            "maxbatch": None,
            "n_effective": np.inf,
            "stop_function": None,
            "stop_kwargs": None,
            "use_stop": True,
            "save_bounds": True,
            "print_func": None,
            "live_points": None,
            "bound":"multi",
            "wrapped_params": None,
            "sample":"auto",
            "periodic": None,
            "reflective": None,
            "update_interval": None,
            "first_update": None,
            "npdim": None,
            "rstate": None,
            "use_pool": None,
            "logl_args": None,
            "logl_kwargs": None,
            "ptform_args": None,
            "ptform_kwargs": None,
            "gradient": None,
            "grad_args": None,
            "grad_kwargs": None,
            "compute_jac": False,
            "enlarge": None,
            "bootstrap": 0,
            "vol_dec": 0.5,
            "vol_check": 2.0,
            "walks": 25,
            "facc": 0.5,
            "slices": 5,
            "fmove": 0.9,
            "max_move": 100,
            "update_func": None,


        })


    
    corner_style: CornerStyle =CornerStyle()
    

    
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
