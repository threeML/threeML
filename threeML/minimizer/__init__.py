from .._lazy_exports import setup_lazy_exports

_EXPORTS = {
    "GridMinimizer": (".grid_minimizer", []),
    "LocalMinimization": (".minimization", []),
    "GlobalMinimization": (".minimization", []),
    "MinuitMinimizer": (".minuit_minimizer", []),
    "PAGMOMinimizer": (".pagmo_minimizer", []),
    "ROOTMinimizer": (".root_minimizer", []),
    "ScipyMinimizer": (".scipy_minimizer", []),
}

setup_lazy_exports(globals(), _EXPORTS)
