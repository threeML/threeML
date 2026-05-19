from ..._lazy_exports import setup_lazy_exports

_EXPORTS = {
    "CastroLike": (".castro_like", []),
    "SherpaLike": (".sherpa_like", ["sherpa"]),
    "VERITASLike": (".veritas_like", ["ROOT"]),
}

setup_lazy_exports(globals(), _EXPORTS)
