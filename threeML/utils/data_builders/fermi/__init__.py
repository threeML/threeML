from ...._lazy_exports import setup_lazy_exports

_EXPORTS = {
    "GBMTTEFile": (".gbm_data", []),
    "GBMCdata": (".gbm_data", []),
    "LLEFile": (".lat_data", []),
    "TransientLATDataBuilder": (".lat_transient_builder", []),
    "LATObservation": (".lat_transient_builder", []),
}

setup_lazy_exports(globals(), _EXPORTS)
