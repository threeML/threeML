from .._lazy_exports import setup_lazy_exports

_EXPORTS = {
    "FermiGBMBurstCatalog": (".FermiGBM", []),
    "FermiGBMTriggerCatalog": (".FermiGBM", []),
    "FermiLATSourceCatalog": (".FermiLAT", []),
    "FermiPySourceCatalog": (".FermiLAT", ["fermipy"]),
    "FermiLLEBurstCatalog": (".FermiLLE", []),
    "SwiftGRBCatalog": (".Swift", []),
    "VirtualObservatoryCatalog": (".VirtualObservatoryCatalog", []),
    "ConeSearchFailed": (".VirtualObservatoryCatalog", []),
}

setup_lazy_exports(globals(), _EXPORTS)
