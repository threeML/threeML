from ..._lazy_exports import setup_lazy_exports

_exports = {
    "download_GBM_daily_data": (".Fermi_GBM", []),
    "download_GBM_trigger_data": (".Fermi_GBM", []),
}

setup_lazy_exports(globals(), _exports)
