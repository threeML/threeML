from threeML._lazy_exports import setup_lazy_exports

_EXPORTS = {
    "TimeSeriesBuilder": (".time_series_builder", []),
    "TransientLATDataBuilder": (".fermi", []),
}

setup_lazy_exports(globals(), _EXPORTS)
