from threeML._lazy_exports import setup_lazy_exports

_EXPORTS = {
    "DispersionSpectrumLike": (".dispersionspectrum_like", []),
    "SpectrumLike": (".spectrum_like", []),
    "FermiLATLike": (
        ".fermilat_like",
        ["GtApp", "GtBurst", "UnbinnedAnalysis", "BinnedAnalysis", "pyLikelihood"],
    ),
    "FermipyLike": (".fermipy_like", ["fermipy"]),
    "PhotometryLike": (".photometry_like", ["speclite"]),
    "OGIPLike": (".ogip_like", []),
    "SwiftXRTLike": (".swiftxrt_like", []),
    "UnbinnedPoissonLike": (".unbinnedpoisson_like", []),
    "EventObservation": (".unbinnedpoisson_like", []),
    "UnresolvedExtendedXYLike": (".unresolvedextendedxy_like", []),
    "XYLike": (".xy_like", []),
}

setup_lazy_exports(globals(), _EXPORTS)
