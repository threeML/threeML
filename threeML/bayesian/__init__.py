from threeML._lazy_exports import setup_lazy_exports


_EXPORTS = {
    "BayesianAnalysis": (".bayesian_analysis", []),
    "AutoEmceeSampler": (".autoemcee_sampler", ["autoemcee"]),
    "DynestyPool": (".dynesty_sampler", ["dynesty"]),
    "DynestyNestedSampler": (".dynesty_sampler", ["dynesty"]),
    "DynestyDynamicSampler": (".dynesty_sampler", ["dynesty"]),
    "EmceeSampler": (".emcee_sampler", ["emcee"]),
    "MultiNestSampler": (".multinest_sampler", ["pymultinest"]),
    "NautilusSampler": (".nautilus_sampler", ["nautilus"]),
    "SamplerBase": (".sampler_base", []),
}
setup_lazy_exports(globals(), _EXPORTS)
