from .._lazy_exports import setup_lazy_exports

_EXPORTS = {
    "GoodnessOfFit": (".goodness_of_fit", []),
    "JointLikelihood": (".joint_likelihood", []),
    "JointLikelihoodSet": (".joint_likelihood_set", []),
    "JointLikelihoodSetAnalyzer": (".joint_likelihood_set", []),
    "LikelihoodRatioTest": (".likelihood_ratio_test", []),
}

setup_lazy_exports(globals(), _EXPORTS)
