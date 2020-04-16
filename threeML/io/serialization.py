from future import standard_library

standard_library.install_aliases()
from threeML.classicMLE.joint_likelihood import JointLikelihood
from threeML.bayesian.bayesian_analysis import BayesianAnalysis

__all__ = []

# copyreg is called copy_reg in python2
try:

    import copyreg  # py3

except ImportError:

    import copyreg as copyreg  # py2


# Serialization for JointLikelihood object
def pickle_joint_likelihood(jl):

    return JointLikelihood, (jl.likelihood_model, jl.data_list)


copyreg.pickle(JointLikelihood, pickle_joint_likelihood)

# Serialization for BayesianAnalysis object
def pickle_bayesian_analysis(bs):

    return BayesianAnalysis, (bs.likelihood_model, bs.data_list)


copyreg.pickle(BayesianAnalysis, pickle_bayesian_analysis)
