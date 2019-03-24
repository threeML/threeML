from scipy.special import gammaln
from numba import vectorize, int64, float64

@vectorize([float64(int64)], fastmath=True)
def logfactorial(n):

    return gammaln(n + 1)
