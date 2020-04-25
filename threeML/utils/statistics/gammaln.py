from math import lgamma
from numba import vectorize, int64, float64

@vectorize([float64(int64)], fastmath=True)
def logfactorial(n):

    return lgamma(n + 1)
