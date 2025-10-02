from math import lgamma

from numba import float64, int64, vectorize


@vectorize([float64(int64)], fastmath=True)
def logfactorial(n):
    return lgamma(n + 1)
