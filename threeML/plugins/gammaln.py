import numpy
from scipy.special import gammaln

def _gammaln(x):
    """Vectorized calculation of ln(abs(gamma(array))) across a Numpy array.

    Numpy does not have a native implementation of gammaln.
    U{Scipy does <http://docs.scipy.org/doc/scipy/reference/generated/scipy.special.gammaln.html>},
    but that would introduce a dependency.
    """
    
    array       = numpy.asarray(x)
    gammaln_cof = [76.18009173, -86.50532033, 24.01409822, -1.231739516e0, 0.120858003e-2, -0.536382e-5]
    gammaln_stp = 2.50662827465
    x = numpy.array(array - 1.0)
    tmp = x + 5.5
    tmp = ((x + 0.5)*numpy.log(tmp)) - tmp
    ser = numpy.ones(array.shape[0], dtype=numpy.dtype(float))
    for cof in gammaln_cof:
        x += 1.0
        ser += cof/x
    return (tmp + numpy.log(gammaln_stp*ser))

def logfactorial(n):
    
    return gammaln( n + 1)
