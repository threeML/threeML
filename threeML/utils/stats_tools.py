import numpy as np


def li_and_ma(total, background, alpha=1.):
    """"Li and Ma (1983) signal to noise significance"""

    a = total / (total + background)
    b = background / (total + background)
    S = np.sqrt(2) * np.sqrt((total * np.log(((1. + alpha) / alpha) * a) + background * np.log((1 + alpha) * b)))

    return S
