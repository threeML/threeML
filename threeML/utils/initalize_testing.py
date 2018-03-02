import numpy as np


def initialize_testing():


    np.random.seed(0)
    np.seterr(over='ignore',
              under='ignore',
              divide='ignore',
              invalid='ignore'
              )

