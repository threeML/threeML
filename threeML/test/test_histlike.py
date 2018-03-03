import numpy as np
from threeML import *
from threeML.utils.histogram import Histogram

from threeML.utils.initalize_testing import initialize_testing


initialize_testing()



def test_hist_like():




    rnum = np.random.normal(loc=0., scale=1., size=10000)

    hrnum = np.histogram(rnum, bins=100, normed=False)

    hh = Histogram.from_numpy_histogram(hrnum, is_poisson=True)

    hlike = HistLike('hist', hh)

    data_list = DataList(hlike)

    normal = Gaussian()

    res, lh = hlike.fit(normal)

    norm = res['value']['source.spectrum.main.Gaussian.F']
    mu = res['value']['source.spectrum.main.Gaussian.mu']
    sigma = res['value']['source.spectrum.main.Gaussian.sigma']

    # assert is_within_tolerance(1E6,norm,relative_tolerance=1E2)
    # assert is_within_tolerance(1,mu,relative_tolerance=.5)
    # assert is_within_tolerance(1,sigma,relative_tolerance=.01)

    # ps = PointSource('source',0,0,spectral_shape=normal)
    #
    # model = Model(ps)
    #
    # jl = JointLikelihood(model,data_list=data_list)
    #
    # display_histogram_fit(jl)
    #
    # hh = Histogram.from_numpy_histogram(hrnum, errors=np.ones(100))
    #
    # hlike = HistLike('hist', hh)
    #
    # hlike.fit(normal)
