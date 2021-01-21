import numpy as np
import pytest
from astromodels import Gaussian, Line, Log_normal, Model, PointSource

from threeML.bayesian.bayesian_analysis import BayesianAnalysis
from threeML.classicMLE.joint_likelihood import JointLikelihood
from threeML.data_list import DataList
from threeML.plugins.UnbinnedPoissonLike import (EventObservation,
                                                 UnbinnedPoissonLike)

from .conftest import event_observation_contiguous, event_observation_split


def test_event_observation(event_observation_contiguous, event_observation_split):

    assert not event_observation_contiguous.is_multi_interval

    assert event_observation_split.is_multi_interval

    # test all exists
    for obs in [event_observation_split, event_observation_contiguous]:

        obs.exposure
        obs.start
        obs.stop
        obs.events

    assert isinstance(event_observation_contiguous.start, float)
    assert isinstance(event_observation_contiguous.stop, float)

    for a, b in zip(event_observation_split.start, event_observation_split.stop):

        assert a < b

    with pytest.raises(AssertionError):

        EventObservation([0, 1, 2, 3], exposure=1, start=10, stop=1)


def test_ubinned_poisson_full(event_observation_contiguous, event_observation_split):

    s = Line()

    ps = PointSource("s", 0, 0, spectral_shape=s)

    s.a.bounds = (0, None)
    s.a.value = .1
    s.b.value = .1

    s.a.prior = Log_normal(mu=np.log(10), sigma=1)
    s.b.prior = Gaussian(mu=0, sigma=1)

    m = Model(ps)

    ######
    ######
    ######

    
    ub1 = UnbinnedPoissonLike("test", observation=event_observation_contiguous)

    jl = JointLikelihood(m, DataList(ub1))

    jl.fit(quiet=True)

    np.testing.assert_allclose([s.a.value, s.b.value], [6.11, 1.45], rtol=.5)

    ba = BayesianAnalysis(m, DataList(ub1))

    ba.set_sampler("emcee")

    ba.sampler.setup(n_burn_in=100, n_walkers=20, n_iterations=500)

    ba.sample(quiet=True)

    ba.restore_median_fit()

    np.testing.assert_allclose([s.a.value, s.b.value], [6.11, 1.45], rtol=.5)

    ######
    ######
    ######

    ub2 = UnbinnedPoissonLike("test", observation=event_observation_split)

    jl = JointLikelihood(m, DataList(ub2))

    jl.fit(quiet=True)

    np.testing.assert_allclose([s.a.value, s.b.value], [2., .2], rtol=.5)

    ba = BayesianAnalysis(m, DataList(ub2))

    ba.set_sampler("emcee")

    ba.sampler.setup(n_burn_in=100, n_walkers=20, n_iterations=500)

    ba.sample(quiet=True)

    ba.restore_median_fit()

    np.testing.assert_allclose([s.a.value, s.b.value], [2., .2], rtol=.5)
