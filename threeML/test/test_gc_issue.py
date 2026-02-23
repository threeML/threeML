import gc
import os
import weakref

import numpy as np
import pytest
from astromodels import (
    Disk_on_sphere,
    ExtendedSource,
    Gaussian,
    Model,
    Uniform_prior,
)

from threeML import BayesianAnalysis, DataList, JointLikelihood, PluginPrototype


class MemoryHeavyPlugin(PluginPrototype):
    def __init__(self, name, mb=32):
        super().__init__(name, {})
        # Large array used to detect retention/leaks across iterations.
        n_float64 = mb * 1024 * 1024 // 8
        self.bigarray = np.ones(n_float64, dtype=np.float64)

    def set_model(self, model):
        self._model = model

    def get_log_like(self):
        f_value = list(self._model.extended_sources.values())[
            0
        ].spectrum.main.shape.F.value
        return -((f_value - 10.0) ** 2)

    def inner_fit(self):
        return self.get_log_like()


def _make_model(for_bayesian=False):
    src = ExtendedSource(
        "src",
        spectral_shape=Gaussian(),
        spatial_shape=Disk_on_sphere(),
    )

    src.spectrum.main.shape.F.value = 3e-5
    src.spectrum.main.shape.F.min_value = 0
    src.spectrum.main.shape.F.max_value = 100
    src.spectrum.main.shape.F.free = True

    if for_bayesian:
        src.spectrum.main.shape.F.prior = Uniform_prior(lower_bound=0, upper_bound=100)

    src.spectrum.main.shape.mu.free = False
    src.spectrum.main.shape.sigma.free = False
    src.spatial_shape.lon0.free = False
    src.spatial_shape.lat0.free = False
    src.spatial_shape.radius.free = False

    return Model(src)


def _rss_mb():
    psutil = pytest.importorskip("psutil")
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2


def test_joint_likelihood_plugins_are_collected():
    model = _make_model(for_bayesian=False)

    for i in range(6):
        plugin = MemoryHeavyPlugin(f"mle_{i}")
        plugin_ref = weakref.ref(plugin)

        data = DataList(plugin)
        like = JointLikelihood(model, data, verbose=False, record=False)
        like.fit(quiet=True, compute_covariance=False)

        del like
        del data
        del plugin
        gc.collect()

        assert plugin_ref() is None


def test_bayesian_plugins_are_collected():
    pytest.importorskip("emcee")

    model = _make_model(for_bayesian=True)

    for i in range(4):
        plugin = MemoryHeavyPlugin(f"bayes_{i}")
        plugin_ref = weakref.ref(plugin)

        data = DataList(plugin)
        bayes = BayesianAnalysis(model, data)
        bayes.set_sampler("emcee")
        bayes.sampler.setup(n_iterations=8, n_burn_in=8, n_walkers=10, seed=1234)
        bayes.sample(quiet=True)

        del bayes
        del data
        del plugin
        gc.collect()

        assert plugin_ref() is None


def test_joint_likelihood_rss_does_not_grow_linearly():
    model = _make_model(for_bayesian=False)
    baseline = []
    chunk_mb = 32

    for i in range(6):
        gc.collect()
        baseline.append(_rss_mb())

        plugin = MemoryHeavyPlugin(f"mle_rss_{i}", mb=chunk_mb)
        data = DataList(plugin)
        like = JointLikelihood(model, data, verbose=False, record=False)
        like.fit(quiet=True, compute_covariance=False)

        del like
        del data
        del plugin

    drift_mb = baseline[-1] - baseline[0]
    # Leaking would be ~N*chunk_mb. Allow broad allocator noise.
    assert drift_mb < 3.0 * chunk_mb


def test_bayesian_rss_does_not_grow_linearly():
    pytest.importorskip("emcee")

    model = _make_model(for_bayesian=True)
    baseline = []
    chunk_mb = 32

    for i in range(4):
        gc.collect()
        baseline.append(_rss_mb())

        plugin = MemoryHeavyPlugin(f"bayes_rss_{i}", mb=chunk_mb)
        data = DataList(plugin)
        bayes = BayesianAnalysis(model, data)
        bayes.set_sampler("emcee")
        bayes.sampler.setup(n_iterations=8, n_burn_in=8, n_walkers=10, seed=1234)
        bayes.sample(quiet=True)

        del bayes
        del data
        del plugin

    drift_mb = baseline[-1] - baseline[0]
    assert drift_mb < 3.0 * chunk_mb
