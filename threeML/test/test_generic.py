import pytest

from threeML import *


def test_step_generator_setup():
    ra, dec = 0, 0
    name = "test"

    powerlaw = Powerlaw()

    line = Line()

    ps = PointSource(name, ra, dec, spectral_shape=powerlaw)

    model = Model(ps)

    # test with

    step = step_generator([1, 2, 3, 4, 5], powerlaw.K)

    step = step_generator([[1, 2], [3, 4]], powerlaw.K)
