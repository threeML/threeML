from threeML import *

# from threeML.utils.cartesian import cartesian
from threeML.utils.statistics.stats_tools import PoissonResiduals, Significance


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


def test_poisson_classes():

    net = 100
    Noff = 1000
    Non = Noff + net
    alpha = 1

    expected = alpha * Noff

    pr = PoissonResiduals(Non=Non, Noff=Noff, alpha=alpha)

    assert pr.net == Non - expected
    assert pr.expected == expected

    one_side = pr.significance_one_side()

    net = 0
    Noff = 1000
    Non = Noff + net
    alpha = 0.1

    expected = alpha * Noff

    pr = PoissonResiduals(Non=Non, Noff=Noff, alpha=alpha)

    assert pr.net == Non - expected
    assert pr.expected == expected

    one_side = pr.significance_one_side()

    sig = Significance(Non=Non, Noff=Noff)

    res = sig.known_background()
    res = sig.li_and_ma()
    res = sig.li_and_ma_equivalent_for_gaussian_background(1)


# def test_cartesian():
#     cart = cartesian(([1, 2, 3], [1, 2, 3]))
