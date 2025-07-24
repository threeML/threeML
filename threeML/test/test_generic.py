from astromodels.functions.functions_1D.powerlaws import Powerlaw

from threeML.utils.statistics.stats_tools import PoissonResiduals, Significance
from threeML.utils.step_parameter_generator import step_generator


def test_step_generator_setup():
    powerlaw = Powerlaw()

    _ = step_generator([1, 2, 3, 4, 5], powerlaw.K)

    _ = step_generator([[1, 2], [3, 4]], powerlaw.K)


def test_poisson_classes():
    net = 100
    Noff = 1000
    Non = Noff + net
    alpha = 1

    expected = alpha * Noff

    pr = PoissonResiduals(Non=Non, Noff=Noff, alpha=alpha)

    assert pr.net == Non - expected
    assert pr.expected == expected

    _ = pr.significance_one_side()

    net = 0
    Noff = 1000
    Non = Noff + net
    alpha = 0.1

    expected = alpha * Noff

    pr = PoissonResiduals(Non=Non, Noff=Noff, alpha=alpha)

    assert pr.net == Non - expected
    assert pr.expected == expected

    _ = pr.significance_one_side()

    sig = Significance(Non=Non, Noff=Noff)

    _ = sig.known_background()
    _ = sig.li_and_ma()
    _ = sig.li_and_ma_equivalent_for_gaussian_background(1)


# def test_cartesian():
#     cart = cartesian(([1, 2, 3], [1, 2, 3]))
