from threeML import *
from threeML.plugins.UnresolvedExtendedXYLike import UnresolvedExtendedXYLike
from astromodels.functions.functions_2D import Gaussian_on_sphere
import os

def get_signal():
    # Generate a test signal
    generator = Line() + Gaussian()

    generator.mu_2 = 5.0
    generator.sigma_2 = 0.32
    generator.F_2 = 70.4
    generator.a_1 = 40.0

    signal = generator(x)

    return signal


# A simple x

x = np.linspace(0, 10, 50)

# These datasets have been generated adding noise to the signal
gauss_signal = [
    50.7602612346,
    45.9932836018,
    32.7977610209,
    36.854638754,
    39.5900950043,
    41.9882356834,
    35.5464965039,
    47.7006711308,
    51.350490463,
    41.3574154897,
    43.5213662377,
    39.1763197352,
    39.817699622,
    30.515504494,
    33.5154124187,
    52.2808043531,
    37.3822864933,
    54.8713758496,
    31.501229516,
    44.3932720107,
    40.9919050981,
    46.6234446307,
    65.0001223876,
    78.9277394629,
    119.888320901,
    144.295807048,
    94.2679915254,
    68.3366002984,
    48.7021725122,
    43.8175069547,
    48.4100953701,
    53.4893430887,
    44.3520922656,
    37.3317011115,
    48.8614340877,
    45.8279746014,
    41.4841202405,
    54.4940719287,
    34.1601281994,
    38.0362535503,
    50.8319092017,
    45.0868214795,
    50.7982173405,
    59.7236796118,
    42.8220846239,
    47.978397568,
    59.6987029918,
    50.8856593966,
    55.5325981236,
    33.879756494,
]

gauss_sigma = [
    6.32455532034,
    6.34066886319,
    6.35674156037,
    6.37277372091,
    6.38876565,
    6.40471764899,
    6.4206300155,
    6.43650304347,
    6.45233702322,
    6.46813224153,
    6.48388898166,
    6.49960752347,
    6.51528814342,
    6.53093111468,
    6.54653670831,
    6.56210525885,
    6.57763959273,
    6.59320350524,
    6.60982256293,
    6.63810676386,
    6.74610996782,
    7.18007988042,
    8.31336897994,
    10.0117217614,
    11.3276025229,
    11.3366071004,
    10.0422516164,
    8.37451562539,
    7.27888168044,
    6.88089633636,
    6.80509804246,
    6.80755576825,
    6.82140432398,
    6.83628044824,
    6.85118795893,
    6.86606562443,
    6.88091118789,
    6.89572479208,
    6.9105066414,
    6.92525693917,
    6.93997588657,
    6.95466368265,
    6.96932052437,
    6.98394660662,
    6.99854212224,
    7.01310726208,
    7.027642215,
    7.04214716792,
    7.05662230584,
    7.07106781187,
]

poiss_sig = [
    44,
    43,
    38,
    25,
    51,
    37,
    46,
    47,
    55,
    36,
    40,
    32,
    46,
    37,
    44,
    42,
    50,
    48,
    52,
    47,
    39,
    55,
    80,
    93,
    123,
    135,
    96,
    74,
    43,
    49,
    43,
    51,
    27,
    32,
    35,
    42,
    43,
    49,
    38,
    43,
    59,
    54,
    50,
    40,
    50,
    57,
    55,
    47,
    38,
    64,
]


def test_UnresolvedExtendedXYLike_chi2():

    # Get fake data with Gaussian noise

    yerr = np.array(gauss_sigma)
    y = np.array(gauss_signal)

    # Fit

    xy = UnresolvedExtendedXYLike("test", x, y, yerr=yerr)

    fitfun = Line() + Gaussian()
    fitfun.F_2 = 60.0
    fitfun.mu_2 = 4.5

    res = xy.fit(fitfun)

    # Verify that the fit converged where it should have
    assert np.allclose(
        res[0]["value"].values,
        [40.20269202, 0.82896119,  62.80359114, 5.04080011, 0.27286713],
        rtol=0.05,
    )

    # test not setting yerr

    xy = UnresolvedExtendedXYLike("test", x, y)

    assert np.all(xy.yerr == np.ones_like(y))

    fitfun = Line() + Gaussian()
    fitfun.F_2 = 60.0
    fitfun.mu_2 = 4.5

    res = xy.fit(fitfun)


def test_UnresolvedExtendedXYLike_poisson():

    # Now Poisson case
    y = np.array(poiss_sig)

    xy = UnresolvedExtendedXYLike("test", x, y, poisson_data=True)

    fitfun = Line() + Gaussian()

    fitfun.F_2 = 60.0
    fitfun.F_2.bounds = (0, 200.0)
    fitfun.mu_2 = 5.0
    fitfun.b_1.bounds = (0.1, 5.0)
    fitfun.a_1.bounds = (0.1, 100.0)

    res = xy.fit(fitfun)

    # Verify that the fit converged where it should have

    # print res[0]['value']
    assert np.allclose(
        res[0]["value"], [40.344599, 0.783748,  71.560055, 4.989727, 0.330570], rtol=0.05
    )


def test_UnresolvedExtendedXYLike_assign_to_source():

    # Get fake data with Gaussian noise

    yerr = np.array(gauss_sigma)
    y = np.array(gauss_signal)

    # Fit

    xy = UnresolvedExtendedXYLike("test", x, y, yerr=yerr)

    xy.assign_to_source("exs1")

    fitfun = Line() + Gaussian()
    fitfun.F_2 = 60.0
    fitfun.mu_2 = 4.5

    fitfun2 = Line() + Gaussian()
    fitfun2.F_2 = 60.0
    fitfun2.mu_2 = 4.5

    shape = Gaussian_on_sphere()
    exs1 = ExtendedSource("exs1", spatial_shape=shape, spectral_shape=fitfun)
    pts2 = PointSource("pts2", ra=2.5, dec=3.2, spectral_shape=fitfun2)

    for parameter in list(fitfun2.parameters.values()):
        parameter.fix = True

    for parameter in list(shape.parameters.values()):
        parameter.fix = True

    model = Model(exs1, pts2)
    data = DataList(xy)

    jl = JointLikelihood(model, data)

    _ = jl.fit()

    predicted_parameters = np.array(
        [40.20269202, 0.82896119,  62.80359114, 5.04080011, 0.27286713]
    )

    assert np.allclose(
        [
            fitfun.a_1.value,
            fitfun.b_1.value,
            fitfun.F_2.value,
            fitfun.mu_2.value,
            fitfun.sigma_2.value,
        ],
        predicted_parameters,
        rtol=0.05,
    )

    # Test that the likelihood does not change by changing the parameters of the other source
    log_like_before = jl.minus_log_like_profile(*predicted_parameters)

    fitfun2.F_2 = 120.0

    log_like_after = jl.minus_log_like_profile(*predicted_parameters)

    assert log_like_before == log_like_after

    # Now test that if we do not assign a source, then the log likelihood value will change
    xy.assign_to_source(None)

    # Test that the likelihood this time changes by changing the parameters of the other source
    log_like_before = jl.minus_log_like_profile(*predicted_parameters)

    fitfun2.F_2 = 60.0

    log_like_after = jl.minus_log_like_profile(*predicted_parameters)

    assert log_like_before != log_like_after


def test_UnresolvedExtendedXYLike_dataframe():

    yerr = np.array(gauss_sigma)
    y = np.array(gauss_signal)

    # chi2 version

    xy = UnresolvedExtendedXYLike("test", x, y, yerr)

    df = xy.to_dataframe()

    # read back in dataframe

    new_xy = UnresolvedExtendedXYLike.from_dataframe("df", df)

    assert not xy.is_poisson

    # poisson version

    xy = UnresolvedExtendedXYLike("test", x, y, poisson_data=True)

    df = xy.to_dataframe()

    # read back in dataframe

    new_xy = UnresolvedExtendedXYLike.from_dataframe("df", df, poisson=True)

    assert xy.is_poisson


def test_UnresolvedExtendedXYLike_txt():

    yerr = np.array(gauss_sigma)
    y = np.array(gauss_signal)

    # chi2 version

    xy = UnresolvedExtendedXYLike("test", x, y, yerr)

    fname = "test_txt.txt"

    xy.to_txt(fname)

    # read back in txt file

    new_xy = UnresolvedExtendedXYLike.from_text_file("txt", fname)

    assert not xy.is_poisson

    # poisson version

    xy = UnresolvedExtendedXYLike("test", x, y, poisson_data=True)

    fname = "test_txt_poisson.txt"

    xy.to_txt(fname)

    # read back in txt file

    new_xy = UnresolvedExtendedXYLike.from_text_file("txt", fname)

    assert new_xy.is_poisson

    # Remove files
    os.remove("test_txt.txt")
    os.remove("test_txt_poisson.txt")


def test_UnresolvedExtendedxy_plot():
    # Get fake data with Gaussian noise

    yerr = np.array(gauss_sigma)
    y = np.array(gauss_signal)

    # Fit

    xy = UnresolvedExtendedXYLike("test", x, y, yerr)

    xy.plot()

    fitfun = Line() + Gaussian()
    fitfun.F_2 = 60.0
    fitfun.mu_2 = 4.5

    res = xy.fit(fitfun)

    xy.plot()
