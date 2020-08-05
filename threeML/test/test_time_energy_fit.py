from builtins import map
import pytest

from astromodels import *
from threeML.plugins.XYLike import XYLike
from threeML.data_list import DataList
from threeML.classicMLE.joint_likelihood import JointLikelihood


def test_energy_time_fit():

    # Let's generate our dataset of 4 spectra with a normalization that follows
    # a powerlaw in time

    def generate_one(K):
        # Let's generate some data with y = Powerlaw(x)

        gen_function = Powerlaw()
        gen_function.K = K

        # Generate a dataset using the power law, and a
        # constant 30% error

        x = np.logspace(0, 2, 50)

        xyl_generator = XYLike.from_function(
            "sim_data", function=gen_function, x=x, yerr=0.3 * gen_function(x)
        )

        y = xyl_generator.y
        y_err = xyl_generator.yerr

        # xyl = XYLike("data", x, y, y_err)

        # xyl.plot(x_scale='log', y_scale='log')

        return x, y, y_err

    time_tags = np.array([1.0, 2.0, 5.0, 10.0])

    # This is the power law that defines the normalization as a function of time

    normalizations = 0.23 * time_tags ** (-1.2)

    datasets = list(map(generate_one, normalizations))

    # Now set up the fit and fit it

    time = IndependentVariable("time", 1.0, u.s)

    plugins = []

    for i, dataset in enumerate(datasets):
        x, y, y_err = dataset

        xyl = XYLike("data%i" % i, x, y, y_err)

        xyl.tag = (time, time_tags[i])

        assert xyl.tag == (time, time_tags[i], None)

        plugins.append(xyl)

    data = DataList(*plugins)

    spectrum = Powerlaw()
    spectrum.K.bounds = (0.01, 1000.0)

    src = PointSource("test", 0.0, 0.0, spectrum)

    model = Model(src)

    model.add_independent_variable(time)

    time_po = Powerlaw()
    time_po.K.bounds = (0.01, 1000)
    time_po.K.value = 2.0
    time_po.index = -1.5

    model.link(spectrum.K, time, time_po)

    jl = JointLikelihood(model, data)

    jl.set_minimizer("minuit")

    best_fit_parameters, likelihood_values = jl.fit()

    # Make sure we are within 10% of the expected result

    assert np.allclose(
        best_fit_parameters["value"].values,
        [0.25496115, -1.2282951, -2.01508341],
        rtol=0.1,
    )
