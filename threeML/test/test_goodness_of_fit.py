import pytest

import numpy as np
import scipy.stats

from astromodels import Powerlaw
from threeML.plugins.XYLike import XYLike


def test_goodness_of_fit():

    # Let's generate some data with y = Powerlaw(x)

    gen_function = Powerlaw()

    # Generate a dataset using the power law, and a
    # constant 30% error

    x = np.logspace(0, 2, 50)

    xyl_generator = XYLike.from_function(
        "sim_data", function=gen_function, x=x, yerr=0.3 * gen_function(x)
    )

    y = xyl_generator.y
    y_err = xyl_generator.yerr

    fit_function = Powerlaw()

    xyl = XYLike("data", x, y, y_err)

    parameters, like_values = xyl.fit(fit_function)

    gof, all_results, all_like_values = xyl.goodness_of_fit()

    # Compute the number of degrees of freedom
    n_dof = len(xyl.x) - len(fit_function.free_parameters)

    # Get the observed value for chi2
    obs_chi2 = 2 * like_values["-log(likelihood)"]["data"]

    theoretical_gof = scipy.stats.chi2(n_dof).sf(obs_chi2)

    assert np.isclose(theoretical_gof, gof["total"], rtol=0.1)
