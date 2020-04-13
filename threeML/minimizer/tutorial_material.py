from __future__ import division

# Leave these imports here, even though they look not used in the module, as they are used in the tutorial

from builtins import zip
from builtins import map
from builtins import range
from past.utils import old_div
from threeML.minimizer.grid_minimizer import GridMinimizer

# from threeML.minimizer.ROOT_minimizer import ROOTMinimizer
from threeML.minimizer.minuit_minimizer import MinuitMinimizer
from threeML.minimizer.grid_minimizer import GridMinimizer

from astromodels import Gaussian, Function1D, FunctionMeta, Model, PointSource
from threeML.plugin_prototype import PluginPrototype
from threeML.data_list import DataList
from threeML.classicMLE.joint_likelihood import JointLikelihood

from astromodels import use_astromodels_memoization

import matplotlib.pyplot as plt
import numpy as np
from future.utils import with_metaclass


# You don't need to do this in a normal 3ML analysis
# This is only for illustrative purposes
def get_callback(jl):
    def global_minim_callback(best_value, minimum):

        jl.likelihood_model.test.spectrum.main.shape.jump_tracking()

    return global_minim_callback


class JointLikelihoodWrap(JointLikelihood):
    def fit(self, *args, **kwargs):

        self.likelihood_model.test.spectrum.main.shape.reset_tracking()
        self.likelihood_model.test.spectrum.main.shape.start_tracking()

        with use_astromodels_memoization(False):

            try:

                super(JointLikelihoodWrap, self).fit(*args, **kwargs)

            except:

                raise

            finally:

                self.likelihood_model.test.spectrum.main.shape.stop_tracking()


def get_joint_likelihood_object_simple_likelihood():

    minus_log_L = Simple()

    # Instance a plugin (in this case a special one for illustrative purposes)
    plugin = CustomLikelihoodLike("custom")
    # Set the log likelihood function explicitly. This is not needed for any other
    # plugin
    plugin.set_minus_log_likelihood(minus_log_L)

    # Make the data list (in this case just one dataset)
    data = DataList(plugin)

    src = PointSource("test", ra=0.0, dec=0.0, spectral_shape=minus_log_L)
    model = Model(src)

    jl = JointLikelihoodWrap(model, data, verbose=False)

    return jl, model


def get_joint_likelihood_object_complex_likelihood():

    minus_log_L = Complex()

    # Instance a plugin (in this case a special one for illustrative purposes)
    plugin = CustomLikelihoodLike("custom")
    # Set the log likelihood function explicitly. This is not needed for any other
    # plugin
    plugin.set_minus_log_likelihood(minus_log_L)

    # Make the data list (in this case just one dataset)
    data = DataList(plugin)

    src = PointSource("test", ra=0.0, dec=0.0, spectral_shape=minus_log_L)
    model = Model(src)

    jl = JointLikelihoodWrap(model, data, verbose=False)

    return jl, model


def plot_likelihood_function(jl, fig=None):

    if fig is None:

        fig, sub = plt.subplots(1, 1)

    original_mu = jl.likelihood_model.test.spectrum.main.shape.mu.value

    # Let's have a look at the -log(L) by plotting it

    mus = np.arange(1, 100, 0.01)  # These are 1,2,3,4...99
    _ = plt.plot(mus, list(map(jl.minus_log_like_profile, mus)))

    _ = plt.xlabel(r"$\mu$")
    _ = plt.ylabel(r"$-\log{L(\mu)}$")

    # Reset the tracking within the function
    jl.likelihood_model.test.spectrum.main.shape.mu = original_mu

    return fig


def plot_minimizer_path(jl, points=False):
    """

    :param jl:
    :type jl: JointLikelihood
    :return:
    """

    qx_ = np.array(
        jl.likelihood_model.test.spectrum.main.shape._traversed_points, dtype=float
    )
    qy_ = np.array(
        jl.likelihood_model.test.spectrum.main.shape._returned_values, dtype=float
    )

    fig, sub = plt.subplots(1, 1)

    # Every np.nan divide a set
    qx_sets = np.split(qx_, np.where(~np.isfinite(qy_))[0])
    qy_sets = np.split(qy_, np.where(~np.isfinite(qy_))[0])

    if not points:

        # Color map
        N = len(qx_sets)
        cmap = plt.cm.get_cmap("gist_earth", N + 1)

        for i, (qx, qy) in enumerate(zip(qx_sets, qy_sets)):

            sub.quiver(
                qx[:-1],
                qy[:-1],
                qx[1:] - qx[:-1],
                qy[1:] - qy[:-1],
                scale_units="xy",
                angles="xy",
                scale=1,
                color=cmap(i),
            )

    else:

        for i, (qx, qy) in enumerate(zip(qx_sets, qy_sets)):

            sub.plot(qx, qy, ".")

    # Now plot the likelihood function
    plot_likelihood_function(jl, fig)

    return fig


class CustomLikelihoodLike(PluginPrototype):
    def __init__(self, name):

        self._minus_log_l = None
        self._free_parameters = None

        super(CustomLikelihoodLike, self).__init__(name, {})

    def set_minus_log_likelihood(self, likelihood_function):

        self._minus_log_l = likelihood_function

    def set_model(self, likelihood_model_instance):
        """
        Set the model to be used in the joint minimization. Must be a LikelihoodModel instance.
        """

        # Gather free parameters
        self._free_parameters = likelihood_model_instance.free_parameters

    def get_log_like(self):
        """
        Return the value of the log-likelihood with the current values for the
        parameters
        """

        # Gather values
        values = [x.value for x in list(self._free_parameters.values())]

        return -self._minus_log_l(*values)

    inner_fit = get_log_like

    def get_number_of_data_points(self):

        return 1


class Simple(with_metaclass(FunctionMeta, Function1D)):
    """
    description :

        A convex log likelihood

    latex : n.a.

    parameters :

        k :
            desc : normalization
            initial value : 1.0
            fix : yes

        mu :

            desc : parameter
            initial value : 5.0
            min : 1.0
            max : 100

        """

    def _setup(self):

        self._gau = Gaussian(F=100.0, mu=40, sigma=10)  # type: Gaussian

        self._returned_values = []
        self._traversed_points = []

        self._track = False

    def reset_tracking(self):

        self._returned_values = []
        self._traversed_points = []

    def start_tracking(self):

        self._track = True

    def stop_tracking(self):

        self._track = False

    def jump_tracking(self):

        self._returned_values.append(np.nan)
        self._traversed_points.append(np.nan)

    def _set_units(self, x_unit, y_unit):

        self.mu.unit = x_unit
        self.k.unit = y_unit

    # noinspection PyPep8Naming
    def evaluate(self, x, k, mu):

        val = -k * self._gau(x)

        if self._track:

            self._traversed_points.append(float(mu))
            self._returned_values.append(float(val))

        return val


class Complex(Simple):
    """
    description :

        A convex log likelihood with multiple minima

    latex : n.a.

    parameters :

        k :
            desc : normalization
            initial value : 1.0
            fix : yes

        mu :

            desc : parameter
            initial value : 5.0
            min : 1.0
            max : 100

        """

    def _setup(self):

        self._gau = Gaussian(F=100.0, mu=40, sigma=10)

        # + Gaussian(F=50.0, mu=60, sigma=5)

        for i in range(3):

            self._gau += Gaussian(
                F=100.0 / (i + 1), mu=10 + (i * 25), sigma=old_div(5, (i + 1))
            )

        self._returned_values = []
        self._traversed_points = []

        self._track = False

    def _set_units(self, x_unit, y_unit):

        self.mu.unit = x_unit
        self.k.unit = y_unit

    # noinspection PyPep8Naming
    def evaluate(self, x, k, mu):

        val = -k * self._gau(x)

        if self._track:

            self._traversed_points.append(mu)
            self._returned_values.append(val)

        return val
