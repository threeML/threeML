# Leave these imports here, even though they look not used in the module, as they are used in the tutorial

from threeML.minimizer.grid_minimizer import GridMinimizer
from threeML.minimizer.ROOT_minimizer import ROOTMinimizer
from threeML.minimizer.minuit_minimizer import MinuitMinimizer
from threeML.minimizer.grid_minimizer import GridMinimizer

from astromodels import Gaussian, Function1D, FunctionMeta, Model, PointSource
from threeML.plugin_prototype import PluginPrototype
from threeML.data_list import DataList
from threeML.classicMLE.joint_likelihood import JointLikelihood

from astromodels import use_astromodels_memoization

import matplotlib.pyplot as plt
import numpy as np


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


def plot_likelihood_function(jl):

    original_mu = jl.likelihood_model.test.spectrum.main.shape.mu.value

    # Let's have a look at the -log(L) by plotting it

    mus = np.arange(1, 100, 1.0)  # These are 1,2,3,4...99
    _ = plt.plot(mus, map(jl.minus_log_like_profile, mus))

    _ = plt.xlabel(r"$\mu$")
    _ = plt.ylabel(r"$-\log{L(\mu)}$")

    # Reset the tracking within the function
    jl.likelihood_model.test.spectrum.main.shape.mu = original_mu


def plot_minimizer_path(jl):
    """

    :param jl:
    :type jl: JointLikelihood
    :return:
    """

    qx = np.array(jl.likelihood_model.test.spectrum.main.shape._traversed_points)
    qy = np.array(jl.likelihood_model.test.spectrum.main.shape._returned_values)

    fig, sub = plt.subplots(1,1)

    sub.quiver(qx[:-1], qy[:-1],
               qx[1:] - qx[:-1], qy[1:] - qy[:-1],
               scale_units='xy', angles='xy', scale=1)

    # Now plot the likelihood function
    plot_likelihood_function(jl)

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
        values = map(lambda x:x.value, self._free_parameters.values())

        return -self._minus_log_l(*values)

    inner_fit = get_log_like

    def get_number_of_data_points(self):

        return 1


class Simple(Function1D):
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

    __metaclass__ = FunctionMeta

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
