from builtins import zip
from builtins import map
from astromodels import Model, PointSource, Uniform_prior, Log_uniform_prior
from threeML.data_list import DataList
from threeML.bayesian.bayesian_analysis import BayesianAnalysis
from threeML.minimizer.tutorial_material import Simple, Complex, CustomLikelihoodLike

from astromodels import use_astromodels_memoization

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


class BayesianAnalysisWrap(BayesianAnalysis):
    def sample(self, *args, **kwargs):

        self.likelihood_model.test.spectrum.main.shape.reset_tracking()
        self.likelihood_model.test.spectrum.main.shape.start_tracking()

        with use_astromodels_memoization(False):

            try:

                super(BayesianAnalysisWrap, self).sample(*args, **kwargs)

            except:

                raise

            finally:

                self.likelihood_model.test.spectrum.main.shape.stop_tracking()


def get_bayesian_analysis_object_simple_likelihood():
    minus_log_L = Simple()

    minus_log_L.mu.set_uninformative_prior(Log_uniform_prior)

    # Instance a plugin (in this case a special one for illustrative purposes)
    plugin = CustomLikelihoodLike("custom")
    # Set the log likelihood function explicitly. This is not needed for any other
    # plugin
    plugin.set_minus_log_likelihood(minus_log_L)

    # Make the data list (in this case just one dataset)
    data = DataList(plugin)

    src = PointSource("test", ra=0.0, dec=0.0, spectral_shape=minus_log_L)
    model = Model(src)

    bayes = BayesianAnalysisWrap(model, data, verbose=False)

    return bayes, model


def get_bayesian_analysis_object_complex_likelihood():
    minus_log_L = Complex()

    minus_log_L.mu.set_uninformative_prior(Log_uniform_prior)

    # Instance a plugin (in this case a special one for illustrative purposes)
    plugin = CustomLikelihoodLike("custom")
    # Set the log likelihood function explicitly. This is not needed for any other
    # plugin
    plugin.set_minus_log_likelihood(minus_log_L)

    # Make the data list (in this case just one dataset)
    data = DataList(plugin)

    src = PointSource("test", ra=0.0, dec=0.0, spectral_shape=minus_log_L)
    model = Model(src)

    bayes = BayesianAnalysisWrap(model, data, verbose=False)

    return bayes, model


def array_to_cmap(values, cmap, use_log=False):
    """
    Generates a color map and color list that is normalized
    to the values in an array. Allows for adding a 3rd dimension
    onto a plot

    :param values: a list a values to map into a cmap
    :param cmap: the mpl colormap to use
    :param use_log: if the mapping should be done in log space
    """

    if use_log:

        norm = mpl.colors.LogNorm(vmin=min(values), vmax=max(values))

    else:

        norm = mpl.colors.Normalize(vmin=min(values), vmax=max(values))

    cmap = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    rgb_colors = [cmap.to_rgba(v) for v in values]

    return rgb_colors


def plot_likelihood_function(bayes, fig=None, show_prior=False):
    if fig is None:

        fig, sub = plt.subplots(1, 1)

    else:

        sub = fig.axes[0]

    original_mu = bayes.likelihood_model.test.spectrum.main.shape.mu.value

    # Let's have a look at the -log(L) by plotting it

    mus = np.arange(1, 100, 0.01)  # These are 1,2,3,4...99

    log_like = []

    for mu in mus:
        bayes.likelihood_model.test.spectrum.main.shape.mu.value = mu
        log_like.append(-bayes.sampler._log_like(mu))

    _ = sub.plot(mus, log_like, "k--", alpha=0.8)

    if show_prior:

        prior = []

        for mu in mus:

            prior.append(-bayes.sampler._log_prior([mu]))

        _ = sub.plot(mus, prior, "r")

    _ = sub.set_xlabel(r"$\mu$")
    _ = sub.set_ylabel(r"$-\log{L(\mu)}$")

    # Reset the tracking within the function
    bayes.likelihood_model.test.spectrum.main.shape.mu = original_mu

    return bayes


def plot_sample_path(bayes, burn_in=None, truth=None):
    """

    :param jl:
    :type jl: JointLikelihood
    :return:
    """

    qx_ = np.array(
        bayes.likelihood_model.test.spectrum.main.shape._traversed_points, dtype=float
    )
    qy_ = np.array(
        bayes.likelihood_model.test.spectrum.main.shape._returned_values, dtype=float
    )

    fig, (ax, ax1) = plt.subplots(
        2, 1, sharex=False, gridspec_kw={"height_ratios": [2, 1]}
    )

    time = np.arange(len(qx_)) + 1

    colors = array_to_cmap(time, "viridis")

    ax.scatter(qx_, qy_, c=np.atleast_2d(colors), s=17, alpha=0.4)
    ax1.scatter(time, qx_, c=np.atleast_2d(colors), s=10)
    # for i, (qx, qy) in enumerate(zip(qx_, qy_)):
    #     ax.scatter(qx, qy, c=np.atleast_2d(colors[i]), s=17, alpha=.4)

    #     ax1.scatter(time[i], qx, c=np.atleast_2d(colors[i]), s=10)

    if truth is not None:

        ax1.axhline(truth, ls="--", color="k", label=r"True $\mu=$%d" % truth)

    if burn_in is not None:

        ax1.axvline(burn_in, ls=":", color="#FC2530", label="Burn in")

    ax1.legend(loc="upper right", fontsize=7, frameon=False)

    # Now plot the likelihood function
    plot_likelihood_function(bayes, fig)

    ax1.set_xlabel("Iteration Number")
    ax1.set_ylabel(r"$\mu$")
    fig.subplots_adjust(hspace=0.2)

    return fig
