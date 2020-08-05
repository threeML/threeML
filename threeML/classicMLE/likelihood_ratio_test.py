from builtins import object
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

from astromodels import clone_model

from threeML.classicMLE.joint_likelihood import JointLikelihood
from threeML.classicMLE.joint_likelihood_set import JointLikelihoodSet
from threeML.data_list import DataList
from threeML.exceptions.custom_exceptions import custom_warnings
from threeML.plugins.OGIPLike import OGIPLike
from threeML.utils.OGIP.pha import PHAWrite


class LikelihoodRatioTest(object):
    def __init__(self, joint_likelihood_instance0, joint_likelihood_instance1):

        self._joint_likelihood_instance0 = (
            joint_likelihood_instance0
        )  # type: JointLikelihood
        self._joint_likelihood_instance1 = (
            joint_likelihood_instance1
        )  # type: JointLikelihood

        # Restore best fit and store the reference value for the likelihood
        self._joint_likelihood_instance0.restore_best_fit()
        self._joint_likelihood_instance1.restore_best_fit()

        self._reference_TS = 2 * (
            self._joint_likelihood_instance0.current_minimum
            - self._joint_likelihood_instance1.current_minimum
        )

        # Safety check that the user has provided the models in the right order
        if self._reference_TS < 0:

            custom_warnings.warn(
                "The reference TS is negative, either you specified the likelihood objects "
                "in the wrong order, or the fit for the alternative hyp. has failed. Since the "
                "two hyp. are nested, by definition the more complex hypothesis should give a "
                "better or equal fit with respect to the null hypothesis."
            )

        # Check that the dataset is the same

        if (
            self._joint_likelihood_instance1.data_list
            != self._joint_likelihood_instance0.data_list
        ):

            # Since this check might fail if the user loaded twice the same data, only issue a warning, instead of
            # an exception.

            custom_warnings.warn(
                "The data lists for the null hyp. and for the alternative hyp. seems to be different."
                " If you loaded twice the same data and made the same data selections, disregard this "
                "message. Otherwise, consider the fact that the LRT is meaningless if the two data "
                "sets are not exactly the same. We will use the data loaded as part of the null "
                "hypothesis JointLikelihood object",
                RuntimeWarning,
            )

        # For saving pha files
        self._save_pha = False
        self._data_container = []

    def get_simulated_data(self, id):

        # Generate a new data set for each plugin contained in the data list

        new_datas = []

        for dataset in list(self._joint_likelihood_instance0.data_list.values()):

            # Make sure that the active likelihood model is the null hypothesis
            # This is needed if the user has used the same DataList instance for both
            # JointLikelihood instances
            dataset.set_model(self._joint_likelihood_instance0.likelihood_model)

            new_data = dataset.get_simulated_dataset("%s_sim" % dataset.name)

            new_datas.append(new_data)

        new_data_list = DataList(*new_datas)

        if self._save_pha:

            self._data_container.append(new_data_list)

        return new_data_list

    def get_models(self, id):

        # Make a copy of the best fit models, so that we don't touch the original models during the fit, and we
        # also always restart from the best fit (instead of the last iteration)

        new_model0 = clone_model(self._joint_likelihood_instance0.likelihood_model)
        new_model1 = clone_model(self._joint_likelihood_instance1.likelihood_model)

        return new_model0, new_model1

    def by_mc(self, n_iterations=1000, continue_on_failure=False, save_pha=False):
        """
        Compute the Likelihood Ratio Test by generating Monte Carlo datasets and fitting the current models on them.
        The fraction of synthetic datasets which have a value for the TS larger or equal to the observed one gives
        the null-hypothesis probability (i.e., the probability that the observed TS is obtained by chance from the
        null hypothesis)

        :param n_iterations: number of MC iterations to perform (default: 1000)
        :param continue_of_failure: whether to continue in the case a fit fails (False by default)
        :param save_pha: Saves pha files for reading into XSPEC as a cross check.
         Currently only supports OGIP data. This can become slow! (False by default)
        :return: tuple (null. hyp. probability, TSs, frame with all results, frame with all likelihood values)
        """

        self._save_pha = save_pha

        # Create the joint likelihood set
        jl_set = JointLikelihoodSet(
            self.get_simulated_data,
            self.get_models,
            n_iterations,
            iteration_name="simulation",
        )

        # Use the same minimizer as in the first joint likelihood object

        jl_set.set_minimizer(self._joint_likelihood_instance0.minimizer_in_use)

        # Run the set
        data_frame, like_data_frame = jl_set.go(continue_on_failure=continue_on_failure)

        # Get the TS values

        TS_ = 2 * (
            like_data_frame["-log(likelihood)"][:, "model_0", "total"]
            - like_data_frame["-log(likelihood)"][:, "model_1", "total"]
        )  # type: pd.Series

        TS = pd.Series(TS_.values, name="TS")

        # Compute the null hyp probability
        idx = TS >= self._reference_TS  # type: np.ndarray

        null_hyp_prob = np.sum(idx) / float(n_iterations)

        # save these for later
        self._null_hyp_prob = null_hyp_prob
        self._TS_distribution = TS

        # Save the sims to phas if requested
        if self._save_pha:

            self._process_saved_data()

        return null_hyp_prob, TS, data_frame, like_data_frame

    def plot_TS_distribution(self, show_chi2=True, scale=1.0, **hist_kwargs):
        """
        
        :param show_chi2: 
        :param scale: 
        :param hist_kwargs: 
        :return: 
        """

        fig, ax = plt.subplots()

        counts, bins, _ = ax.hist(
            self._TS_distribution, density=True, label="monte carlo", **hist_kwargs
        )

        ax.axvline(self._reference_TS, color="r", ls="--", label="Ref. TS")

        if show_chi2:

            x_plot = np.linspace(bins[0], bins[-1], 100,)

            # get the difference in number of free parameters

            dof = len(
                self._joint_likelihood_instance1.likelihood_model.free_parameters
            ) - len(self._joint_likelihood_instance0.likelihood_model.free_parameters)

            assert (
                dof >= 0
            ), "The difference in the number of parameters between the alternative and null models is negative!"

            chi2 = stats.chi2.pdf(x_plot, dof)

            if scale == 1.0:

                _scale = ""
            else:

                _scale = "%.1f" % scale

            label = r"$%s\chi^{2}_{%d}$" % (_scale, dof)

            ax.plot(x_plot, scale * chi2, label=label)

        ax.set_yscale("log")

        ax.set_xlabel("TS")
        ax.set_ylabel("Probability distribution")

        return fig

    @property
    def reference_TS(self):

        return self._reference_TS

    @property
    def TS_distribution(self):

        return self._TS_distribution

    @property
    def null_hypothesis_probability(self):

        return self._null_hyp_prob

    def _process_saved_data(self):
        """

        Saves data sets for each plugin to PHAs for OGIP data.


        :return:
        """

        for plugin in list(self._data_container[0].values()):

            assert isinstance(
                plugin, OGIPLike
            ), "Saving simulations is only supported for OGIP plugins currently"

        # The first entry is always a test by the JL Set class.
        # so we do not use it

        for key in list(self._data_container[0].keys()):

            per_plugin_list = []

            for data in self._data_container[1:]:

                per_plugin_list.append(data[key])

            # Now write them

            pha_writer = PHAWrite(*per_plugin_list)

            pha_writer.write("%s" % key, overwrite=True)
