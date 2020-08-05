from builtins import object
import collections
import numpy as np

from threeML.classicMLE.joint_likelihood_set import JointLikelihoodSet
from threeML.data_list import DataList
from astromodels import clone_model


class GoodnessOfFit(object):
    def __init__(self, joint_likelihood_instance, like_data_frame=None):

        self._jl_instance = joint_likelihood_instance

        # Make sure we have a fit
        assert (
            self._jl_instance.results is not None
        ), "You have to perform a fit before using GoodnessOfFit"

        if like_data_frame is None:

            like_data_frame = self._jl_instance.results.get_statistic_frame()

        # Restore best fit and store the reference value for the likelihood
        self._jl_instance.restore_best_fit()

        self._reference_like = like_data_frame["-log(likelihood)"]

        # Store best model
        self._best_fit_model = clone_model(self._jl_instance.likelihood_model)

    def get_simulated_data(self, id):

        # Make sure we start from the best fit model
        self._jl_instance.restore_best_fit()

        # Generate a new data set for each plugin contained in the data list

        new_datas = []

        for dataset in list(self._jl_instance.data_list.values()):

            new_data = dataset.get_simulated_dataset("%s_sim" % dataset.name)

            new_datas.append(new_data)

        new_data_list = DataList(*new_datas)

        return new_data_list

    def get_model(self, id):

        # Make a copy of the best fit model, so that we don't touch the original model during the fit, and we
        # also always restart from the best fit (instead of the last iteration)

        new_model = clone_model(self._best_fit_model)

        return new_model

    def by_mc(self, n_iterations=1000, continue_on_failure=False):
        """
        Compute goodness of fit by generating Monte Carlo datasets and fitting the current model on them. The fraction
        of synthetic datasets which have a value for the likelihood larger or equal to the observed one is a measure
        of the goodness of fit

        :param n_iterations: number of MC iterations to perform (default: 1000)
        :param continue_of_failure: whether to continue in the case a fit fails (False by default)
        :return: tuple (goodness of fit, frame with all results, frame with all likelihood values)
        """

        # Create the joint likelihood set
        jl_set = JointLikelihoodSet(
            self.get_simulated_data,
            self.get_model,
            n_iterations,
            iteration_name="simulation",
        )

        # Use the same minimizer as in the joint likelihood object
        # NOTE: we use a clone so that the original best fit will not be touched

        jl_set.set_minimizer(self._jl_instance.minimizer_in_use)

        # Run the set
        data_frame, like_data_frame = jl_set.go(continue_on_failure=continue_on_failure)

        # Compute goodness of fit

        gof = collections.OrderedDict()

        # Total
        idx = (
            like_data_frame["-log(likelihood)"][:, "total"].values
            >= self._reference_like["total"]
        )  # type: np.ndarray

        gof["total"] = np.sum(idx) / float(n_iterations)

        for dataset in list(self._jl_instance.data_list.values()):

            sim_name = "%s_sim" % dataset.name

            idx = (
                like_data_frame["-log(likelihood)"][:, sim_name].values
                >= self._reference_like[dataset.name]
            )

            gof[dataset.name] = np.sum(idx) / float(n_iterations)

        return gof, data_frame, like_data_frame
