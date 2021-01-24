
import warnings
from builtins import object, range

import numpy as np
import pandas as pd
from astromodels import Model

from threeML.analysis_results import AnalysisResultsSet
from threeML.classicMLE.joint_likelihood import JointLikelihood
from threeML.config.config import threeML_config
from threeML.data_list import DataList
from threeML.minimizer.minimization import (LocalMinimization, _Minimization,
                                            _minimizers)
from threeML.parallel.parallel_client import ParallelClient
from threeML.utils.progress_bar import trange
from threeML.io.logging import setup_logger


log = setup_logger(__name__)


class JointLikelihoodSet(object):
    def __init__(
        self,
        data_getter,
        model_getter,
        n_iterations,
        iteration_name="interval",
        preprocessor=None,
    ):

        # Store the data and model getter

        self._data_getter = data_getter

        # Now get the first model(s) and see whether there is one or more models
        # Then, we make a wrapper if it returns only one model, so that we will not need to specialize
        # the worker, as it will be able to assume that self._model_getter always returns a list of models
        # (of maybe one element)

        model_or_models = model_getter(0)

        try:

            n_models = len(model_or_models)

        except TypeError:

            # Only one instance, let's check that it is actually a model

            assert isinstance(model_or_models, Model), (
                "The model getter function should return a model or a list of " "models"
            )

            # Save that
            self._n_models = 1

            # Wrap the function so that self._model_getter will return a list of one element

            self._model_getter = lambda id: [model_getter(id)]

        else:

            # More than one model

            # Check that all models are instances of Model
            for this_model in model_or_models:

                assert isinstance(
                    this_model, Model
                ), "The model getter function should return a model or a list of models"

            # No need for a wrapper in this case

            self._model_getter = model_getter

            # save the number of models
            self._n_models = n_models

        # Set up some attributes we will need

        self._n_iterations = n_iterations

        # This is used only to print error messages

        self._iteration_name = iteration_name

        # Default minimizer is minuit

        self._minimization = LocalMinimization("minuit")

        # By default, crash if a fit fails

        self._continue_on_failure = False

        # By default do not compute the covariance matrix

        self._compute_covariance = False

        self._all_results = None

        self._preprocessor = preprocessor

    def set_minimizer(self, minimizer):

        if isinstance(minimizer, _Minimization):

            self._minimization = minimizer

        else:

            assert minimizer.upper() in _minimizers, (
                "Minimizer %s is not available on this system. "
                "Available minimizers: %s"
                % (minimizer, ",".join(list(_minimizers.keys())))
            )

            # The string can only specify a local minimization. This will return an error if that is not the case.
            # In order to setup global optimization the user needs to use the GlobalMinimization factory directly

            self._minimization = LocalMinimization(minimizer)

        self._minimization = minimizer

    def worker(self, interval):

        # Get the dataset for this interval

        this_data = self._data_getter(interval)  # type: DataList

        # Get the model for this interval

        this_models = self._model_getter(interval)

        # Apply preprocessor (if any)
        if self._preprocessor is not None:

            self._preprocessor(this_models, this_data)

        n_models = len(this_models)

        # Fit all models and collect the results

        parameters_frames = []
        like_frames = []
        analysis_results = []

        for this_model in this_models:

            # Prepare a joint likelihood and fit it

            with warnings.catch_warnings():

                warnings.simplefilter("ignore", RuntimeWarning)

                jl = JointLikelihood(this_model, this_data)

            this_parameter_frame, this_like_frame = self._fitter(jl)

            # Append results

            parameters_frames.append(this_parameter_frame)
            like_frames.append(this_like_frame)
            analysis_results.append(jl.results)

        # Now merge the results in one data frame for the parameters and one for the likelihood
        # values

        if n_models > 1:

            # Prepare the keys so that the first model will be indexed with model_0, the second model_1 and so on

            keys = ["model_%i" % x for x in range(n_models)]

            # Concatenate all results in one frame for parameters and one for likelihood

            frame_with_parameters = pd.concat(parameters_frames, keys=keys)
            frame_with_like = pd.concat(like_frames, keys=keys)

        else:

            frame_with_parameters = parameters_frames[0]
            frame_with_like = like_frames[0]

        return frame_with_parameters, frame_with_like, analysis_results

    def _fitter(self, jl):

        # Set the minimizer
        jl.set_minimizer(self._minimization)

        try:

            model_results, logl_results = jl.fit(
                quiet=True, compute_covariance=self._compute_covariance
            )

        except Exception as e:

            log.error("\n\n**** FIT FAILED! ***")
            log.error("Reason:")
            log.error(repr(e))
            log.error("\n\n")

            if self._continue_on_failure:

                # Return empty data frame

                return pd.DataFrame(), pd.DataFrame()

            else:

                raise

        return model_results, logl_results

    def go(
        self,
        continue_on_failure=True,
        compute_covariance=False,
        verbose=False,
        **options_for_parallel_computation
    ):

        # Generate the data frame which will contain all results


        self._continue_on_failure = continue_on_failure

        self._compute_covariance = compute_covariance

        # let's iterate, perform the fit and fill the data frame

        if threeML_config["parallel"]["use-parallel"]:

            # Parallel computation

            client = ParallelClient(**options_for_parallel_computation)

            results = client.execute_with_progress_bar(
                self.worker, list(range(self._n_iterations))
            )

        else:

            # Serial computation

            results = []

            for i in trange(self._n_iterations, desc="Goodness of fit computation"):

                results.append(self.worker(i))

        assert len(results) == self._n_iterations, (
            "Something went wrong, I have %s results "
            "for %s intervals" % (len(results), self._n_iterations)
        )

        # Store the results in the data frames

        parameter_frames = pd.concat(
            [x[0] for x in results], keys=list(range(self._n_iterations))
        )
        like_frames = pd.concat(
            [x[1] for x in results], keys=list(range(self._n_iterations))
        )

        # Store a list with all results (this is a list of lists, each list contains the results for the different
        # iterations for the same model)
        self._all_results = []

        for i in range(self._n_models):

            this_model_results = [x[2][i] for x in results]

            self._all_results.append(AnalysisResultsSet(this_model_results))

        return parameter_frames, like_frames

    @property
    def results(self):
        """
        Returns a results set for each model. If there is more than one model, it will return a list of
        AnalysisResultsSet instances, otherwise it will return one AnalysisResultsSet instance

        :return:
        """

        if len(self._all_results) == 1:

            return self._all_results[0]

        else:

            return self._all_results

    def write_to(self, filenames, overwrite=False):
        """
        Write the results to one file per model. If you need more control, get the results using the .results property
        then write each results set by itself.

        :param filenames: list of filenames, one per model, or filename (if there is only one model per interval)
        :param overwrite: overwrite existing files
        :return: None
        """

        # Trick to make filenames always a list
        filenames = list(np.array(filenames, ndmin=1))

        # Check that we have the right amount of file names
        assert len(filenames) == self._n_models

        # Now write one file for each model
        for i in range(self._n_models):

            this_results = self._all_results[i]

            this_results.write_to(filenames[i], overwrite=overwrite)


class JointLikelihoodSetAnalyzer(object):
    """
    A class to help in offline re-analysis of the results obtained with the JointLikelihoodSet class

    """

    def __init__(self, get_data, get_model, data_frame, like_data_frame):

        self._get_data = get_data
        self._get_model = get_model
        self._data_frame = data_frame
        self._like_data_frame = like_data_frame

    def restore_best_fit_model(self, interval):

        # Get sub-frame containing the results for the requested interval

        sub_frame = self._data_frame.loc[interval]

        # Get the model for this interval
        this_model = self._get_model(interval)

        # Get data for this interval
        this_data = self._get_data(interval)

        # Instance a useless joint likelihood object so that plugins have the chance to add nuisance parameters to the
        # model

        _ = JointLikelihood(this_model, this_data)

        # Restore best fit parameters
        for parameter in this_model.free_parameters:

            this_model[parameter].value = sub_frame["value"][parameter]

        return this_model, this_data
