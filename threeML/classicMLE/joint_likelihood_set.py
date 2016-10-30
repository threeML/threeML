import logging

log = logging.getLogger(__name__)

from threeML.classicMLE.joint_likelihood import JointLikelihood
from threeML.parallel.parallel_client import ParallelClient
from threeML.config.config import threeML_config
from threeML.data_list import DataList
from threeML.io.progress_bar import progress_bar
from astromodels import Model
import pandas as pd


class JointLikelihoodSet(object):
    def __init__(self, data_getter, model_getter,
                 n_iterations, iteration_name='interval'):

        # Store the data and model getter

        self._data_getter = data_getter

        # Test it here, so we don't need to do it in the worker (which would slow down things)
        data_test = self._data_getter(0)  # type: DataList

        assert isinstance(data_test, DataList), "The data_getter should return a DataList instance"

        # Now get the first model(s) and see whether there is one or more models
        # Then, we make a wrapper if it returns only one model, so that we will not need to specialize
        # the worker, as it will be able to assume that self._model_getter always returns a list of models
        # (of maybe one element)

        model_or_models = model_getter(0)

        try:

            len(model_or_models)

        except TypeError:

            # Only one instance, let's check that it is actually a model

            assert isinstance(model_or_models, Model), "The model getter function should return a model or a list of " \
                                                       "models"

            # Wrap the function so that self._model_getter will return a list of one element

            self._model_getter = lambda id: [model_getter(id)]

        else:

            # More than one model

            # Check that all models are instances of Model
            for model in model_or_models:

                assert isinstance(model, Model), "The model getter function should return a model or a list of models"

            # No need for a wrapper in this case

            self._model_getter = model_getter

        # Set up some attributes we will need

        self._n_iterations = n_iterations

        # This is used only to print error messages

        self._iteration_name = iteration_name

        # Default minimizer is minuit

        self._minimizer = 'minuit'
        self._algorithm = None
        self._callback = None

        # by default there is no second minimizer

        self._2nd_minimizer = None
        self._2nd_algorithm = None

        # By default, crash if a fit fails

        self._continue_on_failure = False

    def set_minimizer(self, minimizer, algorithm=None, callback=None):

        self._minimizer = minimizer
        self._algorithm = algorithm
        self._callback = callback

    def set_secondary_minimizer(self, minimizer, algorithm=None):
        """
        Set the secondary minimizer, which will run after the first has completed, so that the two minimizers are
        run in a chain

        :param minimizer:
        :param algorithm:
        :return:
        """

        self._2nd_minimizer = minimizer
        self._2nd_algorithm = algorithm

    def worker(self, interval):

        # Print a message to divide one interval from another

        log.info("\n\n\n==========================================")
        log.info("JointLikelihoodSet: processing interval %s" % interval)
        log.info("==========================================\n\n")

        # Get the dataset for this interval

        this_data = self._data_getter(interval)  # type: DataList

        # Get the model for this interval

        this_models = self._model_getter(interval)

        n_models = len(this_models)

        # Fit all models and collect the results

        parameters_frames = []
        like_frames = []

        for this_model in this_models:

            # Prepare a joint likelihood and fit it

            jl = JointLikelihood(this_model, this_data)
            this_parameter_frame, this_like_frame = self._fitter(jl)

            # Append results

            parameters_frames.append(this_parameter_frame)
            like_frames.append(this_like_frame)

        # Now merge the results in one data frame for the parameters and one for the likelihood
        # values

        if n_models > 1:

            # Prepare the keys so that the first model will be indexed with model_0, the second model_1 and so on

            keys = map(lambda x: "model_%i" % x, range(n_models))

            # Concatenate all results in one frame for parameters and one for likelihood

            frame_with_parameters = pd.concat(parameters_frames, keys=keys)
            frame_with_like = pd.concat(like_frames, keys=keys)

        else:

            frame_with_parameters = parameters_frames[0]
            frame_with_like = like_frames[0]

        return frame_with_parameters, frame_with_like

    def _fitter(self, jl):

        # Set the minimizer
        jl.set_minimizer(self._minimizer, self._algorithm, callback=self._callback)

        try:

            model_results, logl_results = jl.fit(quiet=True, compute_covariance=False)

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

        if self._2nd_minimizer is not None:

            jl.set_minimizer(self._2nd_minimizer, self._2nd_algorithm)

            try:

                model_results, logl_results = jl.fit(quiet=True, compute_covariance=False)

            except Exception as e:

                log.error("\n\n**** SECONDARY FIT FAILED! ***")
                log.error("Reason:")
                log.error(repr(e))
                log.error("\n\n")

                if self._continue_on_failure:

                    # Return empty data frame

                    return pd.DataFrame(), pd.DataFrame()

                else:

                    raise

        return model_results, logl_results

    def go(self, continue_on_failure=True, verbose=False, **options_for_parallel_computation):

        # Generate the data frame which will contain all results

        if verbose:

            log.setLevel(logging.INFO)

        self._continue_on_failure = continue_on_failure

        # let's iterate, perform the fit and fill the data frame

        if threeML_config['parallel']['use-parallel']:

            # Parallel computation

            client = ParallelClient(**options_for_parallel_computation)

            # amr = client.interactive_map(self.worker, range(self._n_iterations))
            #
            # results = []
            #
            # with progress_bar(self._n_iterations) as p:
            #
            #     for i, res in enumerate(amr):
            #
            #         results.append(res)
            #
            #         p.increase()

            results = client.execute_with_progress_bar(self.worker, range(self._n_iterations))


        else:

            # Serial computation

            results = []

            with progress_bar(self._n_iterations) as p:

                for i in range(self._n_iterations):

                    results.append(self.worker(i))

                    p.increase()

        assert len(results) == self._n_iterations, "Something went wrong, I have %s results " \
                                                   "for %s intervals" % (len(results), self._n_iterations)

        # Store the results in the data frames

        parameter_frames = pd.concat(map(lambda x: x[0], results), keys=range(self._n_iterations))
        like_frames = pd.concat(map(lambda x: x[1], results), keys=range(self._n_iterations))

        return parameter_frames, like_frames


class JointLikelihoodSetTwoModels(JointLikelihoodSet):
    pass


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

            this_model[parameter].value = sub_frame['value'][parameter]

        return this_model, this_data
