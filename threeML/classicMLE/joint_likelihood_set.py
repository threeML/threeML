import logging

#log = logging.getLogger(__name__)

class hack(object):

    def info(self, something):

        pass

    def error(self, something):

        pass

log = hack()

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

        self._data_getter = data_getter
        self._model_getter = model_getter

        # Get the first model and get all free parameters from there
        first_model = self._model_getter(0)

        assert isinstance(first_model, Model), "The model_getter did not return a Model instance for " \
                                                  "interval 0"

        first_data = self._data_getter(0)

        assert isinstance(first_data, DataList), "The data_getter did not return a DataList instance for " \
                                                 "interval 0"

        # Make a useless JointLikelihood object because the plugins could add their own nuisance parameters
        _ = JointLikelihood(first_model, first_data)

        # Gather names of parameters and plugin instances

        self._parameter_names = first_model.free_parameters.keys()
        self._dataset_names = first_data.keys()

        self._n_iterations = n_iterations

        self._iteration_name = iteration_name

        self._minimizer = 'minuit'
        self._algorithm = None

        self._2nd_minimizer = None
        self._2nd_algorithm = None

    def set_minimizer(self, minimizer, algorithm=None):

        self._minimizer = minimizer
        self._algorithm = algorithm

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

    def go(self, continue_on_failure=True, verbose=False, **options_for_parallel_computation):

        # Generate the data frame which will contain all results

        # First generate the multi-index

        multi_index = pd.MultiIndex.from_product([range(self._n_iterations), self._parameter_names],
                                                 names=[self._iteration_name, 'parameter'])

        # Now the empty data frames

        data_frame = pd.DataFrame(index=multi_index, columns=['value', 'error'])

        # Now do the same for the likelihood dataframe

        # Add the 'total' likelihood
        likelihood_names = ['total']
        likelihood_names.extend(self._dataset_names)

        multi_index_like = pd.MultiIndex.from_product([range(self._n_iterations), likelihood_names],
                                                      names=[self._iteration_name, 'dataset'])

        like_data_frame = pd.DataFrame(index=multi_index_like, columns=['-log(likelihood)'])

        # let's iterate, perform the fit and fill the data frame

        def worker(interval):

            # Print a message to divide one interval from another

            log.info("\n\n\n==========================================")
            log.info("JointLikelihoodSet: processing interval %s" % interval)
            log.info("==========================================\n\n")

            # Get the dataset for this interval

            this_data = self._data_getter(interval)

            assert isinstance(this_data, DataList), "The data_getter did not return a DataList instance for " \
                                                    "interval %s" % interval

            # Enforce that the data list contains all data sets

            for dataset in self._dataset_names:

                if dataset not in this_data.keys():

                    raise ValueError("The dataset %s is not contained in interval %s" % (dataset, interval))

            # Get the model for this interval

            this_model = self._model_getter(interval)

            assert isinstance(this_model, Model), "The model_getter did not return a Model instance for " \
                                                  "interval %s" % interval

            # Instance the joint likelihood (this might generate more parameters, as plugins can add their nuisance
            # parameters to the model)

            jl = JointLikelihood(this_model, this_data, verbose=verbose)

            # Enforce that the model contains all the required parameters

            for parameter in self._parameter_names:

                if parameter not in this_model.free_parameters:

                    raise ValueError("The parameter %s is not a free parameter of the model for "
                                     "interval %s" % (parameter, interval))

            # Set the minimizer
            jl.set_minimizer(self._minimizer, self._algorithm)

            try:

                model_results, logl_results = jl.fit(quiet=True, compute_covariance=False)

            except Exception as e:

                log.error("\n\n**** FIT FAILED! ***")
                log.error("Reason:")
                log.error(repr(e))
                log.error("\n\n")

                if continue_on_failure:

                    return None, None

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

                    if continue_on_failure:

                        return None, None

                    else:

                        raise

            return model_results, logl_results

        if threeML_config['parallel']['use-parallel']:

            # Parallel computation

            client = ParallelClient(**options_for_parallel_computation)

            amr = client.interactive_map(worker, xrange(self._n_iterations))

            results = []

            with progress_bar(self._n_iterations) as p:

                for i, res in enumerate(amr):

                    results.append(res)

                    p.increase()


        else:

            # Serial computation

            results = []

            with progress_bar(self._n_iterations) as p:

                for i in range(self._n_iterations):

                    results.append(worker(i))

                    p.increase()

        # Store the results in the data frame

        for interval, result in enumerate(results):

            model_results, logl_results = result

            if model_results is None:

                if continue_on_failure:

                    continue

                else:

                    raise RuntimeError("Results from interval %s are None!" % interval)

            for parameter in self._parameter_names:

                data_frame['value'][interval][parameter] = model_results['value'][parameter]
                data_frame['error'][interval][parameter] = model_results['error'][parameter]

            # Now store the likelihood values in the likelihood data frame

            for dataset in likelihood_names:

                like_data_frame['-log(likelihood)'][interval][dataset] = logl_results['-log(likelihood)'][dataset]

        return data_frame, like_data_frame


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