import pandas as pd

from threeML.classicMLE.joint_likelihood import JointLikelihood
from threeML.data_list import DataList
from astromodels import Model


class JointLikelihoodSet(object):

    def __init__(self, data_getter, model_getter,
                 n_iterations,
                 parameter_names=None,
                 dataset_names=None,
                 iteration_name='interval'):

        self._data_getter = data_getter
        self._model_getter = model_getter

        if parameter_names is None:

            # Get the first model and get all free parameters from there
            first_model = self._model_getter(0)

            assert isinstance(first_model, Model), "The model_getter did not return a Model instance for " \
                                                      "interval 0"

            self._parameter_names = first_model.free_parameters.keys()

        else:

            self._parameter_names = parameter_names

        if dataset_names is None:

            first_data = self._data_getter(0)

            assert isinstance(first_data, DataList), "The data_getter did not return a DataList instance for " \
                                                     "interval 0"

            self._dataset_names = first_data.keys()

        else:

            self._dataset_names = dataset_names

        self._n_iterations = n_iterations

        self._iteration_name = iteration_name

        self._minimizer = 'minuit'
        self._algoritihm = None

    def set_minimizer(self, minimizer, algorithm=None):

        self._minimizer = minimizer
        self._algoritihm = algorithm

    def go(self):

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

        for interval in range(self._n_iterations):

            # Print a message to divide one interval from another

            print("\n\n\n==========================================")
            print("JointLikelihoodSet: processing interval %s" % interval)
            print("==========================================\n\n")

            # Get the dataset for this interval

            this_data = self._data_getter(interval)

            assert isinstance(this_data, DataList), "The data_getter did not return a DataList instance for " \
                                                    "interval %s" % interval

            # Enforce that the data list contains all data sets

            for dataset in self._dataset_names:

                if dataset not in this_data.keys():

                    raise ValueError("The dataset %s is not contained in interval %s" % dataset)

            # Get the model for this interval

            this_model = self._model_getter(interval)

            assert isinstance(this_model, Model), "The model_getter did not return a Model instace for " \
                                                  "interval %s" % interval

            # Enforce that the model contains all the required parameters

            for parameter in self._parameter_names:

                if parameter not in this_model.free_parameters:

                    raise ValueError("The parameter %s is not a free parameter of the model for interval %s" % interval)

            # Instance the joint likelihood and execute the fit

            jl = JointLikelihood(this_model, this_data)

            # Set the minimizer
            jl.set_minimizer(self._minimizer, self._algoritihm)

            try:

                model_results, logl_results = jl.fit()

            except Exception as e:

                print("\n\n**** FIT FAILED! ***")
                print("Reason:")
                print(repr(e))
                print("\n\n")

                continue

            # Store the results in the data frame

            for parameter in self._parameter_names:

                data_frame['value'][interval][parameter] = model_results['value'][parameter]
                data_frame['error'][interval][parameter] = model_results['error'][parameter]

            # Now store the likelihood values in the likelihood data frame

            for dataset in likelihood_names:

                like_data_frame['-log(likelihood)'][interval][dataset] = logl_results['-log(likelihood)'][dataset]

        return data_frame, like_data_frame