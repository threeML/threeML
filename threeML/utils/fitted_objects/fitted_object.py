__author__ = "grburgess"

import itertools

import numpy as np
import scipy.stats as stats

from threeML.utils.differentiation import get_jacobian
from threeML.utils.stats_tools import highest_density_posterior
from threeML.analysis_results import _AnalysisResults
from threeML.classicMLE.joint_likelihood import JointLikelihood
from threeML.bayesian.bayesian_analysis import BayesianAnalysis



class GenericFittedObject(object):
    def __init__(self, analysis_result, new_function, cl, *independent_variable_range):
        """
        A generic 3ML fitted object

        :param analysis_result: a 3ML analysis result
        :param source: an astromodels source
        :param new_function: the function to use the fitted values to compute new values
        :param sigma: the level of sigma to display in the contours
        :param independent_variable_range: the range(s) of independent values to compute the new function over
        """

        # lock variables to the class

        # we can pass either an analysis (Bayesian or JL) and extract the results
        if isinstance(analysis_result,JointLikelihood) or isinstance(analysis_result,BayesianAnalysis):

            # extract the result
            self._analysis_results = analysis_result.results

        elif isinstance(analysis_result,_AnalysisResults):

            self._analysis_results = analysis_result





        self._analysis = analysis_result
        self._independent_variable_range = independent_variable_range

        self._cl = cl


        # if only 1-D then we must place into its own tuple to
        # keep from confusing itertools

        if len(self._independent_variable_range) == 1:
            self._independent_variable_range = (self._independent_variable_range[0],)

        self._function = new_function

        # figure out the output shape of the best fit and errors

        self._out_shape = tuple(map(len, self._independent_variable_range))

        # fold the function through its independent values
        self._compute_best_fit_values()


        self._compute_error_region()




    @property
    def median(self):
        """
        return the evaluation at the best fit

        :return: best fit value(s)
        """

        return self._analysis_results.median

    @property
    def mean(self):
        """
        return the evaluation at the best fit

        :return: best fit value(s)
        """

        return self._analysis_results.mean

    @property
    def error_region(self):
        """
        The error region of the newly computed quantity

        :return: error region
        """

        return self._error_region

    def update_tag(self, tag, value):

        pass


    def _build_propagated_function(self):

        arguments = {}
        for par in self._function.parameters.values():

            if par.free:

                this_name = par.name

                this_variate = self._analysis_results.get_variates(par.path)

                # Do not use more than 1000 values (would make computation too slow for nothing)

                if len(this_variate) > 1000:
                    this_variate = np.random.choice(this_variate, size=1000)

                arguments[this_name] = this_variate

                # Prepare the error propagator function

        pp = self._analysis_results.propagate(self._function, **arguments)





    def _compute_best_fit_values(self):
        """

        calculate the best or mean fit of the new function or
        quantity



        :return:
        """



        # if there are independent variables
        if self._independent_variable_range:

            values = []

            # scroll through the independent variables

            for variables in itertools.product(*self._independent_variable_range):
                values.append(self._function(*variables))

            values = np.array(values)

            # reshape and attach to the class

            self._best_fit_values = values.reshape(self._out_shape)

        # otherwise just evaluate
        else:

            self._best_fit_values = self._function()

    def _compute_error_region(self, function, *independent_variables):

        raise RuntimeError("Must be implemented in subclass") # pragma: no cover

    def _calculate_alpha(self, sigma):
        """ alpha = 1-p  """

        return (stats.norm.sf(sigma) * 2)

class MLEFittedObject(GenericFittedObject):


    def __add__(self, other):
        """
        how MLE objects add together.

        :param other:
        :return: total best fit and total error dict
        """

        # first add together the main value

        other_best = other.best_fit

        other_error = other.error_region

        # add sum of squares

        total_error = np.sqrt((self.error_region) ** 2 + other_error ** 2)

        total_best = self.best_fit + other_best

        # to facilitate summing

        new_container = ErrorContainer()

        new_container['best_fit'] = total_best
        new_container['error_region'] = total_error

        return new_container


    def __radd__(self, other):

        if other == 0:

            new_container = ErrorContainer()

            new_container['error_region'] = self.error_region
            new_container['best_fit'] = self.best_fit

            return new_container


        elif isinstance(other, ErrorContainer):

            other_error = other['error_region']
            other_best = other['best_fit']

            total_best = self.best_fit + other_best
            total_error = np.sqrt(self.error_region ** 2 + other_error ** 2)

            new_container = ErrorContainer()

            new_container['error_region'] = total_error
            new_container['best_fit'] = total_best

            return new_container



        else:

            raise RuntimeError('Cannot add together these types!')



    def _compute_error_region(self, ):
        """
        propagate errors via gaussian error propagation

        :return:
        """

        self._error_region = self._propagate_into_function()


    @property
    def error_region(self):
        """
        Return the MLE error region and the specified error level

        :return:
        """

        return self._sigma * super(MLEFittedObject,self).error_region


    def _propagate_into_function(self):
        """

        propagate the cavariance matrix into the
        new function


        :return:
        """

        errors = []

        # Get the parameters from the minimizer

        parameters = self._analysis.minimizer.parameters

        # We will compute the error at each  interval
        # so that we can plot the spread as a function of energy

        if self._independent_variable_range:

            # with progress_bar(len(energy)) as p:
            for variables in itertools.product(*self._independent_variable_range):

                first_derivatives = []

                # Now loop through each parameter and free it while
                # holding the others constant. This is the normal (pun intended)
                # error propagation formula.

                for par in parameters.keys():
                    # go back to the best fit

                    self._analysis.restore_best_fit()

                    parameter_best_fit_value = parameters[par].value
                    min_value, max_value = parameters[par].bounds

                    # Create a temporary flux function to take a
                    # derivative w.r.t. the free parameter

                    def tmpflux(current_value):
                        parameters[par].value = current_value

                        return self._function(*variables)  # .value

                    # get the first derivatives and append them for some
                    # linear algebra

                    this_derivative = get_jacobian(tmpflux, parameter_best_fit_value, min_value, max_value)[0][0]

                    first_derivatives.append(this_derivative)

                first_derivatives = np.array(first_derivatives)

                # Now we take the inner product with the covariance matrix

                tmp = first_derivatives.dot(self._analysis.covariance_matrix)

                errors.append(np.sqrt(tmp.dot(first_derivatives)))

            errors = np.array(errors).reshape(self._out_shape)


        else:

            # the function has no independent variables

            first_derivatives = []

            # Now loop through each parameter and free it while
            # holding the others constant. This is the normal (pun intended)
            # error propagation formula.

            for par in parameters.keys():
                # go back to the best fit

                self._analysis.restore_best_fit()

                parameter_best_fit_value = parameters[par].value
                min_value, max_value = parameters[par].bounds

                # Create a temporary flux function to take a
                # derivative w.r.t. the free parameter

                def tmpflux(current_value):
                    parameters[par].value = current_value

                    return self._function()  # .value

                # get the first derivatives and append them for some
                # linear algebra

                this_derivative = get_jacobian(tmpflux, parameter_best_fit_value, min_value, max_value)[0][0]

                first_derivatives.append(this_derivative)

            first_derivatives = np.array(first_derivatives)

            # Now we take the inner product with the covariance matrix

            tmp = first_derivatives.dot(self._analysis.covariance_matrix)

            errors = np.sqrt(tmp.dot(first_derivatives))


        return errors

class BayesianFittedObject(GenericFittedObject):


    def __add__(self, other):
        """
        how bayesian objects add together.

        :param other:
        :return: total best fit and total error dict
        """

        # first add together the main value

        other_best = other.best_fit

        other_chains = other.raw_chains

        # direct sum

        total_chain = self.raw_chains + other_chains

        total_best = self.best_fit + other_best

        total_error = np.array([highest_density_posterior(mc, alpha=self._alpha) for mc in total_chain])

        # to facilitate summing

        new_container = ErrorContainer()

        new_container['best_fit'] = total_best
        new_container['error_region'] = total_error.reshape(self._out_shape +(2,))
        new_container['raw_chains'] = total_chain

        return new_container


    def __radd__(self, other):

        if other == 0:

            new_container = ErrorContainer()

            new_container['error_region'] = self.error_region
            new_container['best_fit'] = self.best_fit
            new_container['raw_chains'] = self.raw_chains

            return new_container


        elif isinstance(other, ErrorContainer):


            other_best = other['best_fit']

            other_chains = other['raw_chains']

            # direct sum

            total_chain = self.raw_chains + other_chains

            total_best = self.best_fit + other_best


            # This may already be an array

            try:

                total_error = np.array([highest_density_posterior(mc, alpha=self._alpha) for mc in total_chain])

            except(ValueError):

                # This means we had a quantity agrument from astropy

                old_unit = total_chain.unit

                total_error = np.array([highest_density_posterior(mc, alpha=self._alpha) for mc in total_chain.value])
                total_error = total_error * old_unit



            # to facilitate summing

            new_container = ErrorContainer()

            new_container['best_fit'] = total_best
            new_container['error_region'] = total_error.reshape(self._out_shape +(2,))
            new_container['raw_chains'] = total_chain

            return new_container



        else:

            raise RuntimeError('Cannot add together these types!')


    def _compute_error_region(self):

        # Get the the number of samples


        n_samples = self._analysis.raw_samples.shape[0]


        thin_step = min(int(1/self._fraction_of_samples), n_samples)

        # temporary list to store the propagated samples
        chains = []

        for sample_number in range(0, n_samples, thin_step):

            # go through parameters
            for parameter in self._analysis.samples.keys():

                mod_par = parameter.split('.')[-1]

                self._free_parameters[mod_par].value = self._analysis.samples[parameter][sample_number]


            values = []
            if self._independent_variable_range:
                for variables in itertools.product(*self._independent_variable_range):

                    values.append(self._function(*variables))


                values = np.array(values)

            else:

                values = self._function()

            chains.append(values)


        chains = np.array(chains).T

        self._raw_chains = chains

        contours = np.array([highest_density_posterior(mc, alpha=self._alpha) for mc in chains])

        # reshape the contours and remember the inner dimension is 2-D

        contours.reshape(self._out_shape +(2,))

        self._error_region = contours


    @property
    def raw_chains(self):
        """
        return the raw chains without being converted to
        HDPs

        :return:
        """
        return self._raw_chains

from collections import MutableMapping

class ErrorContainer(MutableMapping):
    _allowed_keys = "error_region best_fit raw_chains".split()

    def accept(self, key):

        return key in ErrorContainer._allowed_keys

    def __init__(self):
        """
        A simple container to store errors that
        allows for easy adding of error regions

        """

        self.dict = dict()

    def __setitem__(self, key, val):
        if key not in ErrorContainer._allowed_keys:
            raise KeyError(
                'Valid keywords: %s'%', '.join(ErrorContainer._allowed_keys))
        self.dict[key] = val

    def __getitem__(self, key):
        return self.dict[key]

    def __delitem__(self, key):
        RuntimeWarning("You cannot delete keys!")

    def __len__(self):
        return sum(1 for _ in self)

    def __iter__(self):
        for key in self.dict:
            yield key

    def __repr__(self):
        return repr(dict(self))

    def __str__(self):
        return str(dict(self))


    @property
    def raw_chains(self):
        """
        raw mc chains from a bayesian analysis
        :return:
        """

        return self.dict['raw_chains']

    @property
    def error_region(self):
        """
        the error region
        :return:
        """

        return self.dict['error_region']

    @property
    def best_fit(self):
        """
        the best fit of the propagated function
        :return:
        """

        return self.dict['best_fit']