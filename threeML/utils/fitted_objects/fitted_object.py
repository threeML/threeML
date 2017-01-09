__author__ = "grburgess"

import itertools

import numpy as np
import scipy.stats as stats

from threeML.utils.differentiation import get_jacobian




class GenericFittedObject(object):
    def __init__(self, analysis, source, new_function, sigma ,*independent_variable_range):
        """
        A generic 3ML fitted object

        :param analysis: a 3ML JointLikelihood or BayesianAnalysis object
        :param source: an astromodels source
        :param new_function: the function to use the fitted values to compute new values
        :param sigma: the level of sigma to display in the contours
        :param independent_variable_range: the range(s) of independent values to compute the new function over
        """

        # lock variables to the class

        self._analysis = analysis
        self._source = source
        self._independent_variable_range = independent_variable_range

        self._sigma = sigma
        self._alpha = self._calculate_alpha(sigma)

        # if only 1-D then we must place into its own tuple to
        # keep from confusing itertools

        if len(self._independent_variable_range) == 1:
            self._independent_variable_range = (self._independent_variable_range[0],)

        self._function = new_function

        self._get_free_parameters()

        # figure out the output shape of the best fit and errors

        self._out_shape = tuple(map(len, self._independent_variable_range))

        # fold the function through its independent values
        self._compute_best_fit_values()


        self._compute_error_region()

    @property
    def error_region(self):
        """

        The error region of the newly computed quantity

        :return: error region
        """

        return self._error_region

    def update_tag(self, tag, value):

        pass

    def _get_free_parameters(self):

        raise RuntimeError("Must be implemented in subclass") # pragma: no cover

    def _compute_best_fit_values(self):
        """

        calculate the best or mean fit of the new function or
        quantity



        :return:
        """

        self._analysis.restore_best_fit()

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
        :return: total best fit and total error
        """

        # first add together the main value

        total = self._best_fit_values + other._best_fit_values

        # add sum of squares

        total_error = np.sqrt((self.error_region) ** 2 + other.error_region ** 2)

        return total, total_error

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

        total = self._best_fit_values + other._best_fit_values

        error = self._error_region + other._error_region


        return total, error

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

        contours = np.array([self._analysis._hpd(mc, alpha=self._alpha) for mc in chains])

        # reshape the contours and remember the inner dimension is 2-D

        contours.reshape(self._out_shape +(2,))

        self._error_region = contours


