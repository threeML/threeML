import collections
import math
import os

import pymultinest
from astromodels.functions.functions import Uniform_prior, Log_uniform_prior

from threeML.minimizer.minimization import Minimizer
from threeML.io.file_utils import temporary_directory
from threeML.io.suppress_stdout import suppress_stdout

class MultinestMinimizer(Minimizer):

    def __init__(self, function, parameters, ftol=1e3, verbosity=1):

        super(MultinestMinimizer, self).__init__(function, parameters, ftol, verbosity)

    def _setup(self):

        # We need to wrap the function, because multinest maximizes instead of minimizing

        def func_wrapper(values, ndim, nparams):

            # values is a wrapped C class. Extract from it the values in a python list
            values_list = [values[i] for i in range(ndim)]

            return self.function(*values_list) * (-1)

        self._func_wrapper = func_wrapper

        # Now we need to build the global prior function, which in this case is just a set of uniform priors

        # MULTINEST priors are defined on the unit cube
        # and should return the value in the bounds... not the
        # probability.

        # First build a uniform prior for each parameters
        self._param_priors = collections.OrderedDict()

        for parameter_name in self.parameters:

            min_value, max_value = self.parameters[parameter_name].bounds

            assert min_value is not None, "Minimum value of parameter %s is None. In order to use the Multinest " \
                                          "minimizer you need to define proper bounds for each " \
                                          "free parameter" % parameter_name

            assert max_value is not None, "Maximum value of parameter %s is None. In order to use the Multinest " \
                                          "minimizer you need to define proper bounds for each " \
                                          "free parameter" % parameter_name

            # Compute the difference in order of magnitudes between minimum and maximum

            if min_value > 0:

                orders_of_magnitude_span = math.log10(max_value) - math.log10(min_value)

                if orders_of_magnitude_span > 2:

                    # Use a Log-uniform prior
                    self._param_priors[parameter_name] = Log_uniform_prior(lower_bound=min_value, upper_bound=max_value)

                else:

                    # Use a uniform prior
                    self._param_priors[parameter_name] = Uniform_prior(lower_bound=min_value, upper_bound=max_value)

            else:

                # Can only use a uniform prior
                self._param_priors[parameter_name] = Uniform_prior(lower_bound=min_value, upper_bound=max_value)

        def prior(params, ndim, nparams):

            for i, (parameter_name, parameter) in enumerate(self.parameters.iteritems()):

                try:

                    params[i] = self._param_priors[parameter_name].from_unit_cube(params[i])

                except AttributeError:

                    raise RuntimeError("The prior you are trying to use for parameter %s is "
                                       "not compatible with multinest" % parameter_name)

        # Give a test run to the prior to check that it is working. If it crashes while multinest is going
        # it will not stop multinest from running and generate thousands of exceptions (argh!)
        n_dim = len(self.parameters)

        _ = prior([0.5] * n_dim, n_dim, [])

        self._prior = prior

    def minimize(self, compute_covar=True):
        """
            Minimize the function using the Multinest sampler
         """

        n_dim = len(self.parameters)

        # We need to check if the MCMC
        # chains will have a place on
        # the disk to write and if not,
        # create one

        # chain_name = "multinest_minimizer/fit-"
        #
        # mcmc_chains_out_dir = ""
        # tmp = chain_name.split('/')
        # for s in tmp[:-1]:
        #     mcmc_chains_out_dir += s + '/'
        #
        # if not os.path.exists(mcmc_chains_out_dir):
        #
        #     os.makedirs(mcmc_chains_out_dir)

        with temporary_directory(prefix='multinest-', within_directory=os.getcwd()) as mcmc_chains_out_dir:

            outputfiles_basename = os.path.join(mcmc_chains_out_dir,'fit-')

            # print("\nMultinest is exploring the parameter space...\n")

            # Reset the likelihood values
            self._log_like_values = []

            sampler = pymultinest.run(self._func_wrapper,
                                      self._prior,
                                      n_dim,
                                      n_dim,
                                      outputfiles_basename=outputfiles_basename,
                                      n_live_points=max(100, n_dim * 20),
                                      multimodal=True,
                                      resume=False)

            # Use PyMULTINEST analyzer to gather parameter info

            # NOTE: I encapsulate this to avoid the output in the constructor of Analyzer

            with suppress_stdout():

                multinest_analyzer = pymultinest.analyse.Analyzer(n_params=n_dim,
                                                                  outputfiles_basename=outputfiles_basename)

            # Get the function value from the chain
            func_values = multinest_analyzer.get_equal_weighted_posterior()[:, -1]

            # Store the sample for further use (if needed)

            self._sampler = sampler

            # Get the samples from the sampler

            _raw_samples = multinest_analyzer.get_equal_weighted_posterior()[:, :-1]

        # Find the minimum of the function (i.e. the maximum of func_wrapper)

        idx = func_values.argmax()

        best_fit_values = _raw_samples[idx]

        minimum = func_values[idx] * (-1)

        if compute_covar:

            covariance_matrix = self._compute_covariance_matrix(best_fit_values)

        else:

            covariance_matrix = None

        self._store_fit_results(best_fit_values, minimum, covariance_matrix)

        return best_fit_values, minimum