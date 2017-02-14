from threeML.plugin_prototype import PluginPrototype
import numpy as np

from astromodels import Model, PointSource

from threeML.plugins.OGIP.likelihood_functions import poisson_log_likelihood_ideal_bkg
from threeML.plugins.OGIP.likelihood_functions import chi2
from threeML.classicMLE.joint_likelihood import JointLikelihood
from threeML.data_list import DataList


__instrument_name = "n.a."


class XYLike(PluginPrototype):

    def __init__(self, name, x, y, yerr=None, poisson_data=False):

        nuisance_parameters = {}

        super(XYLike, self).__init__(name, nuisance_parameters)

        # Make x and y always arrays so we can handle them always in the same way
        # even if they have only one element

        self._x = np.array(x, ndmin=1)
        self._y = np.array(y, ndmin=1)

        # If there are specified errors, use those (assume Gaussian statistic)
        # otherwise make sure that the user specified poisson_error = True and use
        # Poisson statistic

        if yerr is not None:

            self._yerr = np.array(yerr, ndmin=1)

            assert np.all(self._yerr > 0), "Errors cannot be negative or zero."

            print("Using chi2 statistic with the provided errors.")

            self._is_poisson = False

            self._has_errors = True

        elif not poisson_data:

            # JM: I'm leaving the possibility to do an unweighted fit... we can remove if this is a
            #     a bad idea

            #assert poisson_data, "You did not provide errors and you did not set poisson_error to True."

            self._yerr = np.ones_like(self._y)

            self._is_poisson = False

            self._has_errors = False

            print("Using unweighted chi2 statistic.")

        else:

            print("Using Poisson log-likelihood")

            self._is_poisson = True
            self._yerr = None
            self._has_errors = True


    @property
    def is_poisson(self):

        return self._is_poisson

    @property
    def has_errors(self):

        return self._has_errors



    def set_model(self, likelihood_model_instance):
        """
        Set the model to be used in the joint minimization. Must be a LikelihoodModel instance.

        :param likelihood_model_instance: instance of Model
        :type likelihood_model_instance: astromodels.Model
        """

        assert likelihood_model_instance.get_number_of_extended_sources() == 0, "Extended sources are not supported by " \
                                                                             "XYLike plugin"

        assert likelihood_model_instance.get_number_of_point_sources() > 0, "You have to have at least one point source"

        self._likelihood_model = likelihood_model_instance

    def get_log_like(self):
        """
        Return the value of the log-likelihood with the current values for the
        parameters
        """

        n_point_sources = self._likelihood_model.get_number_of_point_sources()

        # Make a function which will stack all point sources (XYLike do not support spatial dimension)

        expectation = np.sum(map(lambda source: source(self._x),
                             self._likelihood_model.point_sources.values()),
                             axis=0)

        if self._is_poisson:

            # Poisson log-likelihood

            return np.sum(poisson_log_likelihood_ideal_bkg(self._y, np.zeros_like(self._y), expectation))

        else:

            # Chi squared
            chi2_ = chi2(self._y, self._yerr, expectation)

            assert np.all(np.isfinite(chi2_))

            return np.sum(chi2_) * (-1)

    def inner_fit(self):
        """
        This is used for the profile likelihood. Keeping fixed all parameters in the
        LikelihoodModel, this method minimize the logLike over the remaining nuisance
        parameters, i.e., the parameters belonging only to the model for this
        particular detector. If there are no nuisance parameters, simply return the
        logLike value.
        """

        return self.get_log_like()

    def get_model_flux(self):

        pass


    def fit(self, function, minimizer='minuit'):
        """
        Fit the data with the provided function (an astromodels function)

        :param function: astromodels function
        :param minimizer: the minimizer to use
        :return: best fit results
        """

        # This is a wrapper to give an easier way to fit simple data without having to go through the definition
        # of sources
        pts = PointSource("fake", 0.0, 0.0, function)

        model = Model(pts)

        jl = JointLikelihood(model, DataList(self), verbose=False)

        jl.set_minimizer(minimizer)

        return jl.fit()

