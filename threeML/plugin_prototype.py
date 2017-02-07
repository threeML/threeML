"""
Define the interface for a plugin class.
"""

import abc
from astromodels.utils.valid_variable import is_valid_variable_name
import warnings
import functools


def set_external_property(method):
    """
    Sets external property values if they exist


    :param method:
    :return:
    """

    @functools.wraps(method)
    def wrapper(instance, *args, **kwargs):

        if instance._external_properties:

            for property, value in instance._external_properties:

                property.value = value

        return method(instance, *args, **kwargs)

    return wrapper


class PluginPrototype(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, name, nuisance_parameters):

        assert is_valid_variable_name(name), "The name %s cannot be used as a name. You need to use a valid " \
                                             "python identifier: no spaces, cannot start with numbers, cannot contain " \
                                             "operators symbols such as -, +, *, /" % name

        # Make sure total is not used as a name (need to use it for other things, like the total value of the statistic)
        assert name.lower() != "total", "Sorry, you cannot use 'total' as name for a plugin."

        self._name = name

        # This is just to make sure that the plugin is legal

        assert isinstance(nuisance_parameters, dict)

        self._nuisance_parameters = nuisance_parameters

        # These are the external properties (time, polarization, etc.)
        self._external_properties = []

    def get_name(self):

        warnings.warn("Do not use get_name() for plugins, use the .name property", DeprecationWarning)

        return self.name

    @property
    def name(self):
        """
        Returns the name of this instance

        :return: a string (this is enforced to be a valid python identifier)
        """
        return self._name

    @property
    def nuisance_parameters(self):
        """
        Returns a dictionary containing the nuisance parameters for this dataset

        :return: a dictionary
        """

        return self._nuisance_parameters

    def update_nuisance_parameters(self, new_nuisance_parameters):

        assert isinstance(new_nuisance_parameters, dict)

        self._nuisance_parameters = new_nuisance_parameters

    def external_property(self, property, value):
        """
        Set external/auxiliary properties and their value
        :param property: an astromodels auxiliary variable
        :param value: the value of the auxiliary variable for this plugin
        :return:
        """

        self._external_properties.append((property, value))

    ######################################################################
    # The following methods must be implemented by each plugin
    ######################################################################

    @abc.abstractmethod
    def set_model(self, likelihood_model_instance):
        """
        Set the model to be used in the joint minimization. Must be a LikelihoodModel instance.
        """
        pass

    @abc.abstractmethod
    def get_log_like(self):
        """
        Return the value of the log-likelihood with the current values for the
        parameters
        """
        pass

    @abc.abstractmethod
    def inner_fit(self):
        """
        This is used for the profile likelihood. Keeping fixed all parameters in the
        LikelihoodModel, this method minimize the logLike over the remaining nuisance
        parameters, i.e., the parameters belonging only to the model for this
        particular detector. If there are no nuisance parameters, simply return the
        logLike value.
        """
        pass
