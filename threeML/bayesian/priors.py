import abc
import math

import numpy as np
import scipy.stats

UNDEFINED = - np.inf


class Prior(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, prior_name):
        self._prior_name = prior_name

    @property
    def name(self):
        """Return the name of the function"""

        return self._prior_name

    # This is used only by children
    def __set_name(self, new_name):
        self._prior_name = new_name

    @abc.abstractmethod
    def __call__(self, value):
        """
        Return the logarithm of the prior pdf at the given value
        """
        pass


class UniformPrior(Prior):
    def __init__(self, min_value, max_value):
        """
        A uniform (constant) prior. It is constant between min_value and max_value, and undefined outside.

        :param min_value: lower bound for the range where the prior is defined
        :param max_value: upper bound for the range where the prior is defined
        :return:
        """

        # Verify that min_value < max_value

        assert min_value < max_value, "Minimum must be smaller than maximum"

        # Store boundaries

        self._min_value = min_value
        self._max_value = max_value

        # Init parent class

        super(UniformPrior, self).__init__("UniformPrior")

    def _get_min_value(self):
        """
        :return: the lower bound of the range where the prior is defined
        """

        return self._min_value

    def _set_min_value(self, new_minimum):
        """
        Set the lower bound of the range where the prior is defined

        :return: (none)
        """

        assert new_minimum < self._max_value, ("Minimum must be smaller than maximum. You can change both "
                                               "at the same time by using the set_bounds() method.")

        self._min_value = new_minimum

    min_value = property(_get_min_value, _set_min_value,
                         doc='Set or get the lower bound of the range where the prior is defined')

    def _get_max_value(self):
        """
        :return: the upper bound of the range where the prior is defined
        """

        return self._max_value

    def _set_max_value(self, new_maximum):
        """
        Set the upper bound of the range where the prior is defined

        :return: (none)
        """

        assert new_maximum > self._min_value, ("Maximum must be larger than minimum. You can change both "
                                               "at the same time by using the set_bounds() method.")

        self._max_value = new_maximum

    max_value = property(_get_max_value, _set_max_value,
                         doc='Set or get the upper bound of the range where the prior is defined')

    def set_bounds(self, new_min_value, new_max_value):
        """
        Set the upper and lower bound of the range where the prior is defined

        :param new_min_value: lower bound
        :param new_max_value: upper bound
        :return: (none)
        """

        self._min_value = new_min_value
        self._max_value = new_max_value

    def __call__(self, value):

        if self.min_value < value < self.max_value:

            return 0.0

        else:

            return UNDEFINED

    def multinest_call(self, cube):

        return cube * (self._max_value - self._min_value) + self._min_value


class LogUniformPrior(UniformPrior):
    def __init__(self, min_value, max_value):
        """
        A log-uniform prior:

        f(x) = log(1 / x)

        It is defined between min_value and max_value, and undefined outside.

        :param min_value: lower bound for the range where the prior is defined
        :param max_value: upper bound for the range where the prior is defined
        :return:
        """

        # Init parent class

        super(LogUniformPrior, self).__init__(min_value, max_value)

        # Update the name

        self.__set_name("LogUniformPrior")

    # Override the __call__ method of the UniformPrior class

    def __call__(self, value):

        if self._min_value < value < self._max_value and value > 0:

            # This is = log(1/value)

            return -math.log10(value)

        else:

            return UNDEFINED

    def multinest_call(self, cube):

        decades = math.log10(self._max_value) - math.log10(self._min_value)

        start_decade = math.log10(self._min_value)

        return 10 ** ((cube * decades) + start_decade)


class GaussianPrior(Prior):
    def __init__(self, mu, sigma):
        """
        A Gaussian prior centered in mu with standard deviation sigma.

        Note that the value of the prior too far from the center will be a special value which means "undefined",
        to avoid numerical errors in Bayesian samplers.

        :param mu: center of the Gaussian
        :param sigma: standard deviation of the Gaussian
        :return:
        """

        self._mu = float(mu)
        self._sigma = abs(float(sigma))

        self._update_pdf()

        # Init parent class

        super(GaussianPrior, self).__init__("GaussianPrior")

    def _update_pdf(self):

        # Update the PDF with the current sigma and mu

        self._norm = scipy.stats.norm(loc=self._mu, scale=self._sigma)

    def _set_mu(self, new_mu):
        """
        Set a new center for the Gaussian
        :param new_mu: new center
        :return: (none)
        """

        self._mu = new_mu

        self._update_pdf()

    def _get_mu(self):
        """
        :return: get the center of the Gaussian
        """

        return self._mu

    mu = property(_get_mu, _set_mu, doc="Get or set the center of the Gaussian")

    def _set_sigma(self, new_sigma):
        """
        Set a new sigma for the Gaussian
        :param new_sigma: new sigma
        :return: (none)
        """

        self._sigma = new_sigma

        self._update_pdf()

    def _get_sigma(self):
        """
        :return: get the standard deviation of the Gaussian
        """

        return self._sigma

    sigma = property(_get_sigma, _set_sigma, doc="Get or set the standard deviation of the Gaussian")

    def __call__(self, x):

        # Here I multiply by sigma because of scipy "scale" behavior
        # (see scipy documentation)

        value = self._norm.pdf(x) * self._sigma

        # Truncate the value of the Gaussian, to avoid numerical errors, i.e., return UNDEFINED if the value of
        # the prior is too small

        if value < 1e-15:

            return UNDEFINED

        else:

            return math.log10(value)
