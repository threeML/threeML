import numpy as np

from threeML.plugins.XYLike import XYLike
from threeML.utils.histogram import Histogram
from threeML.plugins.OGIP.likelihood_functions import poisson_log_likelihood_ideal_bkg
from threeML.plugins.OGIP.likelihood_functions import chi2


__instrument_name = "n.a."

class HistLike(XYLike):

    def __init__(self,name, histogram):
        """
        Fit a 3ML Histogram such that the model is evaluated its integral over the histogram bins

        :param name: plugin name
        :param histogram: 3ML histogram
        """


        assert isinstance(histogram,Histogram), "input must be a 3ML histogram"

        self._histogram = histogram #type: Histogram


        super(HistLike, self).__init__(name=name,
                                       x=self._histogram.mid_points,
                                       y=self._histogram.contents,
                                       yerr=self._histogram.errors,
                                       poisson_data=self._histogram.is_poisson)


    def _get_diff_flux_and_integral(self):

        n_point_sources = self._likelihood_model.get_number_of_point_sources()

        # Make a function which will stack all point sources (HISTLike does not support spatial dimension)

        def differential_flux(energies):
            fluxes = self._likelihood_model.get_point_source_fluxes(0, energies)

            # If we have only one point source, this will never be executed
            for i in range(1, n_point_sources):
                fluxes += self._likelihood_model.get_point_source_fluxes(i, energies)

            return fluxes

        # The following integrates the diffFlux function using Simpson's rule
        # This assume that the intervals e1,e2 are all small, which is guaranteed
        # for any reasonable response matrix, given that e1 and e2 are Monte-Carlo
        # energies. It also assumes that the function is smooth in the interval
        # e1 - e2 and twice-differentiable, again reasonable on small intervals for
        # decent models. It might fail for models with too sharp features, smaller
        # than the size of the monte carlo interval.

        def integral(e1, e2):
            # Simpson's rule

            return (e2 - e1) / 6.0 * (differential_flux(e1)
                                      + 4 * differential_flux((e1 + e2) / 2.0)
                                      + differential_flux(e2))

        return differential_flux, integral

    def get_log_like(self):
        """
        Return the value of the log-likelihood with the current values for the
        parameters
        """

        # Make a function which will stack all point sources (XYLike do not support spatial dimension)

        expectation = self.get_model_flux()

        if self._is_poisson:

            # Poisson log-likelihood

            return np.sum(poisson_log_likelihood_ideal_bkg(self._y, np.zeros_like(self._y), expectation))

        else:

            # Chi squared

            chi2_ = chi2(self._y, self._yerr, expectation)

            assert np.all(np.isfinite(chi2_))

            return np.sum(chi2_) * (-1)

    def get_model_flux(self):

        _, integral_function = self._get_diff_flux_and_integral()


        model = np.array([integral_function(xmin, xmax) for xmin, xmax in self._histogram.bin_stack])

        return model



    @property
    def histogram(self):

        return self._histogram