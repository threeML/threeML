from __future__ import division
from builtins import zip
from builtins import object
from past.utils import old_div
import speclite.filters as spec_filters
import astropy.units as astro_units
import numpy as np
import astropy.constants as constants

from threeML.utils.interval import IntervalSet


class NotASpeclikeFilter(RuntimeError):
    pass


class FilterSet(object):
    def __init__(self, filter, mask=None):
        """
        This class handles the optical filter functionality. It is build around speclite:
        http://speclite.readthedocs.io/en/latest/

        It accepts speclite fitlerresponse or sequences, allowing for full customization
        of the fitlers.



        :param filter: a speclite FitlerResponse or FilterSequence
        :param mask: an initial mask on the filters (bool array) that remains fixed
        """

        # we explicitly violate duck typing here in order to have one routine
        # to return values from the filters (speclite appends 's' to the end of sequence calls)

        if isinstance(filter, spec_filters.FilterResponse):

            # we will make a sequence

            self._filters = spec_filters.FilterSequence([filter])

        elif isinstance(filter, spec_filters.FilterSequence):

            self._filters = filter  # type: spec_filters.FilterSequence

        else:

            raise NotASpeclikeFilter(
                "filter must be a speclite FilterResponse or FilterSequence"
            )

        if mask is not None:

            tmp = []

            for condition, response in zip(mask, self._filters):

                if condition:

                    tmp.append(response)

            self._filters = spec_filters.FilterSequence(tmp)
            self._names = np.array([name.split("-")[1] for name in self._filters.names])
            self._long_name = self._filters.names

        # haven't set a likelihood model yet
        self._model_set = False

        # calculate the FWHM

        self._calculate_fwhm()

    @property
    def wavelength_bounds(self):
        """
        IntervalSet of FWHM bounds of the filters
        :return:
        """

        return self._wavebounds

    def _calculate_fwhm(self):
        """
        calculate the FWHM of the filters
        :return:
        """

        wmin = []
        wmax = []

        # go through each filter
        # and find the non-gaussian FWHM bounds

        for filter in self._filters:

            response = filter.response
            max_response = response.max()
            idx_max = response.argmax()
            half_max = 0.5 * max_response

            idx1 = abs(response[:idx_max] - half_max).argmin()

            idx2 = abs(response[idx_max:] - half_max).argmin() + idx_max

            # have to grab the private member here
            # bc the library does not expose it!

            w1 = filter._wavelength[idx1]
            w2 = filter._wavelength[idx2]

            wmin.append(w1)
            wmax.append(w2)

        self._wavebounds = IntervalSet.from_starts_and_stops(wmin, wmax)

    def set_model(self, differential_flux):
        """
        set the model of that will be used during the convolution. Not that speclite
        considers a differential flux to be in units of erg/s/cm2/lambda so we must convert
        astromodels into the proper units (using astropy units!)


        """

        conversion_factor = (constants.c ** 2 * constants.h ** 2).to("keV2 * cm2")

        def wrapped_model(x):
            return old_div(differential_flux(x) * conversion_factor, x ** 3)

        self._wrapped_model = wrapped_model

        self._model_set = True

    def ab_magnitudes(self):
        """
        return the effective stimulus of the model and filter for the given
        magnitude system
        :return: np.ndarray of ab magnitudes
        """

        assert self._model_set, "no likelihood model has been set"

        # speclite has issues with unit conversion
        # so we will do the calculation manually here

        ratio = []

        for filter in self._filters:

            # first get the flux and convert it to base units
            synthetic_flux = filter.convolve_with_function(self._wrapped_model).to(
                "1/(cm2 s)"
            )

            # normalize it to the filter's AB magnitude

            ratio.append(
                (old_div(synthetic_flux, filter.ab_zeropoint.to("1/(cm2 s)"))).value
            )

        ratio = np.array(ratio)

        return -2.5 * np.log10(ratio)

        # return self._filters.get_ab_magnitudes(self._wrapped_model).to_pandas().loc[0]

    def plot_filters(self):
        """
        plot the filter/ transmission curves
        :return: fig
        """

        spec_filters.plot_filters(self._filters)

    @property
    def n_bands(self):
        """

        :return: the number of bands
        """

        return len(self._filters.names)

    @property
    def filter_names(self):
        """

        :return: the filter names
        """

        return self._names

    @property
    def native_filter_names(self):
        """
        the native filter names
        :return:
        """

        return self._filters.names

    @property
    def speclite_filters(self):
        """
        exposes the speclite fitlers for simulations

        :return:
        """

        return self._filters

    @property
    def effective_wavelength(self):
        """

        :return: the average wave length of the filters
        """

        return self._filters.effective_wavelengths

    @property
    def waveunits(self):
        """

        :return: the pysynphot wave units
        """

        return astro_units.Angstrom
