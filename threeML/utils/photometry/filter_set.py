from __future__ import division

from builtins import object, zip

import astropy.constants as constants
import astropy.units as astro_units
import numba as nb
import numpy as np
import speclite.filters as spec_filters
from past.utils import old_div

from threeML.utils.interval import IntervalSet

_final_convert = (1. * astro_units.cm**2 * astro_units.keV / (astro_units.erg *
                                                              astro_units.angstrom**2 * astro_units.s * astro_units.cm**2)).to("1/(cm2 s)").value

_hc_constant = (constants.h * constants.c).to(
    astro_units.erg * astro_units.angstrom).value


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
            self._names = np.array([name.split("-")[1]
                                    for name in self._filters.names])
            self._long_name = self._filters.names

        # haven't set a likelihood model yet
        self._model_set = False

        self._n_filters = len(self._filters)

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

        conversion_factor = (constants.c ** 2 *
                             constants.h ** 2).to("keV2 * cm2")

        self._zero_points = np.empty(self._n_filters)
        self._wavelengths = []
        self._energies = []
        self._response = []
        self._factors = []
        self._n_terms = []

        for i, filter in enumerate(self._filters):

            # precompute the zeropoints
            self._zero_points[i] = filter.ab_zeropoint.to("1/(cm2 s)").value

            # save the wavelenghts
            self._wavelengths.append(filter.wavelength)

            # we are going to input things in to the astromodels
            # funtion as keV and convert back later
            self._energies.append(
                (filter.wavelength * astro_units.angstrom).to("keV", equivalencies=astro_units.spectral()).value)

            self._factors.append(
                (conversion_factor / ((filter.wavelength * astro_units.angstrom) ** 3)).value)

            self._response.append(filter.response)
            self._n_terms.append(len(filter.wavelength))

        self._differential_flux = differential_flux

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

        out = []

        for i in range(self._n_filters):

            out.append(_conolve_and_convert(self._differential_flux(self._energies[i]),
                                            self._factors[i],
                                            self._response[i],
                                            self._wavelengths[i],
                                            self._zero_points[i],
                                            self._n_terms[i])

                       )

        return np.array(out)

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


@nb.njit(fastmath=True)
def _conolve_and_convert(diff_flux, factor, response, wavelength, zero_point, N):

    for n in range(N):

        diff_flux[n] *= factor[n] * response[n] * wavelength[n] / _hc_constant

    # this will be in some funky units so we convert to 1/ cm2 s
    synthetic_flux = np.trapz(diff_flux, wavelength) * _final_convert

    ratio = synthetic_flux / zero_point

    return -2.5 * np.log10(ratio)
