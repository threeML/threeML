import speclite.filters as spec_filters
import astropy.units as astro_units
import numpy as np
import astropy.constants as constants

from threeML.utils.interval import IntervalSet


conversion_factor=(constants.c**2 * constants.h**2).to('keV2 * cm2')


class  NotASpeclikeFilter(RuntimeError):
    pass


class FilterSet(object):
    def __init__(self, filter, mask=None):
        """

        :param filter_names: array of filter names
        :param wave_lengths: the wave lengths for each transmission curve
        :param transmission_curves: the transmission or throughput curves
        :param magnitude_systems: the magnitude system for each transmission curve
        :param wavesunits: the units of the wave length. see pysynphot for appropriate units http://pysynphot.readthedocs.io/en/latest/units.html?
        """

        if isinstance(filter,spec_filters.FilterResponse):

            # we will make a sequence

            self._filters = spec_filters.FilterSequence([filter])

        elif isinstance(filter,spec_filters.FilterSequence):

            self._filters = filter  # type: spec_filters.FilterSequence

        else:

            raise NotASpeclikeFilter('filter must be a speclite FilterResponse or FilterSequence')


        if mask is not None:

            tmp = []

            for condition, response in zip(mask, self._filters):

                if condition:

                    tmp.append(response)


            self._filters = spec_filters.FilterSequence(tmp)

            self._names = np.array([name.split('-')[1] for name in self._filters.names])
            self._long_name = self._filters.names





        self._model_set = False

        # calculate the FWHM

        self._calculate_fwhm()


    @property
    def wavelength_bounds(self):

        return self._wavebounds

    def _calculate_fwhm(self):
        """
        calculate the FWHM of the filters
        :return:
        """

        wmin = []
        wmax = []



        for filter in self._filters:

            response = filter.response
            max_response =  response.max()
            idx_max = response.argmax()
            half_max = 0.5 * max_response

            idx1 = abs(response[:idx_max] -
                         half_max).argmin()

            idx2 = abs(response[idx_max:] -
                         half_max).argmin() + idx_max

            # have to grab the private member here
            # bc the library does not expose it!

            w1 = filter._wavelength[idx1]
            w2 = filter._wavelength[idx2]


            wmin.append(w1)
            wmax.append(w2)

        self._wavebounds = IntervalSet.from_starts_and_stops(wmin,wmax)


    def set_model(self, differential_flux):
        """
        Wrap astromodels model into a pysynphot model and assign it to an observation
        """

        def wrapped_model(x):
            return differential_flux(x) * conversion_factor / x ** 3



        self._wrapped_model = wrapped_model

        self._model_set = True

    def ab_magnitudes(self):
        """
        return the effective stimulus of the model and filter for the given
        magnitude system
        :return: np.ndarray of ab magnitudes
        """

        assert self._model_set, 'no likelihood model has been set'


        return self._filters.get_ab_magnitudes(self._wrapped_model).to_pandas().loc[0]



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
