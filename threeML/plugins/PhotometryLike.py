import numpy as np

from threeML.plugins.XYLike import XYLike
from threeML.plugins.photometry.photometric_data import PhotometryData
from threeML.plugins.photometry.filter_set import FilterSet


class PhotometryLike(XYLike):
    def __init__(self, name, photometry_data, filter_set, verbose=True):
        # type: (str, PhotometryData, FilterSet, bool) -> object

        assert isinstance(photometry_data, PhotometryData), 'data must be of type PhotometryData'
        assert isinstance(filter_set, FilterSet), 'filters must be of type FilterSet'

        assert photometry_data.n_bands == filter_set.n_bands, 'number of data filters does not equal number of filters'

        # make sure that the filter names are in order and the same
        assert np.all(
            photometry_data.filter_names == filter_set.filter_names), 'filter names do not match or are not in order'

        self._mask = np.ones(photometry_data.n_bands, dtype=bool)

        self._photometry_data = photometry_data
        self._filter_set = filter_set

        # pass thru to XYLike

        super(PhotometryLike, self).__init__(name=name,
                                             x=filter_set.average_wavelength, # dummy x values
                                             y=photometry_data.magnitudes,
                                             yerr=photometry_data.magnitude_errors,
                                             poisson_data=False)


    def set_active_filters(self, *filter_names):
        """
        set the active filters to be used in the fit
        :param filter_names: filter names ot be set active
        :return:
        """

        # scroll through the known filter names

        for i, name in enumerate(self._filter_set.filter_names):

            for select_name in filter_names:

                # if one of the filters is hit, then activate it

                if name == select_name:
                    self._mask[i] = True

        print("Now using %d of %d filters:\n\tActive Filters: %s", (sum(self._mask),
                                                                    len(self._mask),
                                                                    ', '.join(
                                                                        self._filter_set.filter_names[self._mask])))

        # reconstruct the plugin with selected data

        super(PhotometryLike, self).__init__(name=self.name,
                                             x=self._filter_set.average_wavelength[self._mask],  # dummy x values
                                             y=self._photometry_data.magnitudes[self._mask],
                                             yerr=self._photometry_data.magnitude_errors[self._mask],
                                             poisson_data=False)

    def set_inactive_filters(self, *filter_names):
        """
        set filters to be excluded from the fit
        :param filter_names: filter names ot be set inactive
        :return:
        """

        # scroll through the known filter names

        for i, name in enumerate(self._filter_set.filter_names):

            for select_name in filter_names:

                if name == select_name:
                    self._mask[i] = False

        print("Now using %d of %d filters:\n\tActive Filters: %s", (sum(self._mask),
                                                                    len(self._mask),
                                                                    ', '.join(
                                                                        self._filter_set.filter_names[self._mask])))

        # reconstruct the plugin with selected data

        super(PhotometryLike, self).__init__(name=self.name,
                                             x=self._filter_set.average_wavelength[self._mask],
                                             y=self._photometry_data.magnitudes[self._mask],
                                             yerr=self._photometry_data.magnitude_errors[self._mask],
                                             poisson_data=False)

    def set_model(self, likelihood_model):

        super(PhotometryLike, self).set_model(likelihood_model)

        n_point_sources = self._likelihood_model.get_number_of_point_sources()

        def differential_flux(energies):

            for i in xrange(n_point_sources):
                fluxes = self._likelihood_model.get_point_source_fluxes(0, energies)

                # If we have only one point source, this will never be executed
                for i in range(1, n_point_sources):
                    fluxes += self._likelihood_model.get_point_source_fluxes(i, energies)

            return fluxes

        self._filter_set.set_model(differential_flux)

    def _get_expectation(self):

        return self._filter_set.effective_stimulus()

    def display_filters(self):
        """
        display the filter transmission curves

        :return:
        """

        return self._filter_set.plot_filters()