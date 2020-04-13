from builtins import range
import collections
import copy

import numpy as np

from threeML.plugins.XYLike import XYLike
from threeML.utils.photometry.filter_set import FilterSet

__instrument_name = "Generic photometric data"


class PhotometryLike(XYLike):
    def __init__(self, name, filters, **data):
        """
        The photometry plugin is desinged to fit optical/IR/UV photometric data from a given
        filter system. Filters are given in the form a speclite (http://speclite.readthedocs.io)
        FitlerResponse or FilterSequence objects. 3ML contains a vast number of filters via the SVO
        VO service: http://svo2.cab.inta-csic.es/svo/theory/fps/ and can be accessed via:

        from threeML.plugins.photometry.filter_library import threeML_filter_library

        One can also construct their own filters with speclite.

        Example:

        grond = PhotometryLike('GROND',
                       filters=threeML_filter_library.ESO.GROND,
                       g=(20.93,.23),
                       r=(20.6,0.12),
                       i=(20.4,.07),
                       z=(20.3,.04),
                       J=(20.0,.03),
                       H=(19.8,.03),
                       K=(19.7,.04))


        Magnitudes and errors are entered as keyword arguments where the key is the filter name and
        the argument is a tuple containing the data. You can exclude data for individual filters and
        they will be ignored during the fit.

        NOTE: PhotometryLike expects apparent AM magnitudes. Please calibrate your data to this system


        :param name: plugin name
        :param filters: speclite filters
        :param data: keyword args of band name and tuple(mag, mag error)
        """

        # convert names so that only the filters are present
        # speclite uses '-' to separate instrument and filter

        try:

            # we have a filter sequence

            names = [fname.split("-")[1] for fname in filters.names]

        except (AttributeError):

            # we have a filter response

            names = [filters.name.split("-")[1]]

        # since we may only have a few of the  filters in use
        # we will mask the filters not needed. The will stay fixed
        # during the life of the plugin

        starting_mask = np.zeros(len(names), dtype=bool)

        for band in list(data.keys()):

            assert band in names, "band %s is not a member of the filter set %s" % (
                band,
                "blah",
            )
            starting_mask[names.index(band)] = True

        # create a filter set and use only the bands that were specified

        self._filter_set = FilterSet(filters, starting_mask)

        self._magnitudes = np.zeros(self._filter_set.n_bands)

        self._magnitude_errors = np.zeros(self._filter_set.n_bands)

        # we want to fill the magnitudes in the same order as the
        # the filters

        for i, band in enumerate(self._filter_set.filter_names):

            self._magnitudes[i] = data[band][0]
            self._magnitude_errors[i] = data[band][1]

        # pass thru to XYLike

        super(PhotometryLike, self).__init__(
            name=name,
            x=self._filter_set.effective_wavelength,  # dummy x values
            y=self._magnitudes,
            yerr=self._magnitude_errors,
            poisson_data=False,
        )

    @property
    def magnitudes(self):
        return self._magnitudes

    @property
    def magnitude_errors(self):
        return self._magnitude_errors

    # def set_active_filters(self, *filter_names):
    #     """
    #     set the active filters to be used in the fit
    #     :param filter_names: filter names ot be set active
    #     :return:
    #     """
    #
    #     # scroll through the known filter names
    #
    #     for i, name in enumerate(self._filter_set.filter_names):
    #
    #         for select_name in filter_names:
    #
    #             # if one of the filters is hit, then activate it
    #
    #             if name == select_name:
    #                 self._mask[i] = True
    #
    #
    #     print("Now using %d of %d filters:\n\tActive Filters: %s", (sum(self._mask),
    #                                                                 len(self._mask),
    #                                                                 ', '.join(
    #                                                                     self._filter_set.filter_names[self._mask])))
    #
    #     # reconstruct the plugin with selected data
    #
    #     super(PhotometryLike, self).__init__(name=self.name,
    #                                          x=self._filter_set.effective_wavelength[self._mask],  # dummy x values
    #                                          y=self._magnitudes[self._mask],
    #                                          yerr=self._magnitude_errors[self._mask],
    #                                          poisson_data=False)
    #
    # def set_inactive_filters(self, *filter_names):
    #     """
    #     set filters to be excluded from the fit
    #     :param filter_names: filter names ot be set inactive
    #     :return:
    #     """
    #
    #     # scroll through the known filter names
    #
    #
    #     for i, name in enumerate(self._filter_set.filter_names):
    #
    #         for select_name in filter_names:
    #
    #             if name == select_name:
    #                 self._mask[i] = False
    #
    #     print("Now using %d of %d filters:\n\tActive Filters: %s", (sum(self._mask),
    #                                                                 len(self._mask),
    #                                                                 ', '.join(
    #                                                                     self._filter_set.filter_names[self._mask])))
    #
    #     # reconstruct the plugin with selected data
    #
    #     super(PhotometryLike, self).__init__(name=self.name,
    #                                          x=self._filter_set.effective_wavelength[self._mask],  # dummy x values
    #                                          y=self._magnitudes[self._mask],
    #                                          yerr=self._magnitude_errors[self._mask],
    #                                          poisson_data=False)

    def set_model(self, likelihood_model):
        """
        set the likelihood model
        :param likelihood_model:
        :return:
        """

        super(PhotometryLike, self).set_model(likelihood_model)

        n_point_sources = self._likelihood_model.get_number_of_point_sources()

        # sum up the differential

        def differential_flux(energies):

            fluxes = self._likelihood_model.get_point_source_fluxes(
                0, energies, tag=self._tag
            )

            # If we have only one point source, this will never be executed
            for i in range(1, n_point_sources):

                fluxes += self._likelihood_model.get_point_source_fluxes(
                    i, energies, tag=self._tag
                )

            return fluxes

        self._filter_set.set_model(differential_flux)

    def _get_total_expectation(self):

        return self._filter_set.ab_magnitudes()[self._mask]  # .as_matrix()

    def display_filters(self):
        """
        display the filter transmission curves

        :return:
        """

        return self._filter_set.plot_filters()

    def _new_plugin(self, name, x, y, yerr):
        """
        construct a new PhotometryLike plugin. allows for returning a new plugin
        from simulated data set while customizing the constructor
        further down the inheritance tree

        :param name: new name
        :param x: new x
        :param y: new y
        :param yerr: new yerr
        :return: new XYLike


        """

        bands = collections.OrderedDict()

        for i, band in enumerate(self._filter_set.filter_names):

            bands[band] = (y[i], yerr[i])

        new_photo = PhotometryLike(
            name, filters=self._filter_set.speclite_filters, **bands
        )

        # apply the current mask

        new_photo._mask = copy.copy(self._mask)

        return new_photo
