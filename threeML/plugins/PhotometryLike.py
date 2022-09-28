import collections
import copy
from typing import Any, Dict, Optional, Union

import numpy as np
from speclite.filters import FilterResponse, FilterSequence
from threeML.config import threeML_config
from threeML.io.logging import setup_logger
from threeML.io.plotting.data_residual_plot import ResidualPlot
from threeML.plugins.XYLike import XYLike
from threeML.utils.photometry import FilterSet, PhotometericObservation

log = setup_logger(__name__)

__instrument_name = "Generic photometric data"


class BandNode:
    def __init__(self, name, index, value, mask):
        """
        Container class that allows for the shutting on and off of bands
        """
        self._name = name
        self._index = index
        self._mask = mask
        self._value = value

        self._on = True

    def _set_on(self, value=True):

        self._on = value

        self._mask[self._index] = self._on

    def _get_on(self):

        return self._on

    on = property(
        _get_on,
        _set_on,
        doc="Turn on or off the band. Use booleans, like: 'p.on = True' "
        " or 'p.on = False'. ",
    )

    # Define property "fix"

    def _set_off(self, value=True):

        self._on = not value

        self._mask[self._index] = self._on

    def _get_off(self):

        return not self._on

    off = property(
        _get_off,
        _set_off,
        doc="Turn on or off the band. Use booleans, like: 'p.off = True' "
        " or 'p.off = False'. ",
    )

    def __repr__(self):

        return f"on: {self._on}\nvalue: {self._value}"


class PhotometryLike(XYLike):
    def __init__(
        self,
        name: str,
        filters: Union[FilterSequence, FilterResponse],
        observation: PhotometericObservation,
    ):
        """
        The photometry plugin is desinged to fit optical/IR/UV photometric data from a given
        filter system. Filters are given in the form a speclite (http://speclite.readthedocs.io)
        FitlerResponse or FilterSequence objects. 3ML contains a vast number of filters via the SVO
        VO service: http://svo2.cab.inta-csic.es/svo/theory/fps/ and can be accessed via:

        from threeML.utils.photometry import get_photometric_filter_library

        filter_lib = get_photometric_filter_library()


        Bands can be turned on and off by setting


        plugin.band_<band name>.on = False/True
        plugin.band_<band name>.off = False/True


        :param name: plugin name
        :param filters: speclite filters
        :param observation: A PhotometricObservation instance
        """

        assert isinstance(
            observation, PhotometericObservation
        ), "Observation must be PhotometricObservation"

        # convert names so that only the filters are present
        # speclite uses '-' to separate instrument and filter

        if isinstance(filters, FilterSequence):

            # we have a filter sequence

            names = [fname.split("-")[1] for fname in filters.names]

        elif isinstance(filters, FilterResponse):

            # we have a filter response

            names = [filters.name.split("-")[1]]

            filters = FilterSequence([filters])

        else:

            log.error("filters must be A FilterResponse or a FilterSequence")

            RuntimeError("filters must be A FilterResponse or a FilterSequence")

        # since we may only have a few of the  filters in use
        # we will mask the filters not needed. The will stay fixed
        # during the life of the plugin

        if not observation.is_compatible_with_filter_set(filters):

            log.error("The data and filters are not congruent")

            raise AssertionError("The data and filters are not congruent")

        mask = observation.get_mask_from_filter_sequence(filters)

        if not mask.sum() > 0:

            log.error("There are no data in this observation!")

            raise AssertionError("There are no data in this observation!")

        # create a filter set and use only the bands that were specified

        self._filter_set = FilterSet(filters, mask)

        self._magnitudes = np.zeros(self._filter_set.n_bands)

        self._magnitude_errors = np.zeros(self._filter_set.n_bands)

        # we want to fill the magnitudes in the same order as the
        # the filters

        for i, band in enumerate(self._filter_set.filter_names):

            self._magnitudes[i] = observation[band][0]
            self._magnitude_errors[i] = observation[band][1]

        self._observation = observation

        # pass thru to XYLike

        super(PhotometryLike, self).__init__(
            name=name,
            x=self._filter_set.effective_wavelength,  # dummy x values
            y=self._magnitudes,
            yerr=self._magnitude_errors,
            poisson_data=False,
        )

        # now set up the mask zetting

        for i, band in enumerate(self._filter_set.filter_names):

            node = BandNode(
                band,
                i,
                (self._magnitudes[i], self._magnitude_errors[i]),
                self._mask,
            )

            setattr(self, f"band_{band}", node)

    @property
    def observation(self) -> PhotometericObservation:

        return self._observation

    @classmethod
    def from_kwargs(cls, name, filters, **kwargs):
        """
        Example:

        grond = PhotometryLike.from_kwargs('GROND',
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

        NOTE: PhotometryLike expects apparent AB magnitudes. Please calibrate your data to this system


        :param name: plugin name
        :param filters: speclite filters
        :param kwargs: keyword args of band name and tuple(mag, mag error)

        """

        return cls(name, filters, PhotometericObservation.from_kwargs(**kwargs))

    @classmethod
    def from_file(
        cls,
        name: str,
        filters: Union[FilterResponse, FilterSequence],
        file_name: str,
    ):
        """
        Create the a PhotometryLike plugin from a saved HDF5 data file

        :param name: plugin name
        :param filters: speclite filters
        :param file_name: name of the observation file


        """

        return cls(name, filters, PhotometericObservation.from_hdf5(file_name))

    @property
    def magnitudes(self):
        return self._magnitudes

    @property
    def magnitude_errors(self):
        return self._magnitude_errors

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

        return self._filter_set.ab_magnitudes()

    def display_filters(self):
        """
        display the filter transmission curves

        :return:
        """

        return self._filter_set.plot_filters()

    def plot(
        self,
        data_color: str = "r",
        model_color: str = "blue",
        show_data: bool = True,
        show_residuals: bool = True,
        show_legend: bool = True,
        model_label: Optional[str] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        data_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> ResidualPlot:

        """TODO describe function

        :param data_color:
        :type data_color: str
        :param model_color:
        :type model_color: str
        :param show_data:
        :type show_data: bool
        :param show_residuals:
        :type show_residuals: bool
        :param show_legend:
        :type show_legend: bool
        :param model_label:
        :type model_label: Optional[str]
        :param model_kwargs:
        :type model_kwargs: Optional[Dict[str, Any]]
        :param data_kwargs:
        :type data_kwargs: Optional[Dict[str, Any]]
        :returns:

        """

        _default_model_kwargs = dict(color=model_color, alpha=1)

        _sub_menu = threeML_config.plotting.residual_plot

        _default_data_kwargs = dict(
            color=data_color,
            alpha=1,
            fmt=_sub_menu.marker,
            markersize=_sub_menu.size,
            ls="",
            elinewidth=_sub_menu.linewidth,
            capsize=0,
        )

        # overwrite if these are in the confif

        _kwargs_menu = threeML_config.plugins.photo.fit_plot

        if _kwargs_menu.model_mpl_kwargs is not None:

            for k, v in _kwargs_menu.model_mpl_kwargs.items():

                _default_model_kwargs[k] = v

        if _kwargs_menu.data_mpl_kwargs is not None:

            for k, v in _kwargs_menu.data_mpl_kwargs.items():

                _default_data_kwargs[k] = v

        if model_kwargs is not None:

            if not type(model_kwargs) == dict:

                log.error("model_kwargs must be a dict")

                raise RuntimeError()

            for k, v in list(model_kwargs.items()):

                if k in _default_model_kwargs:

                    _default_model_kwargs[k] = v

                else:

                    _default_model_kwargs[k] = v

        if data_kwargs is not None:

            if not type(data_kwargs) == dict:

                log.error("data_kwargs must be a dict")

                raise RuntimeError()

            for k, v in list(data_kwargs.items()):

                if k in _default_data_kwargs:

                    _default_data_kwargs[k] = v

                else:

                    _default_data_kwargs[k] = v

        # since we define some defualts, lets not overwrite
        # the users

        _duplicates = (("ls", "linestyle"), ("lw", "linewidth"))

        for d in _duplicates:

            if (d[0] in _default_model_kwargs) and (
                d[1] in _default_model_kwargs
            ):

                _default_model_kwargs.pop(d[0])

            if (d[0] in _default_data_kwargs) and (
                d[1] in _default_data_kwargs
            ):

                _default_data_kwargs.pop(d[0])

        if model_label is None:
            model_label = f"{self._name} Model"

        avg_wave_length = (
            self._filter_set.effective_wavelength.value
        )  # type: np.ndarray

        # need to sort because filters are not always in order

        sort_idx = avg_wave_length.argsort()

        expected_model_magnitudes = self._get_total_expectation()[sort_idx]
        magnitudes = self.magnitudes[sort_idx]
        mag_errors = self.magnitude_errors[sort_idx]
        avg_wave_length = avg_wave_length[sort_idx]

        residuals = (expected_model_magnitudes - magnitudes) / mag_errors

        widths = self._filter_set.wavelength_bounds.widths[sort_idx]

        residual_plot = ResidualPlot(show_residuals=show_residuals, **kwargs)

        residual_plot.add_data(
            x=avg_wave_length,
            y=magnitudes,
            xerr=widths / 2.0,
            yerr=mag_errors,
            residuals=residuals,
            label=self._name,
            show_data=show_data,
            **_default_data_kwargs,
        )

        residual_plot.add_model(
            avg_wave_length,
            expected_model_magnitudes,
            label=model_label,
            **_default_model_kwargs,
        )

        return residual_plot.finalize(
            xlabel=f"Wavelength\n({self._filter_set.waveunits})",
            ylabel="Magnitudes",
            xscale="linear",
            yscale="linear",
            invert_y=True,
            show_legend=show_legend
        )

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

        new_observation = PhotometericObservation.from_dict(bands)

        new_photo = PhotometryLike(
            name,
            filters=self._filter_set.speclite_filters,
            observation=new_observation,
        )

        # apply the current mask

        new_photo._mask = copy.copy(self._mask)

        return new_photo
