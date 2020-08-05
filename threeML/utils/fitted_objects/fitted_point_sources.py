from __future__ import division
from builtins import range
from past.utils import old_div
from builtins import object

__author__ = "grburgess"

from astropy import units as u
import numpy as np
import scipy.integrate as integrate
import collections


from threeML.utils.fitted_objects.fitted_source_handler import (
    GenericFittedSourceHandler,
)


class NotCompositeModelError(RuntimeError):
    pass


class InvalidUnitError(RuntimeError):
    pass


class FluxConversion(object):
    def __init__(self, flux_unit, energy_unit, flux_model):
        """
        a generic flux conversion class to handle transforming spectra
        between different flux units
        :param flux_unit: the desired flux unit
        :param energy_unit: the energy unit
        :param flux_model: the model to be transformed
        """

        self._flux_unit = flux_unit

        self._energy_unit = energy_unit

        self._model = flux_model

        self._test_value = 1.0 * energy_unit

        self._flux_type = None

        self._determine_quantity()

        self._calculate_conversion()

    def _determine_quantity(self):

        # scroll thru conversions until one works

        for k, v in self._flux_lookup.items():

            try:

                self._flux_unit.to(v)

                self._flux_type = k

            except (u.UnitConversionError):

                continue

        if self._flux_type is None:

            raise InvalidUnitError(
                "The flux_unit provided is not a valid flux quantity"
            )

    def _calculate_conversion(self):

        # convert the model to the right units so that we can
        # convert back later for speed

        tmp = self._model_converter[self._flux_type](self._test_value)

        if (
            tmp.unit == u.dimensionless_unscaled
            or tmp.unit == self._test_value.unit
            or tmp.unit == (self._test_value.unit) ** 2
        ):

            # this is a multiplicative model
            self._conversion = 1.0
            self._is_dimensionless = True

        else:

            self._conversion = tmp.unit.to(self._flux_unit, equivalencies=u.spectral())
            self._is_dimensionless = False

    @property
    def is_dimensionless(self):

        return self._is_dimensionless

    @property
    def model(self):
        """
        the model converted

        :return: a model in the proper units
        """

        return self._model_builder[self._flux_type]

    @property
    def conversion_factor(self):
        """
        the conversion factor needed to finalize the model into the
        proper units after computations

        :return:
        """

        return self._conversion


class DifferentialFluxConversion(FluxConversion):
    def __init__(self, flux_unit, energy_unit, flux_model, test_model):
        """
        Handles differential flux conversion and model building
        for point sources


        :param test_model: model to test the flux on
        :param flux_unit: an astropy unit string for differential flux
        :param energy_unit: an astropy unit string for energy
        :param flux_model: the base flux model to use
        """

        self._flux_lookup = {
            "photon_flux": 1.0 / (u.keV * u.cm ** 2 * u.s),
            "energy_flux": old_div(u.erg, (u.keV * u.cm ** 2 * u.s)),
            "nufnu_flux": old_div(u.erg ** 2, (u.keV * u.cm ** 2 * u.s)),
        }

        self._model_converter = {
            "photon_flux": test_model,
            "energy_flux": lambda x: x * test_model(x),
            "nufnu_flux": lambda x: x * x * test_model(x),
        }

        self._model_builder = {
            "photon_flux": flux_model,
            "energy_flux": lambda x, **param_specification: x
            * flux_model(x, **param_specification),
            "nufnu_flux": lambda x, **param_specification: x
            * x
            * flux_model(x, **param_specification),
        }

        super(DifferentialFluxConversion, self).__init__(
            flux_unit, energy_unit, flux_model
        )


class IntegralFluxConversion(FluxConversion):
    def __init__(self, flux_unit, energy_unit, flux_model, test_model):
        """
         Handles integral flux conversion and model building
         for point sources


         :param flux_unit: an astropy unit string for integral flux
         :param energy_unit: an astropy unit string for energy
         :param flux_model: the base flux model to use
         """

        self._flux_lookup = {
            "photon_flux": 1.0 / (u.cm ** 2 * u.s),
            "energy_flux": old_div(u.erg, (u.cm ** 2 * u.s)),
            "nufnu_flux": old_div(u.erg ** 2, (u.cm ** 2 * u.s)),
        }

        self._model_converter = {
            "photon_flux": lambda x: x * test_model(x),
            "energy_flux": lambda x: x * x * test_model(x),
            "nufnu_flux": lambda x: x ** 3 * test_model(x),
        }

        def photon_integrand(x, param_specification):
            return flux_model(x, **param_specification)

        def energy_integrand(x, param_specification):
            return x * flux_model(x, **param_specification)

        def nufnu_integrand(x, param_specification):
            return x * x * flux_model(x, **param_specification)

        self._model_builder = {
            "photon_flux": lambda e1, e2, **param_specification: integrate.quad(
                photon_integrand, e1, e2, args=(param_specification)
            )[0],
            "energy_flux": lambda e1, e2, **param_specification: integrate.quad(
                energy_integrand, e1, e2, args=(param_specification)
            )[0],
            "nufnu_flux": lambda e1, e2, **param_specification: integrate.quad(
                nufnu_integrand, e1, e2, args=(param_specification)
            )[0],
        }

        super(IntegralFluxConversion, self).__init__(flux_unit, energy_unit, flux_model)


class FittedPointSourceSpectralHandler(GenericFittedSourceHandler):
    def __init__(
        self,
        analysis_result,
        source,
        energy_range,
        energy_unit,
        flux_unit,
        confidence_level=0.68,
        equal_tailed=True,
        component=None,
        is_differential_flux=True,
    ):
        """

        A 3ML fitted point source.


        :param confidence_level:
        :param equal_tailed:
        :param is_differential_flux:
        :param analysis_result: a 3ML analysis
        :param source: the source to solve for
        :param energy_range: an array of energies to calculate the source over
        :param energy_unit: string astropy unit
        :param flux_unit: string astropy flux unit
        :param component: the component name to calculate
        """

        # first extract the source

        self._point_source = analysis_result.optimized_model.sources[source]

        # extract the components

        try:
            composite_model = self._point_source.spectrum.main.composite

            self._components = self._solve_for_component_flux(composite_model)

        except:

            self._components = None

        if component is not None:

            if self._components is not None:

                model = self._components[component]["function"].evaluate_at
                parameters = self._components[component]["function"].parameters
                test_model = self._components[component]["function"]
                parameter_names = self._components[component]["parameter_names"]

            else:

                raise NotCompositeModelError("This is not a composite model!")

        else:

            model = self._point_source.spectrum.main.shape.evaluate_at
            parameters = self._point_source.spectrum.main.shape.parameters
            test_model = self._point_source.spectrum.main.shape
            parameter_names = [
                par.name
                for par in list(
                    self._point_source.spectrum.main.shape.parameters.values()
                )
            ]

        energy_unit = u.Unit(energy_unit)

        # for any plotting, the x-axis remains unaltered outside of this class
        # therefore we convert to keV using the spectral equiv. helper from
        # astropy so that we can easily except energy, wavelength, or frequency
        # energy units

        if isinstance(energy_range, u.Quantity):

            energy_range = (energy_range).to("keV", equivalencies=u.spectral())

        else:

            energy_range = (energy_range * energy_unit).to(
                "keV", equivalencies=u.spectral()
            )

        energy_unit = energy_range.unit

        energy_range = energy_range.value

        flux_unit = u.Unit(flux_unit)

        self._flux_unit = flux_unit

        # now we will will find out what type of units we need

        # if we are doing differential flux plotting:

        if is_differential_flux:

            converter = DifferentialFluxConversion(
                flux_unit, energy_unit, model, test_model
            )

            flux_function = converter.model

            self._conversion = converter.conversion_factor

            super(FittedPointSourceSpectralHandler, self).__init__(
                analysis_result,
                flux_function,
                parameter_names,
                parameters,
                confidence_level,
                equal_tailed,
                energy_range,
            )

        else:

            converter = IntegralFluxConversion(
                flux_unit, energy_unit, model, test_model
            )

            flux_function = converter.model

            self._conversion = converter.conversion_factor

            # we treat the energy range as the range we want to integrate over

            e1 = np.array([energy_range.min()])
            e2 = np.array([energy_range.max()])

            # super will by pass the MLE/Bayes class if it is
            # inherited as well and go right to the general and
            # use the e1, e2 as the integral bounds

            super(FittedPointSourceSpectralHandler, self).__init__(
                analysis_result,
                flux_function,
                parameter_names,
                parameters,
                confidence_level,
                equal_tailed,
                e1,
                e2,
            )

        self._is_dimensionless = converter.is_dimensionless

    @property
    def is_dimensionless(self):

        return self._is_dimensionless

    @property
    def components(self):
        """

        :return: the components of the function
        """

        return self._components

    def _transform(self, value):
        """
        transform the values into the proper flux unit and apply the units
        :param value:
        :return:
        """

        return self._conversion * self._flux_unit * value

    @staticmethod
    def _solve_for_component_flux(composite_model):
        """

        now that we are using RandomVariates, we only need to compute the
        function directly to see the error in a component

        :param composite_model: an astromodels composite model
        :return: dict of component properties
        """

        function_dict = {}

        names = [f.name for f in composite_model.functions]

        counts = collections.Counter(names)
        for s, num in list(counts.items()):
            if num > 1:  # ignore strings that only appear once
                for suffix in range(
                    1, num + 1
                ):  # suffix starts at 1 and increases by 1 each time
                    names[names.index(s)] = "%s_n%i" % (
                        s,
                        suffix,
                    )  # replace each appearance of s

        for i, function in enumerate(composite_model.functions):

            tmp_dict = {}

            # extract the parameter names using the static_name property
            # because this is what the children will use in evaluate_at

            parameter_names = [
                par.static_name for par in list(function.parameters.values())
            ]

            tmp_dict["parameter_names"] = parameter_names

            tmp_dict["function"] = function

            function_dict[names[i]] = tmp_dict

        return function_dict
