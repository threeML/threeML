from astropy import units as u, constants
from sympy import Function, solve, lambdify
from sympy.abc import x

from threeML.utils.fitted_objects.fitted_object import MLEFittedObject, BayesianFittedObject, GenericFittedObject, \
    NotCompositeModelError, InvalidUnitError


class FittedPointSource(GenericFittedObject):

    def __init__(self, analysis, source, energy_range, energy_unit, flux_unit, sigma=1, component = None):
        """

        A 3ML fitted point source.


        :param analysis: a 3ML analysis
        :param source: the source to solve for
        :param energy_range: an array of energies to calculate the source over
        :param energy_unit: string astropy unit
        :param flux_unit: string astropy flux unit
        :param component: the component to calculate
        """

        # extract the components

        try:
            composite_model = source.spectrum.main.composite

            components = self._solve_for_component_flux(composite_model)

            component_names = [function.name for function in composite_model.functions]



            self._components = dict(zip(component_names,components))

        except:

            self._components = None



        if component is not None:

            if self._components is not None:

                model = self._components[component]

            else:

                raise NotCompositeModelError("This is not a composite model!")

        else:

            model = source.spectrum.main

        # see if we have frequency units

        self._x_unit_checker(energy_unit)

        energy_unit = u.Unit(energy_unit)

        if self._convert_to_frequency:

            # we are going to plot in frequency, but

            # the functions take energy.

            energy_range = ((energy_range * energy_unit * constants.h).cgs).to('keV')

            energy_unit = energy_range.unit

            energy_range = energy_range.value

        flux_unit = u.Unit(flux_unit)

        self._flux_unit = flux_unit

        spectrum_type = self._get_spectrum_type()

        flux_function, self._conversion = self._get_flux_function(spectrum_type, model, flux_unit, energy_unit)



        super(FittedPointSource, self).__init__(analysis,
                                                source,
                                                flux_function,
                                                sigma,
                                                energy_range)


    def _get_free_parameters(self):


        self._free_parameters = self._source.spectrum.main.shape.free_parameters


    @property
    def components(self):

        return self._components

    @property
    def error_region(self):
        """
        the error region of the point source spectrum
        """

        # This get the error region here, but calls the super class
        # to return the value. The value can then be manipulated further
        # up by a parent

        return self._conversion * self._flux_unit * super(FittedPointSource, self).error_region

    @property
    def spectrum(self):

        return self._flux_unit * self._conversion * self._best_fit_values

    @staticmethod
    def _calculate_conversion(energy_unit, flux_unit, model):

        x = 1 * energy_unit

        tmp = model(x)
        conversion = tmp.unit.to(flux_unit)

        return conversion

    def _get_flux_function(self, spectrum_type, model, flux_unit, energy_unit):
        """
        Returns the appropriate flux function based off input spectral units
        :param spectrum_type: str from _get_spectrum_type indicating the spectrum type to use
        :param model: a call to an astromodel function
        :param y_unit: astropy unit
        :return: function that calls the correct flux type
        """

        if spectrum_type == "phtflux":

            def this_model(x):
                return model(x)

            conversion = self._calculate_conversion(energy_unit, flux_unit, this_model)

            flux_function = this_model


        elif spectrum_type == "eneflux":

            def this_model(x):
                return x * model(x)

            conversion = self._calculate_conversion(energy_unit, flux_unit, this_model)

            flux_function = this_model


        elif spectrum_type == "vfvflux":

            def this_model(x):
                return x * x * model(x)

            conversion = self._calculate_conversion(energy_unit, flux_unit, this_model)

            flux_function = this_model

        return flux_function, conversion


    def _get_spectrum_type(self):
        """

        :param flux_unit: an astropy unit
        :return: str indicating the type of unit desired
        """

        pht_flux_unit = 1. / (u.keV * u.cm ** 2 * u.s)
        flux_unit = u.erg / (u.keV * u.cm ** 2 * u.s)
        vfv_unit = u.erg ** 2 / (u.keV * u.cm ** 2 * u.s)

        # Try to convert to base units. If it works then return that unit type
        try:

            self._flux_unit.to(pht_flux_unit)

            return "phtflux"


        except(u.UnitConversionError):

            try:

                self._flux_unit.to(flux_unit)

                return "eneflux"

            except(u.UnitConversionError):

                try:
                    self._flux_unit.to(vfv_unit)

                    return "vfvflux"

                except:

                    raise InvalidUnitError("The y_unit provided is not a valid spectral quantity")

    def _x_unit_checker(self, energy_unit):
        """
        Checks if the x unit is in energy or frequency

        :param energy_unit: astropy unit string
        :return:
        """

        x_unit = u.Unit(energy_unit)

        # First we check if the units are energy
        try:

            x_unit.to('keV')

            # well, this is an energy. we do not have to convert at all

            self._convert_to_frequency = False

        except(u.UnitConversionError):

            # now we see if they are frequency

            try:

                x_unit.to('Hz')

                # Ok, we found a frequency. that means we will do the calculation in eV and then
                # convert back at the end



                self._convert_to_frequency = True

            except(u.UnitConversionError):

                raise InvalidUnitError("x unit is not energy or frequency")

    @staticmethod
    def _solve_for_component_flux(composite_model):
        """
        Uses sympy to algebraically solve for functional form of the component models.
        This produces the proper form of the individual flux w.r.t the total flux so
        that error propagation can take into full account the variance in the full model

        :param composite_model: an astromodels composite model
        :return: list of solved component flux functions
        """

        replicated_expression = composite_model.expression

        num_models = len(composite_model.functions)
        mod_solve = []
        function_dict = {}

        # First build the expressions and create a dictionary for
        # the component functions to be referecenced later

        for i, func in enumerate(composite_model.functions):
            # Need to replace all the strings correctly
            replicated_expression = replicated_expression.replace("%s{%d}" % (func.name, i + 1),
                                                                  "mod_solve[%d](x)" % i)

            # build function dict
            function_dict["%s_%d" % (func.name, i + 1)] = func

            # create sympy functions
            mod_solve.append(Function("%s_%d" % (func.name, i + 1)))

        # add the total flux at the end
        function_dict['total'] = composite_model

        replicated_expression += "- mod_solve[%d](x)" % num_models

        mod_solve.append(Function("total"))

        solutions = []
        # go through all models and solve for component fluxes algebraically
        for i, func in enumerate(composite_model.functions):
            solutions.append(solve(eval(replicated_expression), str(mod_solve[i]) + '(x)')[0])

        # use sympy to create new functions for the solved components
        component_flux = [lambdify(x, sol, function_dict) for sol in solutions]

        return component_flux


class MLEPointSource(FittedPointSource, MLEFittedObject):
    def __init__(self, analysis, source, energy_range, energy_unit, flux_unit, sigma=1, component=None):
        """
        an MLE fitted point source


        :param analysis: a JointLikelihood fitted object
        :param source: the astromodels source to be examined
        :param energy_range: a numpy aray of of energies
        :param energy_unit: the energy unit
        :param flux_unit: the flux unit
        :param component: the name (string) of a component of the source model
        """

        assert analysis.analysis_type == "mle"

        super(MLEPointSource, self).__init__(analysis,
                                             source,
                                             energy_range,
                                             energy_unit,
                                             flux_unit,
                                             sigma,
                                             component)


class BayesianPointSource(FittedPointSource, BayesianFittedObject):

    def __init__(self, analysis, source, energy_range, energy_unit, flux_unit, sigma=1, component=None,fraction_of_samples=.1):
        """
        A Bayesian fitted point source


        :param analysis: a BayesianAnalysis fitted object
        :param source: the astromodels source to be examined
        :param energy_range: a numpy aray of of energies
        :param energy_unit: the energy unit
        :param flux_unit: the flux unit
        :param component: the name (string) of a component of the source model
        :param fraction_of_samples: fraction of the samples to use when computing contours
        """

        assert analysis.analysis_type == "bayesian"

        assert 0. < fraction_of_samples < 1., "thin must be between 0 and 1"

        self._fraction_of_samples = fraction_of_samples

        super(BayesianPointSource, self).__init__(analysis,
                                                  source,
                                                  energy_range,
                                                  energy_unit,
                                                  flux_unit,
                                                  sigma,
                                                  component)