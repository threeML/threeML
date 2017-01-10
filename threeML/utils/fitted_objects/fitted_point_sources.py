from astropy import units as u, constants
from sympy import Function, solve, lambdify
from sympy.abc import x

import scipy.integrate as integrate


from threeML.utils.fitted_objects.fitted_object import MLEFittedObject, BayesianFittedObject, GenericFittedObject


class NotCompositeModelError(RuntimeError):
    pass


class InvalidUnitError(RuntimeError):
    pass


class FluxConversion(object):

    def __init__(self, flux_unit, energy_unit, flux_model):

        self._flux_unit = flux_unit

        self._energy_unit = energy_unit

        self._model = flux_model

        self._test_value = 1. * energy_unit

        self._flux_type = None

        self._determine_quantity()

        self._calculate_conversion()

    def _determine_quantity(self):

        for k,v in self._flux_lookup.iteritems():


            try:

                self._flux_unit.to(v)

                self._flux_type = k

            except(u.UnitConversionError):

                continue


        if self._flux_type is None:

            raise InvalidUnitError('The flux_unit provided is not a valid flux quantity')

    def _calculate_conversion(self):

        tmp = self._model_converter[self._flux_type](self._test_value)

        self._conversion = tmp.unit.to(self._flux_unit)


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

    def __init__(self, flux_unit, energy_unit, flux_model):



        self._flux_lookup = {'photon_flux':  1. / (u.keV * u.cm ** 2 * u.s),
                             "energy_flux": u.erg / (u.keV * u.cm ** 2 * u.s),
                             "nufnu_flux": u.erg / (u.keV * u.cm ** 2 * u.s)}


        self._model_converter = {'photon_flux': lambda x: x * flux_model(x),
                                 "energy_flux": lambda x: x * x * flux_model(x),
                                 "nufnu_flux": lambda x: x ** 3 * flux_model(x)}


        self._model_builder = {'photon_flux': flux_model,
                               "energy_flux": lambda x: x * flux_model(x),
                               "nufnu_flux": lambda x: x * x * flux_model(x)}

        super(DifferentialFluxConversion, self).__init__(flux_unit,
                                                         energy_unit,
                                                         flux_model)


class IntegralFluxConversion(FluxConversion):

     def __init__(self, flux_unit, energy_unit, flux_model):

        self._flux_lookup = {'photon_flux': 1. / ( u.cm ** 2 * u.s),
                             "energy_flux": u.erg / ( u.cm ** 2 * u.s),
                             "nufnu_flux": u.erg / ( u.cm ** 2 * u.s)}

        self._model_converter = {'photon_flux': lambda x: x * flux_model(x),
                               "energy_flux": lambda x: x * x * flux_model(x),
                               "nufnu_flux": lambda x: x ** 3 * flux_model(x)}
    

        def photon_integrand(x):
            return flux_model(x)
    
        def energy_integrand(x):
            return x * flux_model(x)
    
        def nufnu_integrand(x):
            return x * x * flux_model(x)
    
        self._model_builder = {'photon_flux': lambda e1, e2: integrate.quad(photon_integrand, e1, e2),
                                  "energy_flux": lambda e1, e2: integrate.quad(energy_integrand, e1, e2),
                                  "nufnu_flux": lambda e1, e2: integrate.quad(nufnu_integrand, e1, e2)}
    
        
        super(IntegralFluxConversion, self).__init__(flux_unit,
                                                     energy_unit,
                                                     flux_model)





class FittedPointSource(GenericFittedObject):
    def __init__(self, analysis, source, energy_range, energy_unit, flux_unit, sigma=1, component=None):
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

            self._components = dict(zip(component_names, components))

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

        # for any plotting, the x-axis remain unaltered outside of this class
        # therefore we convert to keV using the spectral equiv. helper from
        # astropy so that we can easily except energy, wavelength, or frequency
        # energy units


        energy_range = (energy_range * energy_unit).to('keV', equivalencies=u.spectral())

        energy_unit = energy_range.unit

        energy_range = energy_range.value

        flux_unit = u.Unit(flux_unit)

        self._flux_unit = flux_unit

        # now we will will find out what type of units we need

        converter = DifferentialFluxConversion(flux_unit, energy_unit, model)

        flux_function = converter.model
        self._conversion = converter.conversion_factor


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
    def __init__(self, analysis, source, energy_range, energy_unit, flux_unit, sigma=1, component=None,
                 fraction_of_samples=.1):
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
