import astropy.units as u
from astropy.visualization import quantity_support

import matplotlib.pyplot as plt
import numpy as np
from sympy import Function
from sympy.abc import x
from sympy.solvers import solve
from sympy.utilities.lambdify import lambdify
import scipy.integrate as integrate
import collections

import pandas as pd

from threeML.utils.differentiation import get_jacobian


class InvalidUnitError(RuntimeError):
    pass


class SpectralFlux(object):
    def __init__(self, analysis):
        """
        Allows for computation of photon, energy and vFv fluxes in appropriate units
        for either joint likelihood or bayesian analysis. Handles error propapation in either
        scenario with either the fit covariance matrix or the bayesian chain.

        Both the total and component fluxes can be calculated with numerical error propagation in
        either case.

        Args:
            analysis: A joint likelihood or bayesian analysis
        """
        quantity_support()

        # looking at adding together multiple analysis This may be removed

        # Determine the type of analysis

        self._analysis_type = analysis._analysis_type

        self._analysis = analysis

    def model_flux(self, energy_unit='keV', flux_unit='erg/(cm2 keV s)', ene_min=10., ene_max=1E4,
                   sources_to_calculate=[], summed=False,
                   thin=100, alpha=0.05, **kwargs):
        """
        Calculate the flux of the total model. The energy unit describes the unit of the input integration range

        :param energy_unit: energy unit for x-axis
        :param flux_unit: spectral density unit for y-axis
        :param sources_to_calculate: list of str indicating which sources to plot
        :param summed: (bool) sum sources
        :param ene_min: minimum energy of x-axis
        :param ene_max: maximum energy of x-axis
        :param thin: thinning of bayesian samples (only for bayesian fits)
        :param alpha: chance of type I error (only for bayesian fits)
        :param kwargs: keyword args
        """

        # TODO: use pandas concat to make tables nicer. look at jointlikelihood set

        if self._analysis_type == "mle":

            val = self._flux_mle(energy_unit, flux_unit, sources_to_calculate, summed, ene_min, ene_max, **kwargs)

            return val

        elif self._analysis_type == "bayesian":
            return self._flux_bayes(energy_unit, flux_unit, sources_to_calculate, summed, ene_min, ene_max, thin,
                                    alpha, **kwargs)

    def component_flux(self, energy_unit='keV', flux_unit='erg/(cm2 keV s)', ene_min=10., ene_max=1E4,
                       sources_to_calculate=[], summed=False,
                       thin=100, **kwargs):
        """
        Calculate the flux of the model components. The energy unit describes the unit of the input integration range

        :param energy_unit: energy unit for integration range
        :param flux_unit: spectral density unit for y-axis
        :param sources_to_calculate: list of str indicating which sources to plot
        :param summed: (bool) sum sources
        :param ene_min: minimum energy of x-axis
        :param ene_max: maximum energy of x-axis
        :param thin: thinning of bayesian samples (only for bayesian fits)
        :param alpha: chance of type I error (only for bayesian fits)
        :param kwargs: keyword args

        """

        if self._analysis_type == "mle":

            return self._flux_component_mle(energy_unit, flux_unit, sources_to_calculate, summed, ene_min, ene_max,
                                            **kwargs)

        elif self._analysis_type == "bayesian":

            return self._flux_component_bayes(energy_unit, flux_unit, sources_to_calculate, summed, ene_min, ene_max,
                                              thin,
                                              **kwargs)

    def _flux_bayes(self, energy_unit='keV', flux_unit='erg/(cm2 keV s)', sources_to_calculate=[], summed=False,
                    ene_min=10.,
                    ene_max=1E4,
                    thin=1, alpha=0.05,
                    **kwargs):
        """
        Should not be called directly!
        """

        flux_unit = u.Unit(flux_unit)
        energy_unit = u.Unit(energy_unit)

        # looping of all analysis provided:



        # Get the the number of samples
        n_samples = self._analysis.raw_samples.shape[0]

        # First see if we are plotting all the sources
        if not sources_to_calculate:  # Assuming plot all sources

            sources_to_calculate = self._analysis._likelihood_model.point_sources.keys()

        # container for contours
        all_fluxes = []
        all_means = []
        all_hpd = []

        for source in sources_to_calculate:

            # Get the spectrum first

            model = self._analysis._likelihood_model.point_sources[source].spectrum.main.shape

            # Check the  type of function we want
            spectrum_type = self._get_spectrum_type(flux_unit)

            # Retrieve the right flux function (phts, energy, vfv)
            flux_function, conversion = self._get_flux_function(spectrum_type, model, energy_unit, flux_unit)

            # temporary list to store the propagated samples
            tmp = []

            # go through the thinned samples
            for i in range(0, n_samples, thin):

                # go through parameters
                for par in self._analysis.samples.keys():
                    mod_par = par.split('.')[-1]
                    model.free_parameters[mod_par].value = self._analysis.samples[par][i]

                # get the flux for the this sample
                tmp.append(flux_function(ene_min, ene_max))

            tmp = np.array(tmp)

            # pull the highest density posterior at the chosen alpha level

            hpd = self._analysis._hpd(tmp, alpha=alpha)
            mean_flux = np.mean(tmp)

            all_hpd.append(hpd * flux_unit * conversion)

            all_means.append(mean_flux * flux_unit * conversion)

            all_fluxes.append(tmp * conversion)

        flux_dict = collections.OrderedDict()

        all_fluxes = np.array(all_fluxes)

        if not summed:

            flux_dict['Source'] = sources_to_calculate
            flux_dict['Mean Flux'] = all_means
            flux_dict['HPD Flux'] = all_hpd
            flux_dict['Flux Distribution'] = [x for x in (all_fluxes * flux_unit)]

            return pd.DataFrame(flux_dict)



        elif summed:

            fluxes_summed = np.array(all_fluxes).sum(axis=0)

            flux_dict['Mean Summed Flux'] = np.mean(fluxes_summed) * flux_unit
            flux_dict['HPD Summed Flux'] = self._analysis._hpd(fluxes_summed, alpha=alpha) * flux_unit
            flux_dict['Summed Flux Distribution'] = fluxes_summed * flux_unit

            return pd.Series(flux_dict)

    def _flux_mle(self, energy_unit='keV', flux_unit='erg/(cm2 s)', sources_to_calculate=[], summed=False,
                  ene_min=10., ene_max=1E4,
                  **kwargs):

        """
        Should not be called directly!

        """
        self._analysis.restore_best_fit()

        flux_unit = u.Unit(flux_unit)
        energy_unit = u.Unit(energy_unit)

        # Initialize plotting arrays
        y_values = []

        errors = []

        flux_dict = collections.OrderedDict()

        # First see if we are plotting all the sources
        if not sources_to_calculate:  # Assuming plot all sources

            sources_to_calculate = self._analysis.likelihood_model.point_sources.keys()

        for source in sources_to_calculate:
            # Get the spectrum first

            model = self._analysis.likelihood_model.point_sources[source].spectrum.main.shape

            # Check the  type of function we want
            spectrum_type = self._get_spectrum_type(flux_unit)

            flux_function, conversion = self._get_flux_function(spectrum_type, model, energy_unit, flux_unit)

            err = self._propagate_full(flux_function, ene_min, ene_max)

            y_values.append(flux_function(ene_min, ene_max) * flux_unit * conversion)

            errors.append(err * flux_unit * conversion)

        if not summed:

            flux_dict['Source'] = sources_to_calculate
            flux_dict['Flux'] = y_values
            flux_dict['Error'] = errors

            flux_df = pd.DataFrame(flux_dict)

            return flux_df



        elif summed:

            y_values_summed = np.array(y_values).sum(axis=0)
            errors_summed = np.array(errors)
            errors_summed = np.sqrt((errors_summed ** 2).sum(axis=0))

            flux_df = pd.Series(data=[y_values_summed, errors_summed], index=['Summed Flux', 'Summed Error'])

            return flux_df

    def _flux_component_mle(self, energy_unit='keV', flux_unit='erg/(cm2  s)', sources_to_calculate=[], summed=False,
                            ene_min=10.,
                            ene_max=1E4,
                            **kwargs):

        self._analysis.restore_best_fit()

        energy_unit = u.Unit(energy_unit)
        flux_unit = u.Unit(flux_unit)

        y_values = []
        errors = []

        # First see if we are plotting all the sources
        if sources_to_calculate == []:  # Assuming plot all sources

            sources_to_calculate = self._analysis.likelihood_model.point_sources.keys()

        # if components == []: # Assuming plot all sources

        #    sources_to_plot = self.analysis.likelihood_model.point_sources.keys()


        for source in sources_to_calculate:

            composite_model = self._analysis.likelihood_model.point_sources[source].spectrum.main.composite
            models = self._solve_for_component_flux(composite_model)

            # Check the type of function we want
            spectrum_type = self._get_spectrum_type(flux_unit)

            y_vals_per_comp = []
            errors_per_comp = []
            for model in models:
                flux_function, conversion = self._get_flux_function(spectrum_type, model, energy_unit, flux_unit)

                err = self._propagate_full(flux_function, ene_min, ene_max)

                y_vals_per_comp.append(flux_function(ene_min, ene_max) * flux_unit * conversion)

                errors_per_comp.append(err * flux_unit * conversion)

            y_values.append(y_vals_per_comp)
            errors.append(errors_per_comp)

        flux_total_dict = collections.OrderedDict()

        if not summed:

            for y_val_pc, err_pc, source in zip(y_values, errors, sources_to_calculate):
                model_names = [func.name for func in

                               self._analysis.likelihood_model.point_sources[source].spectrum.main.composite.functions]

                flux_dict = collections.OrderedDict()

                flux_dict['Component'] = model_names
                flux_dict['Flux'] = y_val_pc
                flux_dict['Error'] = err_pc

                tmp_df = pd.DataFrame(flux_dict)

                flux_total_dict[source] = tmp_df

            return flux_total_dict

        elif summed:

            # There is an assumption that sources have the same models... may have to alter this!
            y_values_summed = np.array(y_values).sum(axis=0) * flux_unit
            errors_summed = np.array(errors) ** 2
            errors_summed = np.sqrt(errors_summed.sum(axis=0)) * flux_unit

            # This is a kludge assuming all sources have the same models

            model_names = [func.name for func in
                           self._analysis.likelihood_model.point_sources[
                               sources_to_calculate[0]].spectrum.main.composite.functions]

            flux_total_dict['Component'] = model_names
            flux_total_dict['Summed Flux'] = y_values_summed
            flux_total_dict['Summed Error'] = errors_summed

            return pd.DataFrame(flux_total_dict)

    def _flux_component_bayes(self, energy_unit='keV', flux_unit='erg/(cm2  s)', sources_to_calculate=[], summed=False,
                              ene_min=10.,
                              ene_max=1E4, thin=1, alpha=0.05,
                              **kwargs):
        """
        Should not be called directly

        """

        energy_unit = u.Unit(energy_unit)
        flux_unit = u.Unit(flux_unit)

        # Get the the number of samples
        n_samples = self._analysis.raw_samples.shape[0]

        # First see if we are plotting all the sources
        if not sources_to_calculate:  # Assuming plot all sources

            sources_to_calculate = self._analysis._likelihood_model.point_sources.keys()

        # this is a kludge at the moment. Model number may vary!
        num_models = len(
                self._analysis._likelihood_model.point_sources[
                    sources_to_calculate[0]].spectrum.main.composite.functions)

        all_fluxes = []
        all_means = []
        all_hpd = []

        for source in sources_to_calculate:

            composite_model = self._analysis._likelihood_model.point_sources[source].spectrum.main.composite
            models = self._solve_for_component_flux(composite_model)

            # Check the type of function we want
            spectrum_type = self._get_spectrum_type(flux_unit)

            fluxes_per_component = []
            mean_per_component = []
            hpd_per_component = []
            for model in models:

                # Retrieve the right flux function (phts, energy, vfv)
                flux_function, conversion = self._get_flux_function(spectrum_type, model, energy_unit, flux_unit)

                # temporary list to store the propagated samples
                tmp = []

                # go through the thinned samples
                for i in range(0, n_samples, thin):

                    # go through parameters
                    for par in self._analysis.samples.keys():
                        mod_par = par.split('.')[-1]
                        composite_model.free_parameters[mod_par].value = self._analysis.samples[par][i]

                    # get the flux for the this sample
                    tmp.append(flux_function(ene_min, ene_max))

                tmp = np.array(tmp)

                # pull the highest denisty posterior at the choosen alpha level
                hpd_per_component.append(self._analysis._hpd(tmp, alpha=alpha) * conversion * flux_unit)
                mean_per_component.append(np.mean(tmp) * conversion * flux_unit)

                fluxes_per_component.append(tmp * conversion)

            all_fluxes.append(fluxes_per_component)
            all_hpd.append(hpd_per_component)
            all_means.append(mean_per_component)

        all_fluxes = np.array(all_fluxes)

        flux_total_dict = collections.OrderedDict()

        if not summed:

            for flux_pc, mean_pc, hpd_pc, source in zip(all_fluxes, all_means, all_hpd, sources_to_calculate):
                model_names = [func.name for func in
                               self._analysis._likelihood_model.point_sources[source].spectrum.main.composite.functions]

                flux_dict = collections.OrderedDict()
                flux_dict['Component'] = model_names
                flux_dict['Mean Flux'] = mean_pc
                flux_dict['HPD Flux'] = hpd_pc
                flux_dict['Flux Distribution'] = [x for x in (flux_pc * flux_unit)]

                flux_total_dict[source] = pd.DataFrame(flux_dict)

            return flux_total_dict


        elif summed:

            # Assumes all sources have the same model!
            summed_fluxes = np.array(all_fluxes).sum(axis=0)

            # This is a kludge that assumes all sources have the same model!
            model_names = [func.name for func in
                           self._analysis._likelihood_model.point_sources[
                               sources_to_calculate[0]].spectrum.main.composite.functions]

            means = map(np.mean, summed_fluxes) * flux_unit

            hpd = map(lambda x: self._analysis._hpd(x, alpha) * flux_unit, summed_fluxes)

            flux_total_dict['Component'] = model_names
            flux_total_dict['Mean Summed Flux'] = means
            flux_total_dict['HPD Summed Flux'] = hpd
            flux_total_dict['Summed Flux Distribution'] = summed_fluxes * flux_unit

            return flux_total_dict

    def _propagate_full(self, flux_function, ene_min, ene_max):

        # Get the parameters from the minimizer

        parameters = self._analysis.minimizer.parameters

        # We will store the first derivatives @ the best fit values
        first_derivatives = []

        # Now loop through each parameter and free it while
        # holding the others constant. This is the normal (pun intended)
        # error propagation formula.
        for par in parameters.keys():

            # go back to the best fit

            self._analysis.restore_best_fit()


            parameter_best_fit_value = parameters[par].value

            min_value, max_value = parameters[par].bounds

            # Create a temporary flux function to take a
            # derivative w.r.t. the free parameter

            def tmpflux(current_value):

                parameters[par].value = current_value

                return flux_function(ene_min, ene_max)

            # get the first derivatives and append them for some
            # linear algebra

            this_derivative = get_jacobian(tmpflux, parameter_best_fit_value, min_value, max_value)[0][0]

            first_derivatives.append(this_derivative)

        first_derivatives = np.array(first_derivatives)

        # Now we take the inner product with the covariance matrix

        tmp = first_derivatives.dot(self._analysis.covariance_matrix)

        error = (np.sqrt(tmp.dot(first_derivatives)))

        # make sure to restore the best fit to the analysis
        self._analysis.restore_best_fit()

        return error

    @staticmethod
    def _calculate_conversion(energy_unit, flux_unit, integrand):

        x = 1 * energy_unit

        tmp = x * integrand(x)
        conversion = tmp.unit.to(flux_unit)

        return conversion

    def _get_flux_function(self, spectrum_type, model, energy_unit, flux_unit):
        """
        Returns the appropriate flux function based off input spectral units
        :param spectrum_type: str from _get_spectrum_type indicating the spectrum type to use
        :param model: a call to an astromodel function
        :param flux_unit: astropy unit
        :return: function that calls the correct flux type
        """

        if spectrum_type == "phtflux":

            def integrand(x):
                return model(x)

            conversion = self._calculate_conversion(energy_unit, flux_unit, integrand)

            flux_function = lambda ene_min, ene_max: integrate.quad(integrand, ene_min, ene_max)[0]


        elif spectrum_type == "eneflux":

            def integrand(x):
                return x * model(x)

            conversion = self._calculate_conversion(energy_unit, flux_unit, integrand)

            flux_function = lambda ene_min, ene_max: integrate.quad(integrand, ene_min, ene_max)[0]


        elif spectrum_type == "vfvflux":

            def integrand(x):
                return x * x * model(x)

            conversion = self._calculate_conversion(energy_unit, flux_unit, integrand)

            flux_function = lambda ene_min, ene_max: integrate.quad(integrand, ene_min, ene_max)[0]

        return flux_function, conversion

    @staticmethod
    def _get_spectrum_type(y_unit):
        """

        :param y_unit: an astropy unit
        :return: str indicating the type of unit desired
        """

        pht_flux_unit = 1. / (u.cm ** 2 * u.s)
        flux_unit = u.erg / (u.cm ** 2 * u.s)
        vfv_unit = u.erg ** 2 / (u.cm ** 2 * u.s)

        # Try to convert to base units. If it works then return that unit type
        try:

            y_unit.to(pht_flux_unit)

            return "phtflux"


        except(u.UnitConversionError):

            try:

                y_unit.to(flux_unit)

                return "eneflux"

            except(u.UnitConversionError):

                try:
                    y_unit.to(vfv_unit)

                    return "vfvflux"

                except:

                    raise InvalidUnitError("The provided flux_unit is not valid!")

    def _solve_for_component_flux(self, composite_model):
        """
        Uses sympy to algebraically solve for fucntional form of the component models.
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
