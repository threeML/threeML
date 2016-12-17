import astropy.units as u
import astropy.constants as constants
import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import quantity_support
from sympy import Function
from sympy.abc import x
from sympy.solvers import solve
from sympy.utilities.lambdify import lambdify

from threeML.io.progress_bar import progress_bar

from threeML.utils.differentiation import get_jacobian
from threeML.config.config import threeML_config



class InvalidUnitError(RuntimeError):
    pass


class SpectralPlotter(object):
    """
    This class handles plotting of spectral fits and their associated contours.

    MLE fits are plotted with their contours computed by propagating the covariance
    matrix through the function.

    Bayesian fits are plotted with their marginal posteriors plotted through the function.

    :param analysis: an MLE or bayesian analysis object
    """

    def __init__(self, analysis):
        quantity_support()

        # looking at adding together multiple analysis This may be removed

        # Determine the type of analysis

        self._analysis_type = analysis.analysis_type

        self._analysis = analysis

    def plot_model(self, x_unit='keV', y_unit='erg/(cm2 keV s)', sources_to_plot=(), summed=False, x_min=10.,
                   x_max=1E4, num_ene=300, plot_num=1, thin=100, alpha=0.05, legend=True, fit_cmap=None,
                   contour_cmap=None, contour_alpha=0.6, lw=1., ls='-', **kwargs):
        """
        Plot the model and the model contours for the selected sources.

        :param x_unit: energy/frequency unit for x-axis
        :param y_unit: spectral density unit for y-axis
        :param sources_to_plot: list of str indicating which sources to plot
        :param summed: (bool) sum sources
        :param x_min: minimum energy of x-axis
        :param x_max: maximum energy of x-axis
        :param num_ene: number of energies to calculate
        :param plot_num: figure number of plot
        :param thin: thinning of bayesian samples (only for bayesian fits)
        :param alpha: chance of type I error (only for bayesian fits)
        :param legend: (bool) include legend
        :param fit_cmap: a matplotlib color map for the fit
        :param contour_cmap: a matplotlib color map for the contours (MLE only)
        :param contour_alpha: transparency of contours
        :param lw: linewidth for MLE plots
        :param ls: linestyle for MLE plots
        :param kwargs: keyword args
        """

        # First we need to check if the x_unit is an energy or frequency

        self._x_unit_checker(x_unit)

        if self._analysis_type == "mle":
            return self._plot_mle(x_unit, y_unit, sources_to_plot, summed, x_min, x_max, num_ene, plot_num, legend,
                                  fit_cmap,
                                  contour_cmap, contour_alpha, lw, ls, **kwargs)

        elif self._analysis_type == "bayesian":

            return self._plot_bayes(x_unit, y_unit, sources_to_plot, summed, x_min, x_max, num_ene, plot_num, thin,
                                    alpha, legend, fit_cmap, contour_alpha, **kwargs)

    def plot_components(self, x_unit='keV', y_unit='erg/(cm2 keV s)', sources_to_plot=(), summed=False, x_min=10.,
                        x_max=1E4, num_ene=300,
                        thin=100, alpha=0.05, legend=True, fit_cmap=None, contour_cmap=None,
                        contour_alpha=0.6, lw=1., ls='-', **kwargs):
        """
        Plot the components of a fits and their associated contours

        :param x_unit: energy unit for x-axis
        :param y_unit: spectral density unit for y-axis
        :param sources_to_plot: list of str indicating which sources to plot
        :param summed: (bool) sum sources
        :param x_min: minimum energy of x-axis
        :param x_max: maximum energy of x-axis
        :param num_ene: number of energies to calculate
        :param thin: thinning of bayesian samples (only for bayesian fits)
        :param alpha: chance of type I error (only for bayesian fits)
        :param legend: (bool) include legend
        :param fit_cmap: a matplotlib color map for the fit
        :param contour_cmap: a matplotlib color map for the contours (MLE only)
        :param contour_alpha: transparency of contours
        :param lw: linewidth for MLE plots
        :param ls: linestyle for MLE plots
        :param kwargs: keyword args

        """

        self._x_unit_checker(x_unit)

        if self._analysis_type == "mle":

            return self._plot_component_mle(x_unit, y_unit, sources_to_plot, summed, x_min, x_max, num_ene, legend,
                                            fit_cmap, contour_cmap, contour_alpha, lw, ls, **kwargs)

        elif self._analysis_type == "bayesian":

            return self._plot_component_bayes(x_unit, y_unit, sources_to_plot, summed, x_min, x_max, num_ene,

                                              thin,
                                              alpha, legend, fit_cmap, contour_alpha, **kwargs)

    def _plot_bayes(self, x_unit='keV', y_unit='erg/(cm2 keV s)', sources_to_plot=(), summed=False, x_min=10.,
                    x_max=1E4,
                    num_ene=300, plot_num=1, thin=100, alpha=0.05, legend=True, fit_cmap=None, contour_alpha=0.6,
                    **kwargs):
        """
        Should not be called directly!
        """

        x_unit = u.Unit(x_unit)

        y_unit = u.Unit(y_unit)

        # Set the default color map if none is provided
        if not fit_cmap:

            fit_cmap = plt.get_cmap(threeML_config['model plot']['bayes cmap'])


        x_values = np.logspace(np.log10(x_min), np.log10(x_max), num_ene)

        # looping of all analysis provided:



        # Get the the number of samples
        n_samples = self._analysis.raw_samples.shape[0]

        fig, ax = plt.subplots()

        # First see if we are plotting all the sources
        if not sources_to_plot:  # Assuming plot all sources

            sources_to_plot = self._analysis._likelihood_model.point_sources.keys()

        # container for contours
        all_contours = []

        if self._convert_to_frequency:

            # we are going to plot in frequency, but
            # the functions take energy.

            x_range = (x_values * x_unit * constants.h).cgs



            # we will later divide by h to convert back

        else:

            x_range = x_values * x_unit

        for source in sources_to_plot:

            # Get the spectrum first
            call = self._analysis._likelihood_model.point_sources[source].spectrum.main
            model = self._analysis._likelihood_model.point_sources[source].spectrum.main.shape

            # Check the  type of function we want
            spectrum_type = self._get_spectrum_type(y_unit)

            # Retrieve the right flux function (phts, energy, vfv)
            flux_function = self._get_flux_function(spectrum_type, model, y_unit)

            # temporary list to store the propagated samples
            tmp = []

            # go through the thinned samples

            itr = range(0, n_samples, thin)

            with progress_bar(len(itr)) as p:

                for i in itr:

                    # go through parameters
                    for par in self._analysis.samples.keys():
                        mod_par = par.split('.')[-1]
                        model.free_parameters[mod_par].value = self._analysis.samples[par][i]

                    # get the flux for the this sample
                    tmp.append(flux_function(x_range))

                    p.increase()

            tmp = np.array(tmp).T

            # pull the highest denisty posterior at the choosen alpha level
            contours = np.array([self._analysis._hpd(mc, alpha=alpha) for mc in tmp])
            all_contours.append(contours)

        color = np.linspace(0., 1., len(sources_to_plot))
        color_itr = 0

        # Ok, now everything is computed and we can convert back to
        # the x values to to frequency is need

        if self._convert_to_frequency:

            x_range = (x_range / constants.h).to(x_unit)

        if not summed:

            for source, contours in zip(sources_to_plot, all_contours):

                ax.fill_between(x_range,
                                contours[:, 0] * y_unit,
                                contours[:, 1] * y_unit,
                                color=fit_cmap(color[color_itr]),
                                alpha=contour_alpha,
                                label=source)

                ax.set_xscale('log')
                ax.set_yscale('log')
                if legend:
                    ax.legend(**kwargs)

                color_itr += 1

        elif summed:

            contours_summed = np.array(all_contours).sum(axis=0)
            ax.fill_between(x_range,
                            contours_summed[:, 0] * y_unit,
                            contours_summed[:, 1] * y_unit,
                            color=fit_cmap(color[color_itr]),
                            alpha=contour_alpha,
                            label=source)

            ax.set_xscale('log')
            ax.set_yscale('log')

        return fig

    def _plot_mle(self, x_unit='keV', y_unit='erg/(cm2 keV s)', sources_to_plot=[], summed=False,
                  x_min=10., x_max=1E4, num_ene=300, plot_num=1, legend=True, fit_cmap=None, contour_cmap=None,
                  contour_alpha=0.6, lw=1., ls='-',
                  **kwargs):

        """
        Should not be called directly!

        """
        self._analysis.restore_best_fit()

        x_unit = u.Unit(x_unit)
        y_unit = u.Unit(y_unit)

        if fit_cmap is None:
            fit_cmap = plt.get_cmap(threeML_config['model plot']['fit cmap'])

        if contour_cmap is None:
            contour_cmap = plt.get_cmap(threeML_config['model plot']['contour cmap'])

        # Initialize plotting arrays
        y_values = []
        x_values = np.logspace(np.log10(x_min), np.log10(x_max), num_ene)
        errors = []

        fig, ax = plt.subplots()

        if self._convert_to_frequency:

            # we are going to plot in frequency, but
            # the functions take energy.

            x_range = (x_values * x_unit * constants.h).cgs



            # we will later divide by h to convert back

        else:

            x_range = x_values * x_unit

        # First see if we are plotting all the sources
        if not sources_to_plot:  # Assuming plot all sources

            sources_to_plot = self._analysis.likelihood_model.point_sources.keys()

        for source in sources_to_plot:
            # Get the spectrum first

            model = self._analysis.likelihood_model.point_sources[source].spectrum.main.shape

            # Check the  type of function we want
            spectrum_type = self._get_spectrum_type(y_unit)

            flux_function = self._get_flux_function(spectrum_type, model, y_unit)

            err = self._propagate_full(flux_function, x_range)

            y_values.append(flux_function(x_range))

            errors.append(err * y_unit)

        color = np.linspace(0., 1., len(sources_to_plot))
        color_itr = 0

        # Ok, now everything is computed and we can convert back to
        # the x values to to frequency is need

        if self._convert_to_frequency:

            x_range = (x_range / constants.h).to(x_unit)

        if not summed:
            for y_val, err, source in zip(y_values, errors, sources_to_plot):
                ax.loglog(x_range,
                          y_val,
                          color=fit_cmap(color[color_itr]),
                          label=source,
                          lw=lw,
                          linestyle=ls)

                up_y = y_val + err
                down_y = y_val - err

                ax.fill_between(x_range,
                                down_y,
                                up_y,
                                facecolor=contour_cmap(color[color_itr]),
                                alpha=contour_alpha)

                ax.set_xscale('log')
                ax.set_yscale('log')
                if legend:
                    ax.legend(**kwargs)


        elif summed:

            y_values_summed = np.array(y_values).sum(axis=0)
            errors_summed = np.array(errors)
            errors_summed = np.sqrt((errors_summed ** 2).sum(axis=0))

            ax.loglog(x_range,
                      y_values_summed,
                      color=fit_cmap(color[color_itr]),
                      lw=lw,
                      linestyle=ls)

            up_y = y_values_summed + errors_summed
            down_y = y_values_summed - errors_summed

            ax.fill_between(x_range,
                            down_y,
                            up_y,
                            facecolor=contour_cmap(color[color_itr]),
                            alpha=contour_alpha)

            ax.set_xscale('log')
            ax.set_yscale('log')

        return fig

    def _plot_component_mle(self, x_unit='keV', y_unit='erg/(cm2 keV s)', sources_to_plot=[], summed=False, x_min=10.,
                            x_max=1E4, num_ene=300, legend=True, fit_cmap=None, contour_cmap=None,
                            contour_alpha=0.6, lw=1., ls='-',
                            **kwargs):

        self._analysis.restore_best_fit()

        x_unit = u.Unit(x_unit)
        y_unit = u.Unit(y_unit)

        if fit_cmap is None:
            fit_cmap = plt.get_cmap(threeML_config['model plot']['fit cmap'])

        if contour_cmap is None:
            contour_cmap = plt.get_cmap(threeML_config['model plot']['contour cmap'])

        # Initialize plotting arrays
        y_values = []
        x_values = np.logspace(np.log10(x_min), np.log10(x_max), num_ene)

        if self._convert_to_frequency:

            # we are going to plot in frequency, but
            # the functions take energy.

            x_range = (x_values * x_unit * constants.h).cgs



            # we will later divide by h to convert back

        else:

            x_range = x_values * x_unit

        errors = []

        fig, ax = plt.subplots()

        # First see if we are plotting all the sources
        if not sources_to_plot:  # Assuming plot all sources

            sources_to_plot = self._analysis.likelihood_model.point_sources.keys()

        for source in sources_to_plot:

            composite_model = self._analysis.likelihood_model.point_sources[source].spectrum.main.composite
            models = self._solve_for_component_flux(composite_model)

            # Check the type of function we want
            spectrum_type = self._get_spectrum_type(y_unit)

            y_vals_per_comp = []
            errors_per_comp = []
            for model in models:
                flux_function = self._get_flux_function(spectrum_type, model, y_unit)
                err = self._propagate_full(flux_function, x_range)

                y_vals_per_comp.append(flux_function(x_range))

                errors_per_comp.append(err * y_unit)

            y_values.append(y_vals_per_comp)
            errors.append(errors_per_comp)

        color = np.linspace(0., 1., len(sources_to_plot) * len(models))

        color_itr = 0

        # Ok, now everything is computed and we can convert back to
        # the x values to to frequency is need

        if self._convert_to_frequency:

            x_range = (x_range / constants.h).to(x_unit)

        if not summed:
            for y_val_pc, err_pc, source in zip(y_values, errors, sources_to_plot):

                model_names = [func.name for func in
                               self._analysis.likelihood_model.point_sources[source].spectrum.main.composite.functions]
                for y_val, err, name in zip(y_val_pc, err_pc, model_names):
                    pos_mask = np.logical_and(y_val > 0, err > 0)

                    ax.fill_between(x_range[pos_mask],
                                    y_val[pos_mask] - err[pos_mask],
                                    y_val[pos_mask] + err[pos_mask],
                                    color=contour_cmap(color[color_itr]),
                                    alpha=contour_alpha)

                    ax.loglog(x_range[pos_mask],
                              y_val[pos_mask],
                              color=fit_cmap(color[color_itr]),
                              label='%s:%s' % (source, name),
                              lw=lw,
                              ls=ls)

                    ax.set_xscale('log')
                    ax.set_yscale('log')
                    if legend:
                        ax.legend(**kwargs)

                    color_itr += 1
        elif summed:

            color = np.linspace(0., 1., len(models))
            color_itr = 0

            # There is an assumption that sources have the same models... may have to alter this!
            y_values_summed = np.array(y_values).sum(axis=0) * y_unit
            errors_summed = np.array(errors) ** 2
            errors_summed = np.sqrt(errors_summed.sum(axis=0)) * y_unit

            # This is a kludge assuming all sources have the same models

            model_names = [func.name for func in
                           self._analysis.likelihood_model.point_sources[
                               sources_to_plot[0]].spectrum.main.composite.functions]

            for y_val, err, name in zip(y_values_summed, errors_summed, model_names):

                pos_mask = np.logical_and(y_val > 0, err > 0)

                ax.fill_between(x_range[pos_mask],
                                y_val[pos_mask] - err[pos_mask],
                                y_val[pos_mask] + err[pos_mask],
                                color=contour_cmap(color[color_itr]),
                                alpha=contour_alpha)

                ax.loglog(x_range[pos_mask],
                          y_val[pos_mask],
                          color=fit_cmap(color[color_itr]),
                          label='%s' % name,
                          lw=lw,
                          ls=ls)

                ax.set_xscale('log')
                ax.set_yscale('log')
                if legend:
                    ax.legend(**kwargs)

                color_itr += 1

        return fig

    def _plot_component_bayes(self, x_unit='keV', y_unit='erg/(cm2 keV s)', sources_to_plot=[], summed=False,
                              x_min=10.,
                              x_max=1E4, num_ene=300, thin=100, alpha=0.05, legend=True, fit_cmap=None,
                              contour_alpha=0.6,
                              **kwargs):
        """
        Should not be called directly

        """

        x_unit = u.Unit(x_unit)
        y_unit = u.Unit(y_unit)

        if not fit_cmap:

            fit_cmap = plt.get_cmap(threeML_config['model plot']['bayes cmap'])

        x_values = np.logspace(np.log10(x_min), np.log10(x_max), num_ene)

        if self._convert_to_frequency:

            # we are going to plot in frequency, but
            # the functions take energy.

            x_range = (x_values * x_unit * constants.h).cgs



            # we will later divide by h to convert back

        else:

            x_range = x_values * x_unit

        # Get the the number of samples
        n_samples = self._analysis.raw_samples.shape[0]

        fig, ax = plt.subplots()

        # First see if we are plotting all the sources
        if not sources_to_plot:  # Assuming plot all sources

            sources_to_plot = self._analysis._likelihood_model.point_sources.keys()

        # this is a kludge at the moment. Model number may vary!
        num_models = len(
                self._analysis._likelihood_model.point_sources[sources_to_plot[0]].spectrum.main.composite.functions)

        all_contours = []
        for source in sources_to_plot:

            composite_model = self._analysis._likelihood_model.point_sources[source].spectrum.main.composite
            models = self._solve_for_component_flux(composite_model)

            contours_per_component = []
            for model in models:

                # Check the  type of function we want
                spectrum_type = self._get_spectrum_type(y_unit)

                # Set the x values to the proper unit


                # Retrieve the right flux function (phts, energy, vfv)
                flux_function = self._get_flux_function(spectrum_type, model, y_unit)

                # temporary list to store the propagated samples
                tmp = []

                # go through the thinned samples

                itr = range(0, n_samples, thin)
                with progress_bar(len(itr)) as p:
                    for i in itr:

                        # go through parameters
                        for par in self._analysis.samples.keys():
                            mod_par = par.split('.')[-1]
                            composite_model.free_parameters[mod_par].value = self._analysis.samples[par][i]

                        # get the flux for the this sample
                        tmp.append(flux_function(x_range))

                        p.increase()

                tmp = np.array(tmp).T

                # pull the highest denisty posterior at the choosen alpha level
                contours = np.array([self._analysis._hpd(mc, alpha=alpha) for mc in tmp])

                contours_per_component.append(contours)

            all_contours.append(contours_per_component)

        color = np.linspace(0., 1., len(sources_to_plot) * num_models)
        color_itr = 0

        # Ok, now everything is computed and we can convert back to
        # the x values to to frequency is need

        if self._convert_to_frequency:

            x_range = (x_range / constants.h).to(x_unit)

        if not summed:

            for contour_pc, source in zip(all_contours, sources_to_plot):

                model_names = [func.name for func in
                               self._analysis._likelihood_model.point_sources[source].spectrum.main.composite.functions]

                for contour, name in zip(contour_pc, model_names):

                    ax.fill_between(x_range,
                                    contour[:, 0] * y_unit,
                                    contour[:, 1] * y_unit,
                                    color=fit_cmap(color[color_itr]),
                                    alpha=contour_alpha,
                                    label='%s:%s' % (source, name))

                    ax.set_xscale('log')
                    ax.set_yscale('log')
                    if legend:
                        ax.legend(**kwargs)

                    color_itr += 1



        elif summed:

            color = np.linspace(0., 1., num_models)
            color_itr = 0
            # Assumes all sources have the same model!
            summed_contours = np.array(all_contours).sum(axis=0)

            # This is a kludge that assumes all sources have the same model!
            model_names = [func.name for func in
                           self._analysis._likelihood_model.point_sources[
                               sources_to_plot[0]].spectrum.main.composite.functions]

            for contour, name in zip(summed_contours, model_names):

                ax.fill_between(x_range,
                                contour[:, 0] * y_unit,
                                contour[:, 1] * y_unit,
                                color=fit_cmap(color[color_itr]),
                                alpha=contour_alpha,
                                label='%s' % (name))

                ax.set_xscale('log')
                ax.set_yscale('log')
                if legend:
                    ax.legend(**kwargs)

                color_itr += 1

        return fig

    def _propagate_full(self, flux_function, energy):
        """
        This function performs error propagation of the fits to fluxes

        :param flux_function: the function to propagate into
        :param energy: The energies at which to evaluate the function
        :return:
        """

        errors = []

        # Get the parameters from the minimizer

        parameters = self._analysis.minimizer.parameters

        # We will compute the error at each energy interval
        # so that we can plot the spread as a function of energy

        with progress_bar(len(energy)) as p:
            for ene in energy:

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

                        return flux_function(ene).value

                    # get the first derivatives and append them for some
                    # linear algebra

                    this_derivative = get_jacobian(tmpflux, parameter_best_fit_value, min_value, max_value)[0][0]

                    first_derivatives.append(this_derivative)

                first_derivatives = np.array(first_derivatives)

                # Now we take the inner product with the covariance matrix

                tmp = first_derivatives.dot(self._analysis.covariance_matrix)

                errors.append(np.sqrt(tmp.dot(first_derivatives)))

                p.increase()

        return errors

    def _x_unit_checker(self, x_unit):
        """
        Checks if the x unit is in energy or frequency

        :param x_unit: astropy unit string
        :return:
        """

        x_unit = u.Unit(x_unit)

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
    def _get_flux_function(spectrum_type, model, y_unit):
        """
        Returns the appropriate flux function based off input spectral units
        :param spectrum_type: str from _get_spectrum_type indicating the spectrum type to use
        :param model: a call to an astromodel function
        :param y_unit: astropy unit
        :return: function that calls the correct flux type
        """

        if spectrum_type == "badunit":

            raise InvalidUnitError("The y_unit provided is invalid")
            return

        elif spectrum_type == "phtflux":

            flux_function = lambda x: model(x).to(y_unit)


        elif spectrum_type == "eneflux":

            flux_function = lambda x: (x * model(x)).to(y_unit)


        elif spectrum_type == "vfvflux":

            flux_function = lambda x: (x ** 2 * model(x)).to(y_unit)

        return flux_function

    @staticmethod
    def _get_spectrum_type(y_unit):
        """

        :param y_unit: an astropy unit
        :return: str indicating the type of unit desired
        """

        pht_flux_unit = 1. / (u.keV * u.cm ** 2 * u.s)
        flux_unit = u.erg / (u.keV * u.cm ** 2 * u.s)
        vfv_unit = u.erg ** 2 / (u.keV * u.cm ** 2 * u.s)

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

                    raise InvalidUnitError("The y_unit provided is not a valid spectral quantity")

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
