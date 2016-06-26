import astropy.units as u
from astropy.visualization import quantity_support

import matplotlib.pyplot as plt
import numpy as np
from sympy import Function
from sympy.abc import x
from sympy.solvers import solve
from sympy.utilities.lambdify import lambdify


class SpectralPlotter(object):
    """
    This class handles plotting of spectral fits and their associated contours.

    MLE fits are plotted with thier contours computed by propagating the covariance
    matrix through the function.

    Bayesian fits are plotted with thier marginal posteriors plotted through the function.

    :param analysis: an MLE or bayesian analysis object
    """

    def __init__(self, analysis):
        quantity_support()

        # Determine the type of analysis

        self._analysis_type = analysis._analysis_type

        self.analysis = analysis

    def plot_model(self, x_unit='keV', y_unit='erg/(cm2 keV s)', sources_to_plot=[], sum=False, xmin=10., xmax=1E4,
                   plot_num=1, thin=100, alpha=0.05):
        """
        Plot the model and the model contours for the selected sources.

        :param x_unit:
        :param y_unit:
        :param sources_to_plot:
        :param sum:
        :param xmin:
        :param xmax:
        :param plot_num:
        :param: thin
        :param: alpha
        """

        if self._analysis_type == "mle":
            self._plot_mle(x_unit, y_unit, sources_to_plot, sum, xmin, xmax, plot_num)

        elif self._analysis_type == "bayesian":
            self._plot_bayes(x_unit, y_unit, sources_to_plot, sum, xmin, xmax, plot_num, thin, alpha)

    def plot_components(self, x_unit='keV', y_unit='erg/(cm2 keV s)', sources_to_plot=[], sum=False, xmin=10., xmax=1E4,
                        plot_num=1, thin=100, alpha=0.05):
        """
        Plot the components of a fits and their associated contours

        :param x_unit: str with astropy spectral density units
        :param y_unit:
        :param sources_to_plot:
        :param sum:
        :param xmin:
        :param xmax:
        :param plot_num:

        """

        if self._analysis_type == "mle":

            self._plot_component_mle(x_unit, y_unit, sources_to_plot, sum, xmin, xmax, plot_num)

        elif self._analysis_type == "bayesian":

            self._plot_component_bayes(x_unit, y_unit, sources_to_plot, sum, xmin, xmax, plot_num, thin, alpha)

    def _plot_bayes(self, x_unit='keV', y_unit='erg/(cm2 keV s)', sources_to_plot=[], sum=False, xmin=10., xmax=1E4,
                    plot_num=1, thin=100, alpha=0.05):
        """

        :param x_unit:
        :param y_unit:
        :param sources_to_plot:
        :param sum:
        :param xmin:
        :param xmax:
        :param plot_num:
        :param thin:
        :param alpha:
        """

        x_unit = u.Unit(x_unit)
        y_unit = u.Unit(y_unit)

        x_values = np.logspace(np.log10(xmin), np.log10(xmax), 400)

        # Get the the number of samples
        n_samples = self.analysis.raw_samples.shape[0]

        fig = plt.figure(plot_num)
        ax = fig.add_subplot(111)

        # First see if we are plotting all the sources
        if not sources_to_plot:  # Assuming plot all sources

            sources_to_plot = self.analysis._likelihood_model.point_sources.keys()

        color = np.linspace(0., 1., len(sources_to_plot))
        color_itr = 0
        for source in sources_to_plot:

            # Get the spectrum first
            call = self.analysis._likelihood_model.point_sources[source].spectrum.main
            model = self.analysis._likelihood_model.point_sources[source].spectrum.main.shape

            # Check the  type of function we want
            spectrum_type = self._get_spectrum_type(y_unit)

            # Set the x values to the proper unit
            x_range = x_values * x_unit

            # Retrieve the right flux function (phts, energy, vfv)
            flux_function = self._get_flux_function(spectrum_type, model, y_unit)

            # temporary list to store the propagated samples
            tmp = []

            # go through the thinned samples
            for i in range(0, n_samples, thin):

                # go through parameters
                for par in self.analysis.samples.keys():
                    mod_par = par.split('.')[-1]
                    model.free_parameters[mod_par].value = self.analysis.samples[par][i]

                # get the flux for the this sample
                tmp.append(flux_function(x_range))

            tmp = np.array(tmp).T

            # pull the highest denisty posterior at the choosen alpha level
            contours = np.array([self.analysis._hpd(mc, alpha=alpha) for mc in tmp])

            ax.fill_between(x_range,
                            contours[:, 0]*y_unit,
                            contours[:, 1]*y_unit,
                            color=plt.cm.Set1(color[color_itr]),
                            alpha=.6)

            ax.set_xscale('log')
            ax.set_yscale('log')

            color_itr += 1

    def _plot_mle(self, x_unit='keV', y_unit='erg/(cm2 keV s)', sources_to_plot=[], sum=False,
                  xmin=10., xmax=1E4, plot_num=1):

        """

        :type sources_to_plot: object
        """
        self.analysis.restore_best_fit()

        x_unit = u.Unit(x_unit)
        y_unit = u.Unit(y_unit)

        # Initialize plotting arrays
        y_values = []
        x_values = np.logspace(np.log10(xmin), np.log10(xmax), 300)
        errors = []

        fig = plt.figure(plot_num)
        ax = fig.add_subplot(111)

        # First see if we are plotting all the sources
        if not sources_to_plot:  # Assuming plot all sources

            sources_to_plot = self.analysis.likelihood_model.point_sources.keys()

        for source in sources_to_plot:
            # Get the spectrum first
            call = self.analysis.likelihood_model.point_sources[source].spectrum.main
            model = self.analysis.likelihood_model.point_sources[source].spectrum.main.shape

            # Check the  type of function we want
            spectrum_type = self._get_spectrum_type(y_unit)

            x_range = x_values * x_unit

            flux_function = self._get_flux_function(spectrum_type, model, y_unit)

            err = self._propagate_full(source, flux_function, x_range)

            y_values.append(flux_function(x_range))

            errors.append(err * y_unit)

        for y_val, err in zip(y_values, errors):
            ax.loglog(x_range,
                      y_val,
                      color='#F4FA58',
                      lw=.6)

            up_y = y_val + err
            down_y = y_val - err

            ax.fill_between(x_range,
                            down_y,
                            up_y,
                            facecolor='#0080FF',
                            alpha=0.5,
                            lw=0.)

            ax.set_xscale('log')
            ax.set_yscale('log')

    def _plot_component_mle(self, x_unit='keV', y_unit='erg/(cm2 keV s)', sources_to_plot=[], sum=True, xmin=10.,
                            xmax=1E4, plot_num=1):

        self.analysis.restore_best_fit()

        x_unit = u.Unit(x_unit)
        y_unit = u.Unit(y_unit)

        # Initialize plotting arrays
        y_values = []
        x_values = np.logspace(np.log10(xmin), np.log10(xmax), 300)
        errors = []

        fig = plt.figure(plot_num)
        ax = fig.add_subplot(111)

        # First see if we are plotting all the sources
        if sources_to_plot == []:  # Assuming plot all sources

            sources_to_plot = self.analysis.likelihood_model.point_sources.keys()

        # if components == []: # Assuming plot all sources

        #    sources_to_plot = self.analysis.likelihood_model.point_sources.keys()


        for source in sources_to_plot:

            composite_model = self.analysis.likelihood_model.point_sources[source].spectrum.main.composite
            models = self._solve_for_component_flux(composite_model)

            # Check the type of function we want
            spectrum_type = self._get_spectrum_type(y_unit)

            x_range = x_values * x_unit
            y_vals_per_comp = []
            errors_per_comp = []
            for model in models:
                flux_function = self._get_flux_function(spectrum_type, model, y_unit)
                err = self._propagate_full(source, flux_function, x_range)

                y_vals_per_comp.append(flux_function(x_range))

                errors_per_comp.append(err * y_unit)

            y_values.append(y_vals_per_comp)
            errors.append(errors_per_comp)

        color = np.linspace(0., 1., len(sources_to_plot) * len(models))
        color_itr = 0
        for y_val_pc, err_pc in zip(y_values, errors):

            for y_val, err in zip(y_val_pc, err_pc):
                pos_mask = np.logical_and(y_val > 0, err > 0)

                ax.fill_between(x_range[pos_mask],
                                y_val[pos_mask] - err[pos_mask],
                                y_val[pos_mask] + err[pos_mask],
                                color=plt.cm.Set3(color[color_itr]),
                                alpha=.8)

                ax.loglog(x_range[pos_mask],
                          y_val[pos_mask], color=plt.cm.Set1(color[color_itr]), lw=.8)

                ax.set_xscale('log')
                ax.set_yscale('log')
                color_itr += 1

    def _plot_component_bayes(self, x_unit='keV', y_unit='erg/(cm2 keV s)', sources_to_plot=[], sum=False, xmin=10.,
                              xmax=1E4, plot_num=1, thin=100, alpha=0.05):
        """

        :param x_unit:
        :param y_unit:
        :param sources_to_plot:
        :param sum:
        :param xmin:
        :param xmax:
        :param plot_num:
        :param thin:
        :param: alpha
        :return:
        """

        x_unit = u.Unit(x_unit)
        y_unit = u.Unit(y_unit)

        x_values = np.logspace(np.log10(xmin), np.log10(xmax), 400)

        # Get the the number of samples
        n_samples = self.analysis.raw_samples.shape[0]

        fig = plt.figure(plot_num)
        ax = fig.add_subplot(111)

        # First see if we are plotting all the sources
        if not sources_to_plot:  # Assuming plot all sources

            sources_to_plot = self.analysis._likelihood_model.point_sources.keys()

        # this is a kludge at the moment. Model number may vary!
        num_models = len(
            self.analysis._likelihood_model.point_sources[sources_to_plot[0]].spectrum.main.composite.functions)

        color = np.linspace(0., 1., len(sources_to_plot) * num_models)
        color_itr = 0
        for source in sources_to_plot:

            composite_model = self.analysis._likelihood_model.point_sources[source].spectrum.main.composite
            models = self._solve_for_component_flux(composite_model)

            # Check the type of function we want
            spectrum_type = self._get_spectrum_type(y_unit)

            x_range = x_values * x_unit
            y_vals_per_comp = []
            errors_per_comp = []
            for model in models:

                # Check the  type of function we want
                spectrum_type = self._get_spectrum_type(y_unit)

                # Set the x values to the proper unit
                x_range = x_values * x_unit

                # Retrieve the right flux function (phts, energy, vfv)
                flux_function = self._get_flux_function(spectrum_type, model, y_unit)

                # temporary list to store the propagated samples
                tmp = []

                # go through the thinned samples
                for i in range(0, n_samples, thin):

                    # go through parameters
                    for par in self.analysis.samples.keys():
                        mod_par = par.split('.')[-1]
                        composite_model.free_parameters[mod_par].value = self.analysis.samples[par][i]

                    # get the flux for the this sample
                    tmp.append(flux_function(x_range))

                tmp = np.array(tmp).T

                # pull the highest denisty posterior at the choosen alpha level
                contours = np.array([self.analysis._hpd(mc, alpha=alpha) for mc in tmp])

                ax.fill_between(x_range,
                                contours[:, 0]*y_unit,
                                contours[:, 1]*y_unit,
                                color=plt.cm.Set1(color[color_itr]),
                                alpha=.6)

                ax.set_xscale('log')
                ax.set_yscale('log')

                color_itr += 1

    @staticmethod
    def _derivative(f):
        def df(x):
            h = 0.1e-7
            return (f(x + h / 2) - f(x - h / 2)) / h

        return df

    def _propagate_full(self, source, flux_function, energy):

        errors = []
        model = self.analysis.likelihood_model.point_sources[source]
        for ene in energy:

            first_derivatives = []

            for par in model.spectrum.main.shape.free_parameters.keys():
                self.analysis.restore_best_fit()

                parameter_best_fit_value = model.spectrum.main.shape.free_parameters[par].value

                def tmpflux(current_value):
                    model.spectrum.main.shape.free_parameters[par].value = current_value

                    return flux_function(ene).value

                this_derivate = self._derivative(tmpflux)

                first_derivatives.append(this_derivate(parameter_best_fit_value))

            first_derivatives = np.array(first_derivatives)

            tmp = first_derivatives.dot(self.analysis.covariance_matrix)

            errors.append(np.sqrt(tmp.dot(first_derivatives)))

        return errors

    @staticmethod
    def _get_flux_function(spectrum_type, model, y_unit):

        if spectrum_type == "badunit":

            print "The y_unit provided is invalid"
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
        Determines the type of spectral denisty to plot
        :rtype: flux function of correct units
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

                    return "badunit"

    def _solve_for_component_flux(self, composite_model):

        base_expression = composite_model.expression
        replicated_expression = composite_model.expression

        num_models = len(composite_model.functions)
        mod_solve = []
        function_dict = {}

        # First build the expressions and create a dictionary for
        # the component functions to be referecenced

        for i, func in enumerate(composite_model.functions):
            # Need to replace all the strings correctly
            replicated_expression = replicated_expression.replace("%s{%d}" % (func.name, i + 1),
                                                                  "mod_solve[%d](x)" % i)

            function_dict["%s_%d" % (func.name, i + 1)] = func

            mod_solve.append(Function("%s_%d" % (func.name, i + 1)))

        function_dict['total'] = composite_model

        replicated_expression += "- mod_solve[%d](x)" % num_models

        mod_solve.append(Function("total"))

        solutions = []
        for i, func in enumerate(composite_model.functions):
            solutions.append(solve(eval(replicated_expression), str(mod_solve[i]) + '(x)')[0])

        component_flux = [lambdify(x, sol, function_dict) for sol in solutions]

        return component_flux
