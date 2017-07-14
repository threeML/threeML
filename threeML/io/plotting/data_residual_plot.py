import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from threeML.io.plotting.step_plot import step_plot
from threeML.config.config import threeML_config

from threeML.exceptions.custom_exceptions import custom_warnings

class ResidualPlot(object):

    def __init__(self,**kwargs):
        """

        :param show_residuals: to show the residuals
        :param ratio_residuals: to use ratios instead of sigma
        """


        self._ratio_residuals = False
        self._show_residuals = True


        if 'show_residuals' in kwargs:

            self._show_residuals = bool(kwargs.pop('show_residuals'))

        if 'ratio_residuals' in kwargs:
            self._ratio_residuals = bool(kwargs.pop('ratio_residuals'))


        # this lets you overplot other fits

        if 'model_subplot' in kwargs:

            model_subplot = kwargs.pop('model_subplot')

            # turn on or off residuals

            if self._show_residuals:

                assert len(
                    model_subplot) == 2, 'you have requested to overplot a model with residuals, but only provided one axis to plot'

                self._ax, self._ax1 = model_subplot

            else:

                self._ax = model_subplot


            self._fig = self._ax.get_figure()




            # turn on or off residuals


        else:

            if self._show_residuals:

                self._fig, (self._ax, self._ax1) = plt.subplots(2, 1,
                                                                sharex=True,
                                                                gridspec_kw={'height_ratios': [2, 1]},
                                                                **kwargs)

            else:

                self._fig, self._ax = plt.subplots(**kwargs)




    @property
    def show_residuals(self):

        return self._show_residuals

    @property
    def ratio_residuals(self):
        return self._ratio_residuals

    def add_model_step(self, xmin, xmax, xwidth, y, label, color='r'):
        """

        :param xmin:
        :param xmax:
        :param xwidth:
        :param y:
        :param residuals:
        :param label:
        :param color:
        :return:
        """



        step_plot(np.asarray(zip(xmin, xmax)),
                  y / xwidth,
                  self._ax, alpha=.8,
                  label=label, color=color)

    def add_model(self,x,y,label,color):
        """

        :param x:
        :param y:
        :param label:
        :param color:
        :return:
        """

        self._ax.plot(x,y,label=label,color=color,alpha=.8)


    def add_data(self, x, y, residuals, label, xerr=None, yerr=None, residual_yerr=None, color='r', show_data=True):
        """

        :param x:
        :param y:
        :param residuals:
        :param label:
        :param xerr:
        :param yerr:
        :param color:
        :return:
        """



        if show_data:
            self._ax.errorbar(x,
                        y,
                        yerr=yerr,
                        xerr=xerr,
                        fmt=threeML_config['residual plot']['error marker'],
                        markersize=threeML_config['residual plot']['error marker size'],
                        linestyle='',
                        elinewidth=threeML_config['residual plot']['error line width'],
                        alpha=.9,
                        capsize=0,
                        label=label,
                        color=color)



        #ax.plot(x, expected_model_magnitudes, label='%s Model' % data._name, color=model_color)

        #residuals = (expected_model_magnitudes - mag_errors) / mag_errors

        if self._show_residuals:

            if not self.ratio_residuals:
                residual_yerr = np.ones_like(residuals)

            self._ax1.axhline(0, linestyle='--', color='k')


            self._ax1.errorbar(x,
                         residuals,
                         yerr=residual_yerr,
                         capsize=0,
                         fmt=threeML_config['residual plot']['error marker'],
                         elinewidth=threeML_config['residual plot']['error line width'],
                         markersize=threeML_config['residual plot']['error marker size'],
                         color=color)


    def finalize(self, xlabel='x', ylabel='y',xscale='log',yscale='log', show_legend=True,invert_y=False):
        """

        :param xlabel:
        :param ylabel:
        :param xscale:
        :param yscale:
        :param show_legend:
        :return:
        """


        if show_legend:
            self._ax.legend(fontsize='x-small', loc=0)

        self._ax.set_ylabel(ylabel)

        self._ax.set_xscale(xscale)
        if yscale == 'log':

            self._ax.set_yscale(yscale, nonposy='clip')

        else:

            self._ax.set_yscale(yscale)

        if self._show_residuals:

            self._ax1.set_xscale(xscale)

            locator = MaxNLocator(prune='upper', nbins=5)
            self._ax1.yaxis.set_major_locator(locator)

            self._ax1.set_xlabel(xlabel)

            if self.ratio_residuals:
                custom_warnings.warn("Residuals plotted as ratios: beware that they are not statistical quantites, and can not be used to asses fit quality")
                self._ax1.set_ylabel("Residuals\n(fraction of model)")
            else:
                self._ax1.set_ylabel("Residuals\n($\sigma$)")


        else:

            self._ax.set_xlabel(xlabel)




            # This takes care of making space for all labels around the figure

        self._fig.tight_layout()

        # Now remove the space between the two subplots
        # NOTE: this must be placed *after* tight_layout, otherwise it will be ineffective

        self._fig.subplots_adjust(hspace=0)

        if invert_y:
            self._ax.set_ylim(self._ax.get_ylim()[::-1])


        return self._fig