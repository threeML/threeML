from __future__ import division
from builtins import zip
from builtins import range
from past.utils import old_div
from builtins import object
import numpy as np
from sherpa.astro import datastack
from sherpa.models import TableModel
from threeML.plugin_prototype import PluginPrototype
import matplotlib.pyplot as plt

__instrument_name = "All OGIP compliant instruments"


class Likelihood2SherpaTableModel(object):
    """Creates from a 3ML Likelihhod model a table model that can be used in sherpa.
    It should be used to convert a threeML.models.LikelihoodModel
    into a sherpa.models.TableModel such that values are evaluated
    at the boundaries of the energy bins for the pha data for which one wants to calculate
    the likelihood.

    Parameters
    -----------
    likelihoodModel :  threeML.models.LikelihoodModel
    model
    """

    def __init__(self, likelihoodModel):
        self.likelihoodModel = likelihoodModel
        self.table_model = TableModel("table.source")

        # fetch energies
        self.e_lo = np.array(datastack.get_arf(1).energ_lo)
        self.e_hi = np.array(datastack.get_arf(1).energ_hi)

        # TODO figure out what to do if the binning is different across the datastack
        self.table_model._TableModel__x = (
            self.e_lo
        )  # according to Sherpa TableModel specs, TBV

        # determine which sources are inside the ON region
        self.onPtSrc = []  # list of point sources in the ON region
        nPtsrc = self.likelihoodModel.getNumberOfPointSources()
        for ipt in range(nPtsrc):
            # TODO check if source is in the ON region?
            self.onPtSrc.append(ipt)
        self.onExtSrc = []  # list of extended sources in the ON region
        nExtsrc = self.likelihoodModel.getNumberOfExtendedSources()
        if nExtsrc > 0:
            raise NotImplemented("Cannot support extended sources yet")

    def update(self):
        """Update the model values.
        """
        vals = np.zeros(len(self.table_model._TableModel__x))
        for ipt in self.onPtSrc:
            vals += [
                self.likelihoodModel.pointSources[ipt].spectralModel.photonFlux(
                    bounds[0], bounds[1]
                )
                for bounds in zip(self.e_lo, self.e_hi)
            ]
            # integrated fluxes over same energy bins as for dataset, according to Sherpa TableModel specs, TBV
        self.table_model._TableModel__y = vals


class SherpaLike(PluginPrototype):
    """Generic plugin based on sherpa for data in OGIP format

    Parameters
    ----------
    name : str
    dataset name
    phalist : list of strings
    pha file names
    stat : str
    statistics to be used
    """

    def __init__(self, name, phalist, stat):
        # load data and set statistics
        self.name = name
        self.ds = datastack.DataStack()
        for phaname in phalist:
            self.ds.load_pha(phaname)
        # TODO add manual specs of bkg, arf, and rmf

        datastack.ui.set_stat(stat)

        # Effective area correction is disabled by default, i.e.,
        # the nuisance parameter is fixed to 1
        self.nuisanceParameters = {}

    def set_model(self, likelihoodModel):
        """Set model for the source region

        Parameters
        ----------
        likelihoodModel : threeML.models.LikelihoodModel
        sky model for the source region
        """
        self.model = Likelihood2SherpaTableModel(likelihoodModel)
        self.model.update()  # to initialize values
        self.model.ampl = 1.0
        self.ds.set_source(self.model.table_model)

    def _updateModel(self):
        """Updates the sherpa table model"""
        self.model.update()
        self.ds.set_source(self.model.table_model)

    def setEnergyRange(self, e_lo, e_hi):
        """Define an energy threshold for the fit
        which is different from the full range in the pha files

        Parameters
        ------------
        e_lo : float
        lower energy threshold in keV
        e_hi : float
        higher energy threshold in keV
        """
        self.ds.notice(e_lo, e_hi)

    def get_log_like(self):
        """Returns the current statistics value

        Returns
        -------------
        statval : float
        value of the statistics
        """
        self._updateModel()
        return -datastack.ui.calc_stat()

    def get_name(self):
        """Return a name for this dataset set during the construction

        Returns:
        ----------
        name : str
        name of the dataset
        """
        return self.name

    def get_nuisance_parameters(self):
        """Return a list of nuisance parameters.
        Return an empty list if there are no nuisance parameters.
        Not implemented yet.
        """
        # TODO implement nuisance parameters
        return list(self.nuisanceParameters.keys())

    def inner_fit(self):
        """Inner fit. Just a hack to get it to work now.
        Will be removed.
        """
        # TODO remove once the inner fit requirement has been dropped
        return self.get_log_like()

    def display(self):
        """creates plots comparing data to model
        """
        # datastack.ui.set_xlog()
        # datastack.ui.set_ylog()
        # self.ds.plot_data()
        # self.ds.plot_model(overplot=True)
        # TODO see if possible to show model subcomponents
        f, axarr = plt.subplots(2, sharex=True)
        f.subplots_adjust(hspace=0)
        plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
        energies = datastack.ui.get_data_plot(1).x
        dlne = np.log(energies[1:]) - np.log(energies[:-1])
        dlne = np.append(dlne[0], dlne)  # TODO do this properly for arbitrary binning
        de = np.power(10, np.log10(energies) + dlne) - np.power(
            10, np.log10(energies) - dlne
        )
        # TODO figure out what to do if different binning within the ds
        counts = np.zeros(len(energies))
        model = np.zeros(len(energies))
        bkg = np.zeros(len(energies))
        for id in self.ds.ids:
            counts += datastack.ui.get_data_plot(id).y * datastack.get_exposure(id) * de
            model += datastack.ui.get_model_plot(id).y * datastack.get_exposure(id) * de
            bkg += (
                datastack.ui.get_bkg_plot(id).y
                * datastack.get_exposure(id)
                * de
                * datastack.ui.get_bkg_scale(id)
            )
        tot = model + bkg
        axarr[0].errorbar(
            energies,
            counts,
            xerr=np.zeros(len(energies)),
            yerr=np.sqrt(counts),
            fmt="ko",
            capsize=0,
        )
        axarr[0].plot(energies, model, label="source")
        axarr[0].plot(energies, bkg, label="background")
        axarr[0].plot(energies, tot, label="total model")
        leg = axarr[0].legend()
        axarr[1].errorbar(
            energies[counts > 0],
            (old_div((counts - tot), tot))[counts > 0],
            xerr=np.zeros(len(energies[counts > 0])),
            yerr=(old_div(np.sqrt(counts), tot))[counts > 0],
            fmt="ko",
            capsize=0,
        )
        axarr[1].plot(energies, np.zeros(len(energies)), color="k", linestyle="--")
        axarr[0].set_xscale("log")
        axarr[1].set_xscale("log")
        axarr[0].set_yscale("log")
        axarr[0].set_ylabel("counts")
        axarr[1].set_ylabel("residuals (counts-model)/model")
        axarr[1].set_xlabel("energy (keV)")
