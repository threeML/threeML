from __future__ import division

import collections
import os
from builtins import object, range, zip
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import astropy.io.fits as fits
import BinnedAnalysis
import matplotlib.pyplot as plt
import numpy
import pyLikelihood as pyLike
import UnbinnedAnalysis
from astromodels import Model, Parameter
from GtBurst import FuncFactory, LikelihoodComponent
from matplotlib import gridspec
from past.utils import old_div

from threeML.config.config import threeML_config
from threeML.config.plotting_structure import BinnedSpectrumPlot
from threeML.io.file_utils import get_random_unique_name
from threeML.io.logging import setup_logger
from threeML.io.package_data import get_path_of_data_file
from threeML.io.plotting.data_residual_plot import ResidualPlot
from threeML.io.suppress_stdout import suppress_stdout
from threeML.plugin_prototype import PluginPrototype
from threeML.utils.statistics.gammaln import logfactorial
from threeML.utils.statistics.stats_tools import Significance

plt.style.use(str(get_path_of_data_file("threeml.mplstyle")))


__instrument_name = "Fermi LAT (standard classes)"



log = setup_logger(__name__)

class MyPointSource(LikelihoodComponent.GenericSource):
    def __init__(self, source, name, temp_file):
        """Container class for indexing likelihood sources

        :param source: 
        :param name: 
        :param temp_file: 
        :returns: 
        :rtype: 

        """
        self.source = source
        self.source.name = name
        self.temp_file = temp_file

        super(MyPointSource, self).__init__()

@dataclass
class LikelihoodModelConverter:

    likelihood_model: Model
    irfs: List[str]
    source_name: Optional[str] = None
    

    def set_file_spectrum_energies(self, emin_kev, emax_kev, n_energies):
        """Make a log spaced array from emin_kev, to emax_kev with n_energies bins

        :param emin_kev: starting energy in keV
        :param emax_kev: ending energy in keV
        :param n_energies: number of energy bins
        :returns: 
        :rtype: 

        """

        self.energies_kev = numpy.logspace(numpy.log10(emin_kev), numpy.log10(emax_kev), n_energies)

    def write_xml(self, xmlfile, ra, dec, roi) -> List[str]:
        """Loop through all the sources in the likelihood model and generate a FileSpectrum
           for all of them. This is necessary to allow the FermiLATLike class
           to update the spectrum in pyLikelihood without having to write and read a .xml file
           on the disk

        :param xmlfile: 
        :param ra: 
        :param dec: 
        :param roi: 
        :returns: 
        :rtype: 

        """

        all_sources_for_pylike = []

        temp_files = []

        if self.source_name is None:

            n_pt_src = self.likelihood_model.get_number_of_point_sources()

            for ip in range(n_pt_src):

                this_src = self._make_file_spectrum(ip)

                all_sources_for_pylike.append(this_src)
                temp_files.append(this_src.temp_file)

        else:
            # We pass from the model just one source

            log.info(f'Setting single point source {self.likelihood_model} ... ')

            index = self.likelihood_model.point_sources.keys().index(self.source_name)
            this_src = self._make_file_spectrum(index)
            all_sources_for_pylike.append(this_src)

            temp_files.append(this_src.temp_file)

        # Now the same for extended sources
        n_ext_src = self.likelihood_model.get_number_of_extended_sources()

        if n_ext_src > 0:

            log.error("Cannot support extended sources yet!")
            
            raise NotImplemented("Cannot support extended sources yet!")

        iso = LikelihoodComponent.IsotropicTemplate(self.irfs)

        iso.source.spectrum.Normalization.max = 1.5
        iso.source.spectrum.Normalization.min = 0.5
        iso.source.spectrum.setAttributes()

        all_sources_for_pylike.append(iso)

        # Get a temporary filename which is guaranteed to be unique
        self._unique_filename = get_random_unique_name()

        gal = LikelihoodComponent.GalaxyAndExtragalacticDiffuse(
            self.irfs, ra, dec, 2.5 * roi, cutout_name=self._unique_filename
        )
        gal.source.spectrum.Value.max = 1.5
        gal.source.spectrum.Value.min = 0.5
        gal.source.spectrum.setAttributes()

        all_sources_for_pylike.append(gal)

        # Now generate the xml file with also the Galactic and Isotropic diffuse
        # templates
        xml = LikelihoodComponent.LikelihoodModel()
        xml.addSources(*all_sources_for_pylike)
        xml.writeXML(xmlfile)

        return temp_files

    def _make_file_spectrum(self, ip) -> MyPointSource:
        """Write the xml code for one point source. The model used is the FileFunction,
        so we can accomodate any model from astromodels

        :param ip: identification number for the source
        :returns: MyPointSource
        :rtype: 

        """

        name = self.likelihood_model.get_point_source_name(ip)
        values = self.likelihood_model.get_point_source_fluxes(ip, 
                 self.energies_kev)

        temp_name = "__%s_%s.txt" % (name, get_random_unique_name())

        with open(temp_name, "w+") as f:
            for e, v in zip(self.energies_kev, values):
                # Gtlike needs energies in MeV and fluxes in ph/MeV/cm2)

                f.write("%s %s\n" % (e / 1000.0, v * 1000.0))

        # p                         = fileFunction.parameter("Normalization")
        # p.setBounds(1-float(effAreaAllowedSize),1+effAreaAllowedSize)

        # Now generate the XML source wrapper
        # This is convoluted, but that's the ST way of doing things...
        # The final product is a class with a writeXml method

        src = "\n".join(
            (
                ('<source name= "%s" ' % name) + 'type="PointSource">',
                '   <spectrum type="PowerLaw2"/>',
                "   <!-- point source units are cm^-2 s^-1 MeV^-1 -->",
                '   <spatialModel type="SkyDirFunction"/>',
                "</source>\n",
            )
        )
        src = FuncFactory.minidom.parseString(
            src).getElementsByTagName("source")[0]
        src = FuncFactory.Source(src)

        src.spectrum = FuncFactory.FileFunction()
        src.spectrum.file = temp_name
        src.spectrum.parameters["Normalization"].value = 1.0
        src.spectrum.parameters["Normalization"].max = 1.1
        src.spectrum.parameters["Normalization"].min = 0.9
        src.spectrum.parameters["Normalization"].free = False
        src.spectrum.setAttributes()
        src.deleteChildElements("spectrum")
        src.node.appendChild(src.spectrum.node)

        src.spatialModel = FuncFactory.SkyDirFunction()
        src.deleteChildElements("spatialModel")
        src.node.appendChild(src.spatialModel.node)

        ra, dec = self.likelihood_model.get_point_source_position(ip)

        src.spatialModel.RA.value = ra
        src.spatialModel.DEC.value = dec
        src.spatialModel.setAttributes()
        src.setAttributes()

        return MyPointSource(src, name, temp_name)


class FermiLATUnpickler(object):

    def __call__(self, name, event_file, ft2_file, livetime_cube_file, kind, exposure_map_file, likelihood_model,
                 inner_minimization):
        """Create an instance of the FermiLATLike pligin

        :param name: name for the plugin
        :param event_file: FT1 file congtaining the events
        :param ft2_file: FT2 file containing  the pointing history of the satellite
        :param livetime_cube_file: Livetime cube file (created by the fermitool gtltcube)
        :param kind: Analysis type, BINNED or UNBINNED
        :param exposure_map_file: exposure map file created by the fermitool gtexpmap
        :param likelihood_model: file containing the likelihood model
        :param inner_minimization: Turn on/off the minimization of the internal Fermi
        parameters
        :returns: 
        :rtype: 

        """

        instance = FermiLATLike(name, event_file, ft2_file, livetime_cube_file, kind, exposure_map_file)

        instance.set_inner_minimization(inner_minimization)

        instance.set_model(likelihood_model)

        return instance

class FermiLATLike(PluginPrototype):

    def __init__(self,
                 name: str,
                 event_file: str,
                 ft2_file: str,
                 livetime_cube_file: str,
                 kind,
                 exposure_map_file: Optional[str] = None,
                 source_maps=None,
                 binned_expo_map=None,
                 source_name: Optional[str] = None):
        """

        Fermi-LAT plugin utilizing the low-end Fermi ST stack. Allows for binned
        or unbinned analysis
        :param name: name for the plugin
        :param event_file: FT1 file congtaining the events
        :param ft2_file: FT2 file containing  the pointing history of the satellite
        :param livetime_cube_file: Livetime cube file (created by the fermitool gtltcube)
        :param kind: Analysis type, BINNED or UNBINNED
        :param source_maps:: source map file created by the fermitool gtsrcmap
        :param binned_expo_map: binned exposure map
        :param source_name: Name of  the source to be fitted
        :returns:
        :rtype:

        """

        # Initially the nuisance parameters dict is empty, as we don't know yet
        # the likelihood model. They will be updated in set_model

        super(FermiLATLike, self).__init__(name, {})

        # Read the ROI cut
        cc = pyLike.RoiCuts()
        cc.readCuts(event_file, "EVENTS")
        self.ra, self.dec, self.rad = cc.roiCone()

        # Read the IRF selection
        c = pyLike.Cuts(event_file, "EVENTS")
        self.irf = c.CALDB_implied_irfs()

        self._ft2_file = ft2_file
        self._livetime_cube_file = livetime_cube_file

        # These are the boundaries and the number of energies for the computation
        # of the model (in keV)
        self.emin = 1e4
        self.emax = 5e8
        self.n_energies = 200

        with fits.open(event_file) as file:
            self.__observation_duration = file[0].header['TSTOP'] - file[0].header['TSTART']


        # This is the limit on the effective area correction factor,
        # which is a multiplicative factor in front of the whole model
        # to account for inter-calibration issues. By default it can vary
        # by 10%. This can be changed by issuing:
        # FermiLATUnbinnedLikeInstance.effCorrLimit = [new limit]
        # where for example a [new limit] of 0.2 allow for an effective
        # area correction up to +/- 20 %

        self.eff_corr_limit = 0.1

        if kind.upper() != "UNBINNED" and kind.upper() != "BINNED":

            log.error("Accepted values for the kind parameter are: "
                + "binned, unbinned. You specified: %s" % (kind))
            
            raise RuntimeError(
                "Accepted values for the kind parameter are: "
                + "binned, unbinned. You specified: %s" % (kind)
            )

        else:

            self.kind = kind.upper()

        if kind.upper() == "UNBINNED":

            if exposure_map_file is None:

                log.error("You have to provide an exposure map")
                
                raise AssertionError()

            self._event_file = event_file
            self._exposure_map_file = exposure_map_file
            # Read the files and generate the pyLikelihood object
            self.obs = UnbinnedAnalysis.UnbinnedObs(
                self._event_file,
                self._ft2_file,
                expMap=self._exposure_map_file,
                expCube=self._livetime_cube_file,
                irfs=self.irf)

        elif kind.upper() == "BINNED":
            
            if source_maps is  None:

                log.error("You have to provide a source map")
                raise AssertionError()
                
            if binned_expo_map is None:

                log.error("You have to provided a (binned) exposure map")
                raise AssertionError()
                

            self._source_maps = source_maps
            self._binned_expo_map = binned_expo_map

            self.obs = BinnedAnalysis.BinnedObs(
                srcMaps=self._source_maps,
                expCube=self._livetime_cube_file,
                binnedExpMap=self._binned_expo_map,
                irfs=self.irf)
        

        # Activate inner minimization by default
        self.set_inner_minimization(True)

        self._source_name: str = source_name

    

    def set_model(self, likelihood_model: Model, source_name: Optional[str]=None):
        """
        Set the model to be used in the joint minimization.
        Must be a likelihood_model instance.

        This method can also set or override a previously set source name.
        :param likelihood_model: Model of the ROI for the likelihood analysis
        :param source_name: source to be fitted
        :return:
        """

        # with suppress_stdout():

        if self._source_name is not None:
            if (source_name is not None) and (source_name != self._source_name):
                log.warning('Changing target source from %s to %s' %
                      (self._source_name, source_name))
                self._source_name = source_name

            if self._source_name not in likelihood_model.point_sources:

                log.error(f'Source {self._source_name} is not a source in the likelihood model!')

                raise AssertionError()
                
                
        self._lmc: LikelihoodModelConverter = LikelihoodModelConverter(likelihood_model, self.irf, source_name=self._source_name)

        self._lmc.set_file_spectrum_energies(self.emin, self.emax, self.n_energies)

        xml_file = str("%s.xml" % get_random_unique_name())
        temp_files = self._lmc.write_xml(xml_file, self.ra, self.dec, self.rad)

        if self.kind == "BINNED":
            self.like = BinnedAnalysis.BinnedAnalysis(
                self.obs, xml_file, optimizer="DRMNFB"
            )

        else:
            self.like = UnbinnedAnalysis.UnbinnedAnalysis(
                self.obs, xml_file, optimizer="DRMNFB"
            )

        self.likelihood_model: Model = likelihood_model

        # Here we need also to compute the logLike value, so that the model
        # in the XML file will be changed if needed
        dumb = self.get_log_like()

        # Since now the Galactic template is in RAM, we can remove the temporary file

        log.debug("Removing non-needed files")
        
        Path(self._lmc._unique_filename).unlink()

        log.debug(r"removed {self._lmc._unique_filename}")
        
        Path(xml_file).unlink()

        log.debug(r"removed {xml_file}")
        
        # Delete temporary spectral files
        for temp_file in temp_files:

            Path(temp_file).unlink()

            log.debug(r"removed {temp_file}")
            
        # Build the list of the nuisance parameters
        new_nuisance_parameters = self._set_nuisance_parameters()

        self.update_nuisance_parameters(new_nuisance_parameters)

    def clear_source_name(self) -> None:
        if self._source_name is not None:
            
            log.info(f'Clearing {self._source_name} as a source for this plugin.')
            
            self._source_name = None
            
        else:
            
            log.error('Source not named. Use set_model to set a source.')

            raise AssertionError()
            
    def get_name(self) -> str:
        """
        Return a name for this dataset (likely set during the constructor)
        """
        return self.name

    def set_inner_minimization(self, flag: bool) -> None:

        """
        Turn on the minimization of the internal Fermi
        parameters

        :param flag: turing on and off the minimization  of the Fermi internal parameters
        :type flag: bool
        :returns: 

        """
        self._fit_nuisance_params: bool = bool(flag)

        for parameter in self.nuisance_parameters:

            self.nuisance_parameters[parameter].free = self._fit_nuisance_params

    def inner_fit(self):
        """
        This is used for the profile likelihood. Keeping fixed all parameters in the
        modelManager, this method minimize the logLike over the remaining nuisance
        parameters, i.e., the parameters belonging only to the model for this
        particular detector
        """

        return self.get_log_like()

    def _update_gtlike_model(self):
        """
        #Slow! But no other options at the moment
        self.like.write_xml(self.xmlModel)
        self.like.logLike.reReadXml(self.xmlModel)
        """

        energies = self._lmc.energies_kev

        if self._source_name is not None:

            # create a tuple with only this source
            
            itr = ([self.likelihood_model.point_sources.keys().index(self._source_name)] , [self._source_name])

        else:

            itr = enumerate(self.likelihood_model.point_sources.keys())

            
        for id, src_name in itr:

            values = self.likelihood_model.get_point_source_fluxes(
                id, energies, tag=self._tag
            )

            # on the second iteration, self.like doesn't have the second src_name defined so that needs to be carried from flags
            gtlike_src_model = self.like[src_name]

            my_function = gtlike_src_model.getSrcFuncs()["Spectrum"]

            my_file_function = pyLike.FileFunction_cast(my_function)

            my_file_function.setParam("Normalization", 1)

            # Cap the values to avoid numerical errors

            capped_values = numpy.minimum(
                numpy.maximum(values * 1000, 1e-25), 1e5)

            my_file_function.setSpectrum(energies / 1000.0, capped_values)

            gtlike_src_model.setSpectrum(my_file_function)

                # TODO: extended sources

        self.like.syncSrcParams()

    def get_log_like(self):
        """
        Return the value of the log-likelihood with the current values for the
        parameters stored in the ModelManager instance
        """

        self._update_gtlike_model()

        if self._fit_nuisance_params:

            for parameter in self.nuisance_parameters:
                self.set_nuisance_parameter_value(parameter, self.nuisance_parameters[parameter].value)

            self.like.syncSrcParams()

        log_like = self.like.logLike.value()

        return log_like - logfactorial(int(self.like.total_nobs()))

    #
    def __reduce__(self):

        return (
            FermiLATUnpickler(), 
            (
                self.name, 
                self._event_file, 
                self._ft2_file, 
                self._livetime_cube_file, 
                self.kind,
                self._exposure_map_file, 
                self.likelihood_model, 
                self._fit_nuisance_params
            ),
        )
    
    # def __setstate__(self, likelihood_model):
    #
    #     import pdb;pdb.set_trace()
    #
    #     self.set_model(likelihood_model)

    def get_model_and_data(self):

        fake = numpy.array([])

        return fake, fake, fake, fake

    def get_observation_duration(self) -> float:

        return self.__observation_duration

    def display(self):

        e1 = self.like.energies[:-1]
        e2 = self.like.energies[1:]

        ec = (e1 + e2) / 2.0
        de = (e2 - e1) / 2.0

        sum_model = numpy.zeros_like(
            self.like._srcCnts(self.like.sourceNames()[0]))

        fig = plt.figure()

        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
        gs.update(hspace=0)

        sub = plt.subplot(gs[0])

        for source_name in self.like.sourceNames():
            sum_model = sum_model + self.like._srcCnts(source_name)

            sub.plot(ec, self.like._srcCnts(source_name), label=source_name)

        sub.plot(ec, sum_model, label="Total Model")

        sub.errorbar(
            ec,
            self.like.nobs,
            xerr=de,
            yerr=numpy.sqrt(self.like.nobs),
            fmt=".",
            label="Counts",
        )

        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, numpoints=1)

        # Residuals

        sub1 = plt.subplot(gs[1])
        
        # Using model variance to account for low statistic

        resid = old_div((self.like.nobs - sum_model), sum_model)
        resid_err=old_div(numpy.sqrt(self.like.nobs), sum_model)

        sub1.axhline(0, linestyle="--")
        sub1.errorbar(
            ec,
            resid,
            xerr=de,
            yerr=resid_err,
            capsize=0,
            fmt=".",
        )

        sub.set_xscale("log")
        sub.set_yscale("log", nonpositive="clip")


        sub.set_ylabel("Counts per bin")

        sub1.set_xscale("log")

        sub1.set_xlabel("Energy (MeV)")
        sub1.set_ylabel("(data - mo.) / mo.")

        sub.set_xticks([])

        fig.tight_layout()

        # Now remove the space between the two subplots
        # NOTE: this must be placed *after* tight_layout, otherwise it will be ineffective

        fig.subplots_adjust(hspace=0)

        return fig

    def display_model(
        self,
        data_color: str = "k",
        model_color: str = "r",
        background_color: str = "b",
        show_data: bool = True,
        show_residuals: bool = True,
        ratio_residuals: bool = False,
        show_legend: bool = True,
        model_label: Optional[str] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        data_kwargs: Optional[
        Dict[str, Any]] = None,
        background_kwargs: Optional[Dict[str, Any]] = None,
        ** kwargs
    ) -> ResidualPlot:
        """
        Plot the current model with or without the data and the residuals. Multiple models can be plotted by supplying
        a previous axis to 'model_subplot'.

        Example usage:

        fig = data.display_model()

        fig2 = data2.display_model(model_subplot=fig.axes)


        :param data_color: the color of the data
        :param model_color: the color of the model
        :param show_data: (bool) show_the data with the model
        :param show_residuals: (bool) shoe the residuals
        :param ratio_residuals: (bool) use model ratio instead of residuals
        :param show_legend: (bool) show legend
        :param model_label: (optional) the label to use for the model default is plugin name
        :param model_kwargs: plotting kwargs affecting the plotting of the model
        :param data_kwargs:  plotting kwargs affecting the plotting of the data and residuls
        :param background_kwargs: plotting kwargs affecting the plotting of the background
        :return:
        """

        # set up the default plotting

        _default_model_kwargs = dict(color=model_color, alpha=1)

        _default_background_kwargs = dict(
            color=background_color, alpha=1, ls="--")

        _sub_menu = threeML_config.plotting.residual_plot

        _default_data_kwargs = dict(
            color=data_color,
            alpha=1,
            fmt=_sub_menu.marker,
            markersize=_sub_menu.size,
            ls="",
            elinewidth=2,#_sub_menu.linewidth,
            capsize=0,
        )

        # overwrite if these are in the confif

        _kwargs_menu: BinnedSpectrumPlot = threeML_config.plugins.ogip.fit_plot

        if _kwargs_menu.model_mpl_kwargs is not None:

            for k, v in _kwargs_menu.model_mpl_kwargs.items():

                _default_model_kwargs[k] = v

        if _kwargs_menu.data_mpl_kwargs is not None:

            for k, v in _kwargs_menu.data_mpl_kwargs.items():

                _default_data_kwargs[k] = v

        if _kwargs_menu.background_mpl_kwargs is not None:

            for k, v in _kwargs_menu.background_mpl_kwargs.items():

                _default_background_kwargs[k] = v

        if model_kwargs is not None:

            assert type(model_kwargs) == dict, "model_kwargs must be a dict"

            for k, v in list(model_kwargs.items()):

                if k in _default_model_kwargs:

                    _default_model_kwargs[k] = v

                else:

                    _default_model_kwargs[k] = v

        if data_kwargs is not None:

            assert type(data_kwargs) == dict, "data_kwargs must be a dict"

            for k, v in list(data_kwargs.items()):

                if k in _default_data_kwargs:

                    _default_data_kwargs[k] = v

                else:

                    _default_data_kwargs[k] = v

        if background_kwargs is not None:

            assert type(
                background_kwargs) == dict, "background_kwargs must be a dict"

            for k, v in list(background_kwargs.items()):

                if k in _default_background_kwargs:

                    _default_background_kwargs[k] = v

                else:

                    _default_background_kwargs[k] = v

        # since we define some defualts, lets not overwrite
        # the users

        _duplicates = (("ls", "linestyle"), ("lw", "linewidth"))

        for d in _duplicates:

            if (d[0] in _default_model_kwargs) and (d[1] in _default_model_kwargs):

                _default_model_kwargs.pop(d[0])

            if (d[0] in _default_data_kwargs) and (d[1] in _default_data_kwargs):

                _default_data_kwargs.pop(d[0])

            if (d[0] in _default_background_kwargs) and (d[1] in _default_background_kwargs):

                _default_background_kwargs.pop(d[0])

        if model_label is None:
            model_label = "%s Model" % self._name

        residual_plot = ResidualPlot(
            show_residuals=show_residuals, ratio_residuals=ratio_residuals, **kwargs
        )

        e1 = self.like.energies[:-1]*1000.0 # this has to be in keV
        e2 = self.like.energies[1:]*1000.0  # this has to be in keV

        ec = (e1 + e2) / 2.0
        de = (e2 - e1) / 2.0

        conversion_factor = de * self.__observation_duration
        sum_model = numpy.zeros_like(
            self.like._srcCnts(self.like.sourceNames()[0]))

        sum_backgrounds = numpy.zeros_like(
            self.like._srcCnts(self.like.sourceNames()[0]))

        for source_name in self.like.sourceNames():

            source_counts = self.like._srcCnts(source_name)

            sum_model     = sum_model + source_counts
            if source_name != self._source_name:
                sum_backgrounds = sum_backgrounds + source_counts

            residual_plot.add_model(
                ec, source_counts/conversion_factor, label=source_name#, **_default_model_kwargs
            )
            #sub.plot(ec, self.like._srcCnts(source_name), label=source_name)
        residual_plot.add_model(
            ec, sum_model/conversion_factor, label='Total Model', **_default_model_kwargs
        )

        #sub.plot(ec, sum_model, label="Total Model")

        y         = self.like.nobs
        y_err     = numpy.sqrt(y)

        significance_calc = Significance(
            Non=y,
            Noff=sum_backgrounds)

        if ratio_residuals:
            resid     = old_div((self.like.nobs - sum_model), sum_model)
            resid_err = old_div(y_err, sum_model)
        else:
            #resid     = significance_calc.li_and_ma()
            resid     = significance_calc.known_background()
            resid_err = numpy.ones_like(resid)


        y         = y/conversion_factor
        y_err     = y_err/conversion_factor

        residual_plot.add_data(
            ec[y>0],
            y[y>0],
            resid[y>0],
            residual_yerr=resid_err[y>0],
            yerr=y_err[y>0],
            xerr=de[y>0],
            label=self._name,
            show_data=show_data,
            **_default_data_kwargs,
        )
        y_label = "Net rate\n(counts s$^{-1}$ keV$^{-1}$)"

        return residual_plot.finalize(
            xlabel="Energy\n(keV)",
            ylabel=y_label,
            xscale="log",
            yscale="log",
            show_legend=show_legend,
        )



    def _set_nuisance_parameters(self):

        # Get the list of the sources
        sources = list(self.like.model.srcNames)

        free_param_names = []
        for src_name in sources:
            thisNamesV = pyLike.StringVector()
            thisSrc = self.like.logLike.getSource(src_name)
            thisSrc.spectrum().getFreeParamNames(thisNamesV)
            thisNames = map(lambda x: "%s_%s" % (src_name, x), thisNamesV)
            free_param_names.extend(thisNames)
        pass

        nuisance_parameters = collections.OrderedDict()

        for name in free_param_names:

            value = self.get_nuisance_parameter_value(name)
            bounds = self.get_nuisance_parameter_bounds(name)
            delta = self.get_nuisance_parameter_delta(name)

            nuisance_parameters["%s_%s" % (self.name, name)] = Parameter(
                "%s_%s" % (self.name, name),
                value,
                min_value=bounds[0],
                max_value=bounds[1],
                delta=delta,
            )

            nuisance_parameters[
                "%s_%s" % (self.name, name)
            ].free = self._fit_nuisance_params

        return nuisance_parameters

    def _get_nuisance_parameter(self, param_name):

        tokens = param_name.split("_")

        pname = tokens[-1]

        src = "_".join(tokens[:-1])

        like_src = self.like.model[src]

        if like_src is None:

            src = "_".join(tokens[1:-1])

            like_src = self.like.model[src]

        assert like_src is not None

        return like_src.funcs["Spectrum"].getParam(pname)

    def set_nuisance_parameter_value(self, paramName, value):

        p = self._get_nuisance_parameter(paramName)

        p.setValue(value)

    def get_nuisance_parameter_value(self, paramName):

        p = self._get_nuisance_parameter(paramName)

        return p.getValue()

    def get_nuisance_parameter_bounds(self, paramName):

        p = self._get_nuisance_parameter(paramName)

        return list(p.getBounds())

    def get_nuisance_parameter_delta(self, paramName):

        p = self._get_nuisance_parameter(paramName)

        value = p.getValue()

        return value / 100.0
