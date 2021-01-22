from __future__ import division
from builtins import zip
from builtins import range
from past.utils import old_div
from builtins import object
import numpy
import collections
import os

import matplotlib.pyplot as plt
from matplotlib import gridspec

from astromodels import Parameter
from astromodels.core.model_parser import ModelParser

from threeML.plugin_prototype import PluginPrototype
from threeML.io.file_utils import get_random_unique_name
from threeML.utils.statistics.gammaln import logfactorial
from threeML.io.suppress_stdout import suppress_stdout

import UnbinnedAnalysis
import BinnedAnalysis
import pyLikelihood as pyLike
from GtBurst import LikelihoodComponent
from GtBurst import FuncFactory

__instrument_name = "Fermi LAT (standard classes)"


class MyPointSource(LikelihoodComponent.GenericSource):
    def __init__(self, source, name, temp_file):
        self.source = source
        self.source.name = name
        self.temp_file = temp_file

        super(MyPointSource, self).__init__()


class LikelihoodModelConverter(object):
    def __init__(self, likelihoodModel, irfs, source_name = None):
        """
        :param likelihoodModel: likelihood model
        :param source_name: name of source (must be contained in likelihood model)
        :return: none
        """

        self.likelihoodModel = likelihoodModel

        self.irfs = irfs

        self._source_name = source_name

    def setFileSpectrumEnergies(self, emin_kev, emax_kev, nEnergies):

        self.energiesKeV = numpy.logspace(
            numpy.log10(emin_kev), numpy.log10(emax_kev), nEnergies
        )

    def writeXml(self, xmlfile, ra, dec, roi):

        # Loop through all the sources in the likelihood model and generate a FileSpectrum
        # for all of them. This is necessary to allow the FermiLATLike class
        # to update the spectrum in pyLikelihood without having to write and read a .xml file
        # on the disk

        allSourcesForPyLike = []
        temp_files = []
        
        #Flag for whether we are attributing the dataset to a single source in the model
        sourceName = self._source_name
       
        if sourceName is None:

            nPtsrc = self.likelihoodModel.get_number_of_point_sources()

            for ip in range(nPtsrc):

                this_src = self._makeFileSpectrum(ip)

                allSourcesForPyLike.append(this_src)
                temp_files.append(this_src.temp_file)

        else:
            #We pass from the model just one source
             
            print('Setting single point source %s ... '%sourceName)

            index = self.likelihoodModel.point_sources.keys().index(sourceName)
            this_src = self._makeFileSpectrum(index)
            allSourcesForPyLike.append(this_src)

            temp_files.append(this_src.temp_file)


        # Now the same for extended sources
        nExtSrc = self.likelihoodModel.get_number_of_extended_sources()
        if nExtSrc > 0:
            raise NotImplemented("Cannot support extended sources yet!")

        iso = LikelihoodComponent.IsotropicTemplate(self.irfs)

        iso.source.spectrum.Normalization.max = 1.5
        iso.source.spectrum.Normalization.min = 0.5
        iso.source.spectrum.setAttributes()

        allSourcesForPyLike.append(iso)

        # Get a temporary filename which is guaranteed to be unique
        self._unique_filename = get_random_unique_name()

        gal = LikelihoodComponent.GalaxyAndExtragalacticDiffuse(
            self.irfs, ra, dec, 2.5 * roi, cutout_name=self._unique_filename
        )
        gal.source.spectrum.Value.max = 1.5
        gal.source.spectrum.Value.min = 0.5
        gal.source.spectrum.setAttributes()

        allSourcesForPyLike.append(gal)

        # Now generate the xml file with also the Galactic and Isotropic diffuse
        # templates
        xml = LikelihoodComponent.LikelihoodModel()
        xml.addSources(*allSourcesForPyLike)
        xml.writeXML(xmlfile)

        return temp_files

    def _makeFileSpectrum(self, ip):

        name = self.likelihoodModel.get_point_source_name(ip)

        values = self.likelihoodModel.get_point_source_fluxes(ip, self.energiesKeV)

        tempName = "__%s_%s.txt" % (name, get_random_unique_name())

        with open(tempName, "w+") as f:
            for e, v in zip(self.energiesKeV, values):
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
        src = FuncFactory.minidom.parseString(src).getElementsByTagName("source")[0]
        src = FuncFactory.Source(src)

        src.spectrum = FuncFactory.FileFunction()
        src.spectrum.file = tempName
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

        ra, dec = self.likelihoodModel.get_point_source_position(ip)

        src.spatialModel.RA.value = ra
        src.spatialModel.DEC.value = dec
        src.spatialModel.setAttributes()
        src.setAttributes()

        return MyPointSource(src, name, tempName)


class FermiLATUnpickler(object):
    def __call__(
        self,
        name,
        eventFile,
        ft2File,
        livetimeCube,
        kind,
        exposureMap,
        likelihoodModel,
        innerMinimization,
    ):

        instance = FermiLATLike(
            name, eventFile, ft2File, livetimeCube, kind, exposureMap
        )

        instance.setInnerMinimization(innerMinimization)

        instance.set_model(likelihoodModel)

        return instance


class FermiLATLike(PluginPrototype):
    def __init__(
        self,
        name,
        eventFile,
        ft2File,
        livetimeCube,
        kind,
        exposureMap=None,
        sourceMaps=None,
        binnedExpoMap=None,
        source_name = None,
    ):

        # Initially the nuisance parameters dict is empty, as we don't know yet
        # the likelihood model. They will be updated in set_model

        super(FermiLATLike, self).__init__(name, {})

        # Read the ROI cut
        cc = pyLike.RoiCuts()
        cc.readCuts(eventFile, "EVENTS")
        self.ra, self.dec, self.rad = cc.roiCone()

        # Read the IRF selection
        c = pyLike.Cuts(eventFile, "EVENTS")
        self.irf = c.CALDB_implied_irfs()

        self.ft2File = ft2File
        self.livetimeCube = livetimeCube

        # These are the boundaries and the number of energies for the computation
        # of the model (in keV)
        self.emin = 1e4
        self.emax = 5e8
        self.Nenergies = 200

        # This is the limit on the effective area correction factor,
        # which is a multiplicative factor in front of the whole model
        # to account for inter-calibration issues. By default it can vary
        # by 10%. This can be changed by issuing:
        # FermiLATUnbinnedLikeInstance.effCorrLimit = [new limit]
        # where for example a [new limit] of 0.2 allow for an effective
        # area correction up to +/- 20 %

        self.effCorrLimit = 0.1

        if kind.upper() != "UNBINNED" and kind.upper() != "BINNED":

            raise RuntimeError(
                "Accepted values for the kind parameter are: "
                + "binned, unbinned. You specified: %s" % (kind)
            )

        else:

            self.kind = kind.upper()

        if kind.upper() == "UNBINNED":

            assert exposureMap is not None, "You have to provide an exposure map"

            self.eventFile = eventFile
            self.exposureMap = exposureMap
            # Read the files and generate the pyLikelihood object
            self.obs = UnbinnedAnalysis.UnbinnedObs(
                self.eventFile,
                self.ft2File,
                expMap=self.exposureMap,
                expCube=self.livetimeCube,
                irfs=self.irf,
            )

        elif kind.upper() == "BINNED":

            assert sourceMaps is not None, "You have to provide a source map"
            assert (
                binnedExpoMap is not None
            ), "You have to provided a (binned) exposure map"

            self.sourceMaps = sourceMaps
            self.binnedExpoMap = binnedExpoMap

            self.obs = BinnedAnalysis.BinnedObs(
                srcMaps=self.sourceMaps,
                expCube=self.livetimeCube,
                binnedExpMap=self.binnedExpoMap,
                irfs=self.irf,
            )
        pass

        # Activate inner minimization by default
        self.setInnerMinimization(True)

        self._source_name = source_name

    pass

    def set_model(self, likelihoodModel, source_name = None):
        """
        Set the model to be used in the joint minimization.
        Must be a LikelihoodModel instance.

        This method can also set or override a previously set source name.
        """

        #with suppress_stdout():

        
        if self._source_name is not None:
            if (source_name is not None) and (source_name != self._source_name):
                print('Warning! Changing target source from %s to %s'%(self._source_name, source_name))
                self._source_name = source_name

            assert self._source_name in likelihoodModel.point_sources, ('Source %s is not a source in the likelihood model! '%self._source_name)

        self.lmc = LikelihoodModelConverter(likelihoodModel, self.irf, source_name=self._source_name)
        self.lmc.setFileSpectrumEnergies(self.emin, self.emax, self.Nenergies)

        xmlFile = str("%s.xml" % get_random_unique_name())

        temp_files = self.lmc.writeXml(xmlFile, self.ra, self.dec, self.rad)

        if self.kind == "BINNED":
            self.like = BinnedAnalysis.BinnedAnalysis(
                self.obs, xmlFile, optimizer="DRMNFB"
            )

        else:
            #import pdb;pdb.set_trace()
            self.like = UnbinnedAnalysis.UnbinnedAnalysis(
                self.obs, xmlFile, optimizer="DRMNFB"
            )

        self.likelihoodModel = likelihoodModel

        # Here we need also to compute the logLike value, so that the model
        # in the XML file will be chanded if needed
        dumb = self.get_log_like()

        # Since now the Galactic template is in RAM, we can remove the temporary file
        os.remove(self.lmc._unique_filename)
        os.remove(xmlFile)

        # Delete temporary spectral files
        for temp_file in temp_files:

            os.remove(temp_file)

        # Build the list of the nuisance parameters
        new_nuisance_parameters = self._setNuisanceParameters()

        self.update_nuisance_parameters(new_nuisance_parameters)

    def clear_source_name(self):
        if self._source_name is not None:
            print('Clearing %s as a source for this plugin.'%self._source_name)
            self._source_name = None
        else:
            print('Source not named. Use set_model to set a source.')

    def get_name(self):
        """
        Return a name for this dataset (likely set during the constructor)
        """
        return self.name

    def setInnerMinimization(self, s):

        self.fit_nuisance_params = bool(s)

        for parameter in self.nuisance_parameters:

            self.nuisance_parameters[parameter].free = self.fit_nuisance_params

    def inner_fit(self):
        """
        This is used for the profile likelihood. Keeping fixed all parameters in the
        modelManager, this method minimize the logLike over the remaining nuisance
        parameters, i.e., the parameters belonging only to the model for this
        particular detector
        """

        return self.get_log_like()

    def _updateGtlikeModel(self):
        """
        #Slow! But no other options at the moment
        self.like.writeXml(self.xmlModel)
        self.like.logLike.reReadXml(self.xmlModel)
        """

        energies = self.lmc.energiesKeV
 
        if self._source_name is None:
            for id, srcName in enumerate(self.likelihoodModel.point_sources.keys()):

                values = self.likelihoodModel.get_point_source_fluxes(
                    id, energies, tag=self._tag
                )
                
                #on the second iteration, self.like doesn't have the second srcName defined so that needs to be carried from flags
                gtlikeSrcModel = self.like[srcName]

                my_function = gtlikeSrcModel.getSrcFuncs()["Spectrum"]
                my_file_function = pyLike.FileFunction_cast(my_function)

                my_file_function.setParam("Normalization", 1)

                # Cap the values to avoid numerical errors

                capped_values = numpy.minimum(numpy.maximum(values * 1000, 1e-25), 1e5)

                my_file_function.setSpectrum(energies / 1000.0, capped_values)
                gtlikeSrcModel.setSpectrum(my_file_function)

                # TODO: extended sources
        else:
            srcName = self._source_name
            id = self.likelihoodModel.point_sources.keys().index(srcName)

            values = self.likelihoodModel.get_point_source_fluxes(
                    id, energies, tag=self._tag
            )

            #on the second iteration, self.like doesn't have the second srcName defined so that needs to be carried from flags
            gtlikeSrcModel = self.like[srcName]

            my_function = gtlikeSrcModel.getSrcFuncs()["Spectrum"]
            my_file_function = pyLike.FileFunction_cast(my_function)

            my_file_function.setParam("Normalization", 1)

            # Cap the values to avoid numerical errors

            capped_values = numpy.minimum(numpy.maximum(values * 1000, 1e-25), 1e5)

            my_file_function.setSpectrum(energies / 1000.0, capped_values)
            gtlikeSrcModel.setSpectrum(my_file_function)


        self.like.syncSrcParams()

    def get_log_like(self):
        """
        Return the value of the log-likelihood with the current values for the
        parameters stored in the ModelManager instance
        """

        self._updateGtlikeModel()

        if self.fit_nuisance_params:

            for parameter in self.nuisance_parameters:
                self.setNuisanceParameterValue(
                    parameter, self.nuisance_parameters[parameter].value
                )

            self.like.syncSrcParams()

        log_like = self.like.logLike.value()

        return log_like - logfactorial(self.like.total_nobs())

    #
    def __reduce__(self):

        return (
            FermiLATUnpickler(),
            (
                self.name,
                self.eventFile,
                self.ft2File,
                self.livetimeCube,
                self.kind,
                self.exposureMap,
                self.likelihoodModel,
                self.fit_nuisance_params,
            ),
        )

    # def __setstate__(self, likelihoodModel):
    #
    #     import pdb;pdb.set_trace()
    #
    #     self.set_model(likelihoodModel)

    def getModelAndData(self):

        fake = numpy.array([])

        return fake, fake, fake, fake

    pass

    def display(self):

        e1 = self.like.energies[:-1]
        e2 = self.like.energies[1:]

        ec = (e1 + e2) / 2.0
        de = (e2 - e1) / 2.0

        sum_model = numpy.zeros_like(self.like._srcCnts(self.like.sourceNames()[0]))

        fig = plt.figure()

        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
        gs.update(hspace=0)

        sub = plt.subplot(gs[0])

        for sourceName in self.like.sourceNames():
            sum_model = sum_model + self.like._srcCnts(sourceName)

            sub.plot(ec, self.like._srcCnts(sourceName), label=sourceName)

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

        sub1.axhline(0, linestyle="--")
        sub1.errorbar(
            ec,
            resid,
            xerr=de,
            yerr=old_div(numpy.sqrt(self.like.nobs), sum_model),
            capsize=0,
            fmt=".",
        )

        sub.set_xscale("log")
        sub.set_yscale("log", nonposy="clip")

        sub.set_ylabel("Counts per bin")

        sub1.set_xscale("log")

        sub1.set_xlabel("Energy (MeV)")
        sub1.set_ylabel("(data - mo.) / mo.")

        sub.set_xticks([])

        return fig

    def _setNuisanceParameters(self):

        # Get the list of the sources
        sources = list(self.like.model.srcNames)

        freeParamNames = []
        for srcName in sources:
            thisNamesV = pyLike.StringVector()
            thisSrc = self.like.logLike.getSource(srcName)
            thisSrc.spectrum().getFreeParamNames(thisNamesV)
            thisNames = ["%s_%s" % (srcName, x) for x in thisNamesV]
            freeParamNames.extend(thisNames)
        pass

        nuisanceParameters = collections.OrderedDict()

        for name in freeParamNames:

            value = self.getNuisanceParameterValue(name)
            bounds = self.getNuisanceParameterBounds(name)
            delta = self.getNuisanceParameterDelta(name)

            nuisanceParameters["%s_%s" % (self.name, name)] = Parameter(
                "%s_%s" % (self.name, name),
                value,
                min_value=bounds[0],
                max_value=bounds[1],
                delta=delta,
            )

            nuisanceParameters[
                "%s_%s" % (self.name, name)
            ].free = self.fit_nuisance_params

        return nuisanceParameters

    def _get_nuisance_param(self, paramName):

        tokens = paramName.split("_")

        pname = tokens[-1]

        src = "_".join(tokens[:-1])

        like_src = self.like.model[src]

        if like_src is None:

            src = "_".join(tokens[1:-1])

            like_src = self.like.model[src]

        assert like_src is not None

        return like_src.funcs["Spectrum"].getParam(pname)

    def setNuisanceParameterValue(self, paramName, value):

        p = self._get_nuisance_param(paramName)

        p.setValue(value)

    def getNuisanceParameterValue(self, paramName):

        p = self._get_nuisance_param(paramName)

        return p.getValue()

    def getNuisanceParameterBounds(self, paramName):

        p = self._get_nuisance_param(paramName)

        return list(p.getBounds())

    def getNuisanceParameterDelta(self, paramName):

        p = self._get_nuisance_param(paramName)

        value = p.getValue()

        return value / 100.0

