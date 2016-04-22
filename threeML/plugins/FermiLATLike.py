import UnbinnedAnalysis
import BinnedAnalysis
import pyLikelihood as pyLike
import numpy
from GtBurst import LikelihoodComponent
from GtBurst import FuncFactory
import matplotlib.pyplot as plt
from matplotlib import gridspec

from threeML.plugin_prototype import PluginPrototype
from astromodels import Parameter
from threeML.plugins.gammaln import logfactorial

__instrument_name = "Fermi LAT (standard classes)"


class myPointSource(LikelihoodComponent.GenericSource):
    def __init__(self, source, name):
        self.source = source
        self.source.name = name


class LikelihoodModelConverter(object):
    def __init__(self, likelihoodModel, irfs):

        self.likelihoodModel = likelihoodModel

        self.irfs = irfs

    def setFileSpectrumEnergies(self, emin_kev, emax_kev, nEnergies):

        self.energiesKeV = numpy.logspace(numpy.log10(emin_kev),
                                          numpy.log10(emax_kev),
                                          nEnergies)

    def writeXml(self, xmlfile, ra, dec, roi):

        # Loop through all the sources in the likelihood model and generate a FileSpectrum
        # for all of them. This is necessary to allow the FermiLATLike class
        # to update the spectrum in pyLikelihood without having to write and read a .xml file
        # on the disk

        allSourcesForPyLike = []

        nPtsrc = self.likelihoodModel.get_number_of_point_sources()

        for ip in range(nPtsrc):
            allSourcesForPyLike.append(self._makeFileSpectrum(ip))

        # Now the same for extended sources

        nExtSrc = self.likelihoodModel.get_number_of_extended_sources()

        if (nExtSrc > 0):
            raise NotImplemented("Cannot support extended sources yet!")

        iso = LikelihoodComponent.IsotropicTemplate(self.irfs)

        iso.source.spectrum.Normalization.max = 1.5
        iso.source.spectrum.Normalization.min = 0.5
        iso.source.spectrum.setAttributes()

        allSourcesForPyLike.append(iso)

        gal = LikelihoodComponent.GalaxyAndExtragalacticDiffuse(
            self.irfs, ra, dec, 2.5 * roi)
        gal.source.spectrum.Value.max = 1.5
        gal.source.spectrum.Value.min = 0.5
        gal.source.spectrum.setAttributes()

        allSourcesForPyLike.append(gal)

        # Now generate the xml file with also the Galactic and Isotropic diffuse
        # templates
        xml = LikelihoodComponent.LikelihoodModel()
        xml.addSources(*allSourcesForPyLike)
        xml.writeXML(xmlfile)

    def _makeFileSpectrum(self, ip):

        name = self.likelihoodModel.get_point_source_name(ip)

        values = self.likelihoodModel.get_point_source_fluxes(ip,
                                                              self.energiesKeV)

        tempName = "__%s.txt" % name

        with open(tempName, "w+") as f:
            for e, v in zip(self.energiesKeV, values):
                # Gtlike needs energies in MeV and fluxes in ph/MeV/cm2)

                f.write("%s %s\n" % (e / 1000.0, v * 1000.0))

        # p                         = fileFunction.parameter("Normalization")
        # p.setBounds(1-float(effAreaAllowedSize),1+effAreaAllowedSize)

        # Now generate the XML source wrapper
        # This is convoluted, but that's the ST way of doing things...
        # The final product is a class with a writeXml method

        src = '\n'.join((('<source name= "%s" ' % name) + 'type="PointSource">',
                         '   <spectrum type="PowerLaw2"/>',
                         '   <!-- point source units are cm^-2 s^-1 MeV^-1 -->',
                         '   <spatialModel type="SkyDirFunction"/>',
                         '</source>\n'))
        src = FuncFactory.minidom.parseString(src).getElementsByTagName('source')[0]
        src = FuncFactory.Source(src)

        src.spectrum = FuncFactory.FileFunction()
        src.spectrum.file = tempName
        src.spectrum.parameters['Normalization'].value = 1.0
        src.spectrum.parameters['Normalization'].max = 1.1
        src.spectrum.parameters['Normalization'].min = 0.9
        src.spectrum.setAttributes()
        src.deleteChildElements('spectrum')
        src.node.appendChild(src.spectrum.node)

        src.spatialModel = FuncFactory.SkyDirFunction()
        src.deleteChildElements('spatialModel')
        src.node.appendChild(src.spatialModel.node)

        ra, dec = self.likelihoodModel.get_point_source_position(ip)

        src.spatialModel.RA.value = ra
        src.spatialModel.DEC.value = dec
        src.spatialModel.setAttributes()
        src.setAttributes()

        return myPointSource(src, name)


class FermiLATLike(PluginPrototype):
    def __init__(self, name, eventFile, ft2File, livetimeCube, kind, *args):

        self.name = name

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
        self.Nenergies = 1200

        # This is the limit on the effective area correction factor,
        # which is a multiplicative factor in front of the whole model
        # to account for inter-calibration issues. By default it can vary
        # by 10%. This can be changed by issuing:
        # FermiLATUnbinnedLikeInstance.effCorrLimit = [new limit]
        # where for example a [new limit] of 0.2 allow for an effective
        # area correction up to +/- 20 %

        self.effCorrLimit = 0.1

        if (kind.upper() != "UNBINNED" and kind.upper() != "BINNED"):

            raise RuntimeError("Accepted values for the kind parameter are: " +
                               "binned, unbinned. You specified: %s" % (kind))

        else:

            self.kind = kind.upper()

        if (kind.upper() == 'UNBINNED'):
            exposureMap = args[0]
            self.eventFile = eventFile
            self.exposureMap = exposureMap
            # Read the files and generate the pyLikelihood object
            self.obs = UnbinnedAnalysis.UnbinnedObs(self.eventFile,
                                                    self.ft2File,
                                                    expMap=self.exposureMap,
                                                    expCube=self.livetimeCube,
                                                    irfs=self.irf)

        elif (kind.upper() == "BINNED"):
            sourceMaps, binnedExpoMap = args
            self.sourceMaps = sourceMaps
            self.binnedExpoMap = binnedExpoMap

            self.obs = BinnedAnalysis.BinnedObs(srcMaps=self.sourceMaps,
                                                expCube=self.livetimeCube,
                                                binnedExpMap=self.binnedExpoMap,
                                                irfs=self.irf)
        pass

        # Activate inner minimization by default
        self.setInnerMinimization(True)

    pass

    def set_model(self, likelihoodModel):
        '''
        Set the model to be used in the joint minimization.
        Must be a LikelihoodModel instance.
        '''
        self.lmc = LikelihoodModelConverter(likelihoodModel,
                                            self.irf)

        self.lmc.setFileSpectrumEnergies(self.emin, self.emax, self.Nenergies)

        xmlFile = '__xmlFile.xml'

        self.lmc.writeXml(xmlFile, self.ra, self.dec, self.rad)

        if (self.kind == "BINNED"):
            self.like = BinnedAnalysis.BinnedAnalysis(self.obs,
                                                      xmlFile,
                                                      optimizer='DRMNFB')

        else:

            self.like = UnbinnedAnalysis.UnbinnedAnalysis(self.obs,
                                                          xmlFile,
                                                          optimizer='DRMNFB')

        self.likelihoodModel = likelihoodModel

        # Here we need also to compute the logLike value, so that the model
        # in the XML file will be chanded if needed
        dumb = self.get_log_like()

        # Build the list of the nuisance parameters
        self._setNuisanceParameters()

    pass

    def get_name(self):
        '''
        Return a name for this dataset (likely set during the constructor)
        '''
        return self.name

    pass

    def setInnerMinimization(self, s):

        self.innerMinimization = bool(s)

    def inner_fit(self):
        '''
        This is used for the profile likelihood. Keeping fixed all parameters in the
        modelManager, this method minimize the logLike over the remaining nuisance
        parameters, i.e., the parameters belonging only to the model for this
        particular detector
        '''
        self._updateGtlikeModel()

        if not self.innerMinimization:

            log_like = self.like.logLike.value()

        else:

            try:
                # Use .optimize instead of .fit because we don't need the errors
                # (.optimize is faster than .fit)
                self.like.optimize(0)
            except:
                # This is necessary because sometimes fitting algorithms go and explore extreme region of the
                # parameter space, which might turn out to give strange model shapes and therefore
                # problems in the likelihood fit
                print("Warning: failed likelihood fit (probably parameters are too extreme).")
                return 1e5
            else:
                # Update the value for the nuisance parameters
                for par in self.nuisanceParameters.values():
                    newValue = self.getNuisanceParameterValue(par.name)
                    par.value = newValue
                pass

                log_like = self.like.logLike.value()

        return log_like - logfactorial(self.like.total_nobs())

    pass

    # def _initFileFunction(self):
    # If the point source is already in the model, delete it
    # (like.deleteSource() returns the model)
    # gtlikeSrcModel            = self.like.deleteSource(self.modelManager.name)

    # Get the new model for the source (with the latest parameter values)
    # and add it back to the likelihood model
    #  self._getNewGtlikeModel(gtlikeSrcModel,self.effCorrLimit)
    #  self.like.addSource(gtlikeSrcModel)

    # Slow! But no other options at the moment
    #  self.like.writeXml(self.xmlModel)
    #  self.like.logLike.reReadXml(self.xmlModel)
    # pass

    # def _getNewGtlikeModel(self,currentGtlikeModel,effAreaAllowedSize=0.1):
    # Write on disk the current model
    #  tempName                  = os.path.join(
    #                              os.path.dirname(
    #                              os.path.abspath(self.xmlModel)
    #                                             ),
    #                             '__fileSpectrum.txt')
    #
    #  #This will recompute the model if necessary
    #  self.modelManager.writeToFile(tempName,self.emin,self.emax,self.Nenergies)
    #
    #  #Generate a new FileFunction spectrum and assign it to the source
    #  fileFunction              = pyLike.FileFunction()
    #  fileFunction.readFunction(tempName)
    #  fileFunction.setParam("Normalization",1)
    #  p                         = fileFunction.parameter("Normalization")
    #  p.setBounds(1-float(effAreaAllowedSize),1+effAreaAllowedSize)
    #
    #  currentGtlikeModel.setSpectrum(fileFunction)
    # pass

    def _updateGtlikeModel(self):
        '''
        #Slow! But no other options at the moment
        self.like.writeXml(self.xmlModel)
        self.like.logLike.reReadXml(self.xmlModel)
        '''

        energies = self.lmc.energiesKeV

        for id, srcName in enumerate(self.likelihoodModel.point_sources.keys()):

            values = self.likelihoodModel.get_point_source_fluxes(id, energies)

            gtlikeSrcModel = self.like[srcName]

            my_function = gtlikeSrcModel.getSrcFuncs()['Spectrum']
            my_file_function = pyLike.FileFunction_cast(my_function)

            my_file_function.setParam("Normalization", 1)

            # Cap the values to avoid numerical errors

            capped_values = numpy.minimum(numpy.maximum(values * 1000, 1e-25), 1e5)

            my_file_function.setSpectrum(energies / 1000.0, capped_values)
            gtlikeSrcModel.setSpectrum(my_file_function)

            # TODO: extended sources

        self.like.syncSrcParams()

    def get_log_like(self):
        '''
        Return the value of the log-likelihood with the current values for the
        parameters stored in the ModelManager instance
        '''
        self._updateGtlikeModel()
        try:

            value = self.like.logLike.value()

        except:

            raise

        return value - logfactorial(self.like.total_nobs())

    pass

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

        sub.plot(ec, sum_model, label='Total Model')

        sub.errorbar(ec, self.like.nobs, xerr=de,
                     yerr=numpy.sqrt(self.like.nobs),
                     fmt='.', label='Counts')

        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, numpoints=1)

        # Residuals

        sub1 = plt.subplot(gs[1])

        # Using model variance to account for low statistic

        resid = (self.like.nobs - sum_model) / sum_model

        sub1.axhline(0, linestyle='--')
        sub1.errorbar(ec, resid, xerr=de,
                      yerr=numpy.sqrt(self.like.nobs) / sum_model,
                      capsize=0, fmt='.')

        sub.set_xscale("log")
        sub.set_yscale("log", nonposy='clip')

        sub.set_ylabel("Counts per bin")

        sub1.set_xscale("log")

        sub1.set_xlabel("Energy (MeV)")
        sub1.set_ylabel("(Mo. - data) / mo.")

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
            thisNames = map(lambda x: "%s-%s" % (srcName, x), thisNamesV)
            freeParamNames.extend(thisNames)
        pass

        self.nuisanceParameters = {}

        for name in freeParamNames:
            value = self.getNuisanceParameterValue(name)
            bounds = self.getNuisanceParameterBounds(name)
            delta = self.getNuisanceParameterDelta(name)
            self.nuisanceParameters[name] = Parameter(name, value, min_value=bounds[0],
                                                      max_value=bounds[1], delta=delta)
        pass

    pass

    def get_nuisance_parameters(self):
        '''
        Return a list of nuisance parameters. Return an empty list if there
        are no nuisance parameters
        '''
        return self.nuisanceParameters

    def getNuisanceParameterValue(self, paramName):
        src, pname = paramName.split("-")
        return self.like.model[src].funcs['Spectrum'].getParam(pname).getValue()

    pass

    def getNuisanceParameterBounds(self, paramName):
        src, pname = paramName.split("-")
        return list(self.like.model[src].funcs['Spectrum'].getParam(pname).getBounds())

    pass

    def getNuisanceParameterDelta(self, paramName):
        src, pname = paramName.split("-")
        value = self.like.model[src].funcs['Spectrum'].getParam(pname).getValue()
        return value / 100.0

    pass

    def setNuisanceParameterValue(self, paramName, value):
        src, pname = paramName.split("-")
        self.like.model[src].funcs['Spectrum'].getParam(pname).setValue(value)

    pass


pass
