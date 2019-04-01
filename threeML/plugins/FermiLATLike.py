import numpy
import collections
import os

import matplotlib.pyplot as plt
from matplotlib import gridspec

from astromodels import Parameter
from astromodels.core.model_parser import ModelParser

from threeML.plugin_prototype import PluginPrototype
from threeML.io.file_utils import get_random_unique_name
from threeML.plugins.gammaln import logfactorial
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

    def __init__(self, likelihood_model, irfs):

        self.likelihood_model = likelihood_model

        self.irfs = irfs

    def set_file_spectrum_energies(self, emin_kev, emax_kev, nEnergies):

        self.energies_kev = numpy.logspace(numpy.log10(emin_kev), numpy.log10(emax_kev), nEnergies)

    def write_xml(self, xmlfile, ra, dec, roi):

        # Loop through all the sources in the likelihood model and generate a FileSpectrum
        # for all of them. This is necessary to allow the FermiLATLike class
        # to update the spectrum in pyLikelihood without having to write and read a .xml file
        # on the disk

        all_sources_for_pylike = []
        temp_files = []

        n_pt_src = self.likelihood_model.get_number_of_point_sources()

        for ip in range(nPtsrc):

            this_src = self._make_file_spectrum(ip)

            all_sources_for_pylike.append(this_src)
            temp_files.append(this_src.temp_file)

        # Now the same for extended sources

        n_ext_src = self.likelihood_model.get_number_of_extended_sources()

        if (n_ext_src > 0):
            raise NotImplemented("Cannot support extended sources yet!")

        iso = LikelihoodComponent.IsotropicTemplate(self.irfs)

        iso.source.spectrum.Normalization.max = 3.5    # changed by JMB... is it right?
        iso.source.spectrum.Normalization.min = 0.5
        iso.source.spectrum.setAttributes()

        all_sources_for_pylike.append(iso)

        # Get a temporary filename which is guaranteed to be unique
        self._unique_filename = get_random_unique_name()

        gal = LikelihoodComponent.GalaxyAndExtragalacticDiffuse(
            self.irfs, ra, dec, 2.5 * roi, cutout_name=self._unique_filename)
        gal.source.spectrum.Value.max = 3.5    # changed by jmb is it correct?
        gal.source.spectrum.Value.min = 0.5
        gal.source.spectrum.setAttributes()

        all_sources_for_pylike.append(gal)

        # Now generate the xml file with also the Galactic and Isotropic diffuse
        # templates
        xml = LikelihoodComponent.LikelihoodModel()
        xml.addSources(*all_sources_for_pylike)
        xml.writeXML(xmlfile)

        return temp_files

    def _make_file_spectrum(self, ip):

        name = self.likelihood_model.get_point_source_name(ip)

        values = self.likelihood_model.get_point_source_fluxes(ip, self.energies_kev)

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

        src = '\n'.join((('<source name= "%s" ' % name) + 'type="PointSource">', '   <spectrum type="PowerLaw2"/>',
                         '   <!-- point source units are cm^-2 s^-1 MeV^-1 -->',
                         '   <spatialModel type="SkyDirFunction"/>', '</source>\n'))
        src = FuncFactory.minidom.parseString(src).getElementsByTagName('source')[0]
        src = FuncFactory.Source(src)

        src.spectrum = FuncFactory.FileFunction()
        src.spectrum.file = temp_name
        src.spectrum.parameters['Normalization'].value = 1.0
        src.spectrum.parameters['Normalization'].max = 1.1
        src.spectrum.parameters['Normalization'].min = 0.9
        src.spectrum.parameters['Normalization'].free = False
        src.spectrum.setAttributes()
        src.deleteChildElements('spectrum')
        src.node.appendChild(src.spectrum.node)

        src.spatialModel = FuncFactory.SkyDirFunction()
        src.deleteChildElements('spatialModel')
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

        instance = FermiLATLike(name, event_file, ft2_file, livetime_cube_file, kind, exposure_map_file)

        instance.set_inner_minimization(inner_minimization)

        instance.set_model(likelihood_model)

        return instance


class FermiLATLike(PluginPrototype):

    def __init__(self,
                 name,
                 event_file,
                 ft2_file,
                 livetime_cube_file,
                 kind,
                 exposure_map_file=None,
                 source_maps=None,
                 binned_expo_map=None):
        """FIXME! briefly describe function

        :param name: 
        :param event_file: 
        :param ft2_file: 
        :param livetime_cubefile: 
        :param kind: 
        :param exposure_map_file: 
        :param source_maps: 
        :param binned_expo_map: 
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

        # This is the limit on the effective area correction factor,
        # which is a multiplicative factor in front of the whole model
        # to account for inter-calibration issues. By default it can vary
        # by 10%. This can be changed by issuing:
        # FermiLATUnbinnedLikeInstance.effCorrLimit = [new limit]
        # where for example a [new limit] of 0.2 allow for an effective
        # area correction up to +/- 20 %

        self.eff_corr_limit = 0.1

        if (kind.upper() != "UNBINNED" and kind.upper() != "BINNED"):

            raise RuntimeError("Accepted values for the kind parameter are: " + "binned, unbinned. You specified: %s" %
                               (kind))

        else:

            self.kind = kind.upper()

        if (kind.upper() == 'UNBINNED'):

            assert exposure_map_file is not None, "You have to provide an exposure map"

            self._event_file = event_file
            self._exposure_map_file = exposure_map_file
            # Read the files and generate the pyLikelihood object
            self.obs = UnbinnedAnalysis.UnbinnedObs(
                self._event_file,
                self._ft2_file,
                expMap=self._exposure_map_file,
                expCube=self._livetime_cube_file,
                irfs=self.irf)

        elif (kind.upper() == "BINNED"):

            assert source_maps is not None, "You have to provide a source map"
            assert binned_expo_map is not None, "You have to provided a (binned) exposure map"

            self._source_maps = source_maps
            self._binned_expo_map = binned_expo_map

            self.obs = BinnedAnalysis.BinnedObs(
                srcMaps=self._source_maps,
                expCube=self._livetime_cube_file,
                binnedExpMap=self._binned_expo_map,
                irfs=self.irf)
        pass

        # Activate inner minimization by default
        self.set_inner_minimization(True)

    pass

    def set_model(self, likelihood_model):
        '''
        Set the model to be used in the joint minimization.
        Must be a LikelihoodModel instance.
        '''

        with suppress_stdout():

            self.lmc = LikelihoodModelConverter(likelihood_model, self.irf)

            self.lmc.set_file_spectrum_energies(self.emin, self.emax, self.n_energies)

            xmlFile = '%s.xml' % get_random_unique_name()

            temp_files = self.lmc.write_xml(xmlFile, self.ra, self.dec, self.rad)

        if (self.kind == "BINNED"):
            self._like = BinnedAnalysis.BinnedAnalysis(self.obs, xmlFile, optimizer='DRMNFB')

        else:

            self._like = UnbinnedAnalysis.UnbinnedAnalysis(self.obs, xmlFile, optimizer='DRMNFB')

        self.likelihood_model = likelihood_model

        # Here we need also to compute the logLike value, so that the model
        # in the XML file will be changed if needed
        dumb = self.get_log_like()

        # Since now the Galactic template is in RAM, we can remove the temporary file
        os.remove(self.lmc._unique_filename)
        os.remove(xmlFile)

        # Delete temporary spectral files
        for temp_file in temp_files:

            os.remove(temp_file)

        # Build the list of the nuisance parameters
        new_nuisance_parameters = self._set_nuisance_parameters()

        self.update_nuisance_parameters(new_nuisance_parameters)

    def get_name(self):
        '''
        Return a name for this dataset (likely set during the constructor)
        '''
        return self.name

    def set_inner_minimization(self, s):

        self.fit_nuisance_params = bool(s)

        for parameter in self.nuisance_parameters:

            self.nuisance_parameters[parameter].free = self.fit_nuisance_params

    def inner_fit(self):
        '''
        This is used for the profile likelihood. Keeping fixed all parameters in the
        modelManager, this method minimize the logLike over the remaining nuisance
        parameters, i.e., the parameters belonging only to the model for this
        particular detector
        '''

        return self.get_log_like()

    def _update_gtlike_model(self):
        '''
        #Slow! But no other options at the moment
        self.like.writeXml(self.xmlModel)
        self.like.logLike.reReadXml(self.xmlModel)
        '''

        energies = self.lmc.energies_kev

        for id, src_name in enumerate(self.likelihood_model.point_sources.keys()):

            values = self.likelihood_model.get_point_source_fluxes(id, energies, tag=self._tag)

            gtlike_src_model = self._like[src_name]

            my_function = gtlike_src_model.getSrcFuncs()['Spectrum']
            my_file_function = pyLike.FileFunction_cast(my_function)

            my_file_function.setParam("Normalization", 1)

            # Cap the values to avoid numerical errors

            capped_values = numpy.minimum(numpy.maximum(values * 1000, 1e-25), 1e5)

            my_file_function.setSpectrum(energies / 1000.0, capped_values)
            gtlike_src_model.setSpectrum(my_file_function)

            # TODO: extended sources

        self._like.syncSrcParams()

    def get_log_like(self):
        '''
        Return the value of the log-likelihood with the current values for the
        parameters stored in the ModelManager instance
        '''

        self._update_gtlike_model()

        if self.fit_nuisance_params:

            for parameter in self.nuisance_parameters:
                self.set_nuisance_parameter_value(parameter, self.nuisance_parameters[parameter].value)

            self._like.syncSrcParams()

        log_like = self._like.logLike.value()

        return log_like - logfactorial(self._like.total_nobs())

    #
    def __reduce__(self):

        return FermiLATUnpickler(), (self.name, self._event_file, self._ft2_file, self._livetime_cube_file, self.kind,
                                     self._exposure_map_file, self.likelihood_model, self.fit_nuisance_params)

    # def __setstate__(self, likelihood_model):
    #
    #     import pdb;pdb.set_trace()
    #
    #     self.set_model(likelihood_model)

    def get_model_and_data(self):

        fake = numpy.array([])

        return fake, fake, fake, fake

    pass

    def display(self):

        e1 = self._like.energies[:-1]
        e2 = self._like.energies[1:]

        ec = (e1 + e2) / 2.0
        de = (e2 - e1) / 2.0

        sum_model = numpy.zeros_like(self._like._srcCnts(self._like.sourceNames()[0]))

        fig, (data_axis, residual_axis) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [2, 1]})

      

        for source_name in self._like.sourceNames():
            sum_model = sum_model + self._like._srcCnts(source_name)

            data_axis.plot(ec, self._like._srcCnts(source_name), label=source_name)

        data_axis.plot(ec, sum_model, label='Total Model')

        data_axis.errorbar(ec, self._like.nobs, xerr=de, yerr=numpy.sqrt(self._like.nobs), fmt='.', label='Counts')

        data_axis.legend(bbox_to_anchor=(1.05, 1), loc=2, numpoints=1)

        # Residuals

        # Using model variance to account for low statistic

        resid = (self._like.nobs - sum_model) / sum_model

        residual_axis.axhline(0, linestyle='--')
        residual_axis.errorbar(ec, resid, xerr=de, yerr=numpy.sqrt(self._like.nobs) / sum_model, capsize=0, fmt='.')

        data_axis.set_xscale("log")
        data_axis.set_yscale("log", nonposy='clip')

        data_axis.set_ylabel("Counts per bin")

        residual_axis.set_xscale("log")

        residual_axis.set_xlabel("Energy (MeV)")
        residual_axis.set_ylabel("(data - mo.) / mo.")

        data_axis.set_xticks([])

        fig.tight_layout()

        # Now remove the space between the two subplots
        # NOTE: this must be placed *after* tight_layout, otherwise it will be ineffective

        fig.subplots_adjust(hspace=0)

        
        return fig

    def _set_nuisance_parameters(self):

        # Get the list of the sources
        sources = list(self._like.model.srcNames)

        free_param_names = []
        for src_name in sources:
            thisNamesV = pyLike.StringVector()
            thisSrc = self._like.logLike.getSource(src_name)
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
                "%s_%s" % (self.name, name), value, min_value=bounds[0], max_value=bounds[1], delta=delta)

            nuisance_parameters["%s_%s" % (self.name, name)].free = self.fit_nuisance_params

        return nuisance_parameters

    def _get_nuisance_parameter(self, param_name):

        tokens = param_name.split("_")

        pname = tokens[-1]

        src = "_".join(tokens[:-1])

        like_src = self._like.model[src]

        if like_src is None:

            src = "_".join(tokens[1:-1])

            like_src = self._like.model[src]

        assert like_src is not None

        return like_src.funcs['Spectrum'].getParam(pname)

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
