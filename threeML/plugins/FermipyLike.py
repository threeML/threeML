from fermipy.gtanalysis import GTAnalysis
import numpy as np
import os
import yaml
import astromodels

from threeML.plugin_prototype import PluginPrototype
from threeML.exceptions.custom_exceptions import custom_warnings
from threeML.io.file_utils import get_random_unique_name
from threeML.io.file_utils import sanitize_filename
from threeML.plugins.gammaln import logfactorial

# These are part of gtburst, which is part of the Fermi ST
from GtBurst.LikelihoodComponent import findGalacticTemplate, findIsotropicTemplate


__instrument_name = "Fermi LAT (with fermipy)"


#########################################################
# NOTE:
# Fermipy does NOT support Unbinned Likelihood analysis
#########################################################


def _get_fermipy_instance(configuration, likelihood_model):
    """
    Generate a 'model' configuration section for fermipy starting from a likelihood model from astromodels

    :param configuration: a dictionary containing the configuration for fermipy
    :param likelihood_model: the input likelihood model from astromodels
    :type likelihood_model: astromodels.Model
    :return: a dictionary with the 'model' section of the fermipy configuration
    """

    # Generate a new 'model' section in the configuration which reflects the model
    # provided as input

    # Get center and radius of ROI
    ra_center = float(configuration['selection']['ra'])
    dec_center = float(configuration['selection']['dec'])

    roi_width = float(configuration['binning']['roiwidth'])
    roi_radius = roi_width / np.sqrt(2.0)

    # Get IRFS
    irfs = configuration['gtlike']['irfs']

    # The fermipy model is just a dictionary. It corresponds to the 'model' section
    # of the configuration file (http://fermipy.readthedocs.io/en/latest/config.html#model)

    fermipy_model = {}

    # Find Galactic and Isotropic templates appropriate for this IRFS
    # (information on the ROI is used to cut the Galactic template, which speeds up the
    # analysis a lot)
    # NOTE: these are going to be absolute paths

    galactic_template = sanitize_filename(findGalacticTemplate(irfs, ra_center, dec_center, roi_radius), True)
    isotropic_template = sanitize_filename(findIsotropicTemplate(irfs), True)

    # Add them to the fermipy model

    fermipy_model['galdiff'] = galactic_template
    fermipy_model['isodiff'] = isotropic_template

    # Now iterate over all sources contained in the likelihood model
    sources = []

    # point sources
    for point_source in likelihood_model.point_sources.values():  # type: astromodels.PointSource

        this_source = {'Index' : 2.56233, 'Scale' : 572.78, 'Prefactor' : 2.4090e-12}
        this_source['name'] = point_source.name
        this_source['ra'] = point_source.position.ra.value
        this_source['dec'] = point_source.position.dec.value

        # The spectrum used here is unconsequential, as it will be substituted by a FileFunction
        # later on. So I will just use PowerLaw for everything
        this_source['SpectrumType'] = 'PowerLaw'

        sources.append(this_source)

    # extended sources
    for extended_source in likelihood_model.extended_sources.values():  # type: astromodels.ExtendedSource

        raise NotImplementedError("Extended sources are not supported yet")

    # Add all sources to the model
    fermipy_model['sources'] = sources

    # Now we can finally instance the GTAnalysis instance
    configuration['model'] = fermipy_model

    gta = GTAnalysis(configuration)

    # This will take a long time if it's the first time we run with this model
    gta.setup()

    # Substitute all spectra for point sources with FileSpectrum, so that we will be able to control
    # them from 3ML

    energies_keV = None

    for point_source in likelihood_model.point_sources.values():  # type: astromodels.PointSource

        # Fix this source, so fermipy will not optimize by itself the parameters
        gta.free_source(point_source.name, False)

        # This will substitute the current spectrum with a FileFunction with the same shape and flux
        gta.set_source_spectrum(point_source.name, 'FileFunction', update_source=False)

        # Get the energies at which to evaluate this source
        this_log_energies, _flux = gta.get_source_dnde(point_source.name)
        this_energies_keV = 10**this_log_energies * 1e3  # fermipy energies are in GeV, we need keV

        if energies_keV is None:

            energies_keV = this_energies_keV

        else:

            # This is to make sure that all sources are evaluated at the same energies

            assert np.all(energies_keV == this_energies_keV)

        dnde = point_source(energies_keV) # ph / (cm2 s keV)
        dnde_per_MeV = dnde * 1000.0 # ph / (cm2 s MeV)
        gta.set_source_dnde(point_source.name, dnde_per_MeV, False)

    # Same for extended source
    for extended_source in likelihood_model.extended_sources.values():  # type: astromodels.ExtendedSource

        raise NotImplementedError("Extended sources are not supported yet")

    return gta, energies_keV


class FermipyLike(PluginPrototype):
    """
    Plugin for the data of the Fermi Large Area Telescope, based on fermipy (http://fermipy.readthedocs.io/)
    """
    def __init__(self, name, configuration):
        """
        :param name: a name for this instance
        :param configuration: either a path to a YAML configuration file or a dictionary containing the configuration
        (see http://fermipy.readthedocs.io/)
        """

        # There are no nuisance parameters

        nuisance_parameters = {}

        super(FermipyLike, self).__init__(name, nuisance_parameters=nuisance_parameters)

        # Check whether the provided configuration is a file

        if not isinstance(configuration, dict):

            # Assume this is a file name
            configuration_file = sanitize_filename(configuration)

            assert os.path.exists(configuration), "Configuration file %s does not exist" % configuration_file

            # Read the configuration
            with open(configuration_file) as f:

                self._configuration = yaml.safe_load(f)

        else:

            # Configuration is a dictionary. Nothing to do
            self._configuration = configuration

        # If the user provided a 'model' key, issue a warning, as the model will be defined
        # later on and will overwrite the one contained in 'model'

        if 'model' in self._configuration:

            custom_warnings.warn("The provided configuration contains a 'model' section, which is useless as it "
                                 "will be overridden")

            self._configuration.pop('model')

        if 'fileio' in self._configuration:

            custom_warnings.warn("The provided configuration contains a 'fileio' section, which will be "
                                 "overwritten")

            self._configuration.pop('fileio')

        # Prepare the 'fileio' part
        # Save all output in a directory with a unique name, so multiple instances of
        # this plugin will be able to coexist

        self._unique_id = "__%s" % self.name # get_random_unique_name()

        self._configuration['fileio'] = {'outdir': self._unique_id}

        # Ensure that there is a complete definition of a Region Of Interest (ROI)
        assert ('ra' in self._configuration['selection']) \
               and ('dec' in self._configuration['selection']), \
            "You have to provide 'ra' and 'dec' in the 'selection' section of the configuration. Source name " \
            "resolution, as well as Galactic coordinates, are not currently supported"

        # This is empty at the beginning, will be instanced in the set_model method
        self._gta = None

    @property
    def configuration(self):
        """
        Returns the loaded configuration

        :return: a dictionary containing the active configuration
        """
        return self._configuration

    @property
    def gta(self):

        assert self._gta is not None, "You have to perform a fit or a bayesian analysis before accessing the " \
                                      "gta object"

        return self._gta

    def set_model(self, likelihood_model_instance):
        """
        Set the model to be used in the joint minimization. Must be a LikelihoodModel instance.
        """

        # This will take a long time if it's the first time we run, as it will select the data,
        # produce livetime cube, expomap, source maps and so on

        self._likelihood_model = likelihood_model_instance

        self._gta, self._pts_energies = _get_fermipy_instance(self._configuration, likelihood_model_instance)

    def _update_model_in_fermipy(self):

        # Substitute all spectra for point sources with FileSpectrum, so that we will be able to control
        # them from 3ML
        for point_source in self._likelihood_model.point_sources.values():  # type: astromodels.PointSource

            # Now set the spectrum of this source to the right one
            dnde = point_source(self._pts_energies) # ph / (cm2 s keV)
            dnde_MeV = dnde * 1000.0  # ph / (cm2 s MeV)

            # NOTE: I use update_source=False because it makes things 100x faster and I verified that
            # it does not change the result.

            self._gta.set_source_dnde(point_source.name, dnde_MeV, False)

        # Same for extended source
        for extended_source in self._likelihood_model.extended_sources.values():  # type: astromodels.ExtendedSource

            raise NotImplementedError("Extended sources are not supported yet")

    def get_log_like(self):
        '''
        Return the value of the log-likelihood with the current values for the
        parameters stored in the ModelManager instance
        '''

        # Update all sources on the fermipy side
        self._update_model_in_fermipy()

        # Get value of the log likelihood

        try:

            value = self._gta.like.logLike.value()

        except:

            raise

        return value #- logfactorial(self._gta.like.total_nobs())

    def inner_fit(self):
        """
        This is used for the profile likelihood. Keeping fixed all parameters in the
        LikelihoodModel, this method minimize the logLike over the remaining nuisance
        parameters, i.e., the parameters belonging only to the model for this
        particular detector. If there are no nuisance parameters, simply return the
        logLike value.
        """
        return self.get_log_like()