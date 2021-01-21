from __future__ import division
from past.utils import old_div
import astromodels
import numpy as np
import os
import yaml
import astropy.io.fits as fits

from threeML.exceptions.custom_exceptions import custom_warnings
from threeML.io.file_utils import sanitize_filename
from threeML.plugin_prototype import PluginPrototype
from threeML.utils.statistics.gammaln import logfactorial
from threeML.utils.unique_deterministic_tag import get_unique_deterministic_tag
from threeML.utils.power_of_two_utils import is_power_of_2
from threeML.io.package_data import get_path_of_data_file
from threeML.io.dict_with_pretty_print import DictWithPrettyPrint

__instrument_name = "Fermi LAT (with fermipy)"


#########################################################
# NOTE:
# Fermipy does NOT support Unbinned Likelihood analysis
#########################################################


# A lookup map for the correspondence between IRFS and evclass
evclass_irf = {
    2: "P8R2_TRANSIENT100E_V6",
    4: "P8R2_TRANSIENT100_V6",
    8: "P8R2_TRANSIENT020E_V6",
    16: "P8R2_TRANSIENT020_V6",
    32: "P8R2_TRANSIENT010E_V6",
    64: "P8R2_TRANSIENT010_V6",
    128: "P8R2_SOURCE_V6",
    256: "P8R2_CLEAN_V6",
    512: "P8R2_ULTRACLEAN_V6",
    1024: "P8R2_ULTRACLEANVETO_V6",
    32768: "P8R2_TRANSIENT100S_V6",
    65536: "P8R2_TRANSIENT015S_V6",
}


def _get_unique_tag_from_configuration(configuration):
    keys_for_hash = (
        ("data", ("evfile", "scfile")),
        ("binning", ("roiwidth", "binsz", "binsperdec")),
        (
            "selection",
            ("emin", "emax", "zmax", "evclass", "evtype", "filter", "ra", "dec"),
        ),
    )

    string_to_hash = []

    for section, keys in keys_for_hash:

        assert section in configuration, (
            "Configuration lacks section %s, which is required" % section
        )

        for key in keys:

            assert key in configuration[section], (
                "Section %s in configuration lacks key %s, which is required" % key
            )

            string_to_hash.append("%s" % configuration[section][key])

    return get_unique_deterministic_tag(",".join(string_to_hash))


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
    ra_center = float(configuration["selection"]["ra"])
    dec_center = float(configuration["selection"]["dec"])

    roi_width = float(configuration["binning"]["roiwidth"])
    roi_radius = old_div(roi_width, np.sqrt(2.0))

    # Get IRFS
    irfs = evclass_irf[int(configuration["selection"]["evclass"])]

    if "gtlike" in configuration and "irfs" in configuration["gtlike"]:

        assert irfs.upper() == configuration["gtlike"]["irfs"].upper(), (
            "Evclass points to IRFS %s, while you specified %s into he "
            "configuration" % (irfs, configuration["gtlike"]["irfs"])
        )

    else:

        if not "gtlike" in configuration:

            configuration["gtlike"] = {}

        configuration["gtlike"]["irfs"] = irfs

    # The fermipy model is just a dictionary. It corresponds to the 'model' section
    # of the configuration file (http://fermipy.readthedocs.io/en/latest/config.html#model)

    fermipy_model = {}

    # Find Galactic and Isotropic templates appropriate for this IRFS
    # (information on the ROI is used to cut the Galactic template, which speeds up the
    # analysis a lot)
    # NOTE: these are going to be absolute paths

    galactic_template = str( sanitize_filename(
        findGalacticTemplate(irfs, ra_center, dec_center, roi_radius), True  # noqa: F821
    ) )
    isotropic_template = str( sanitize_filename(findIsotropicTemplate(irfs), True) ) # noqa: F821

    # Add them to the fermipy model

    fermipy_model["galdiff"] = galactic_template
    fermipy_model["isodiff"] = isotropic_template

    # Now iterate over all sources contained in the likelihood model
    sources = []

    # point sources
    for point_source in list(
        likelihood_model.point_sources.values()
    ):  # type: astromodels.PointSource

        this_source = {"Index": 2.56233, "Scale": 572.78, "Prefactor": 2.4090e-12}
        this_source["name"] = point_source.name
        this_source["ra"] = point_source.position.ra.value
        this_source["dec"] = point_source.position.dec.value

        # The spectrum used here is unconsequential, as it will be substituted by a FileFunction
        # later on. So I will just use PowerLaw for everything
        this_source["SpectrumType"] = "PowerLaw"

        sources.append(this_source)

    # extended sources
    for extended_source in list(
        likelihood_model.extended_sources.values()
    ):  # type: astromodels.ExtendedSource

        raise NotImplementedError("Extended sources are not supported yet")

    # Add all sources to the model
    fermipy_model["sources"] = sources

    # Now we can finally instance the GTAnalysis instance
    configuration["model"] = fermipy_model

    gta = GTAnalysis(configuration)  # noqa: F821

    # This will take a long time if it's the first time we run with this model
    gta.setup()

    # Substitute all spectra for point sources with FileSpectrum, so that we will be able to control
    # them from 3ML

    energies_keV = None

    for point_source in list(
        likelihood_model.point_sources.values()
    ):  # type: astromodels.PointSource

        # Fix this source, so fermipy will not optimize by itself the parameters
        gta.free_source(point_source.name, False)

        # This will substitute the current spectrum with a FileFunction with the same shape and flux
        gta.set_source_spectrum(point_source.name, "FileFunction", update_source=False)

        # Get the energies at which to evaluate this source
        this_log_energies, _flux = gta.get_source_dnde(point_source.name)
        this_energies_keV = (
            10 ** this_log_energies * 1e3
        )  # fermipy energies are in GeV, we need keV

        if energies_keV is None:

            energies_keV = this_energies_keV

        else:

            # This is to make sure that all sources are evaluated at the same energies

            assert np.all(energies_keV == this_energies_keV)

        dnde = point_source(energies_keV)  # ph / (cm2 s keV)
        dnde_per_MeV = dnde * 1000.0  # ph / (cm2 s MeV)
        gta.set_source_dnde(point_source.name, dnde_per_MeV, False)

    # Same for extended source
    for extended_source in list(
        likelihood_model.extended_sources.values()
    ):  # type: astromodels.ExtendedSource

        raise NotImplementedError("Extended sources are not supported yet")

    return gta, energies_keV


def _expensive_imports_hook():

    from fermipy.gtanalysis import GTAnalysis
    from GtBurst.LikelihoodComponent import findGalacticTemplate, findIsotropicTemplate

    globals()["GTAnalysis"] = GTAnalysis
    globals()["findGalacticTemplate"] = findGalacticTemplate
    globals()["findIsotropicTemplate"] = findIsotropicTemplate


class FermipyLike(PluginPrototype):
    """
    Plugin for the data of the Fermi Large Area Telescope, based on fermipy (http://fermipy.readthedocs.io/)
    """

    def __new__(cls, *args, **kwargs):

        instance = object.__new__(cls)

        # we do not catch here

        _expensive_imports_hook()

        return instance

    def __init__(self, name, fermipy_config):
        """
        :param name: a name for this instance
        :param fermipy_config: either a path to a YAML configuration file or a dictionary containing the configuration
        (see http://fermipy.readthedocs.io/)
        """

        # There are no nuisance parameters

        nuisance_parameters = {}

        super(FermipyLike, self).__init__(name, nuisance_parameters=nuisance_parameters)

        # Check whether the provided configuration is a file

        if not isinstance(fermipy_config, dict):

            # Assume this is a file name
            configuration_file = sanitize_filename(fermipy_config)

            assert os.path.exists(fermipy_config), (
                "Configuration file %s does not exist" % configuration_file
            )

            # Read the configuration
            with open(configuration_file) as f:

                self._configuration = yaml.load(f, Loader=yaml.SafeLoader)

        else:

            # Configuration is a dictionary. Nothing to do
            self._configuration = fermipy_config

        # If the user provided a 'model' key, issue a warning, as the model will be defined
        # later on and will overwrite the one contained in 'model'

        if "model" in self._configuration:

            custom_warnings.warn(
                "The provided configuration contains a 'model' section, which is useless as it "
                "will be overridden"
            )

            self._configuration.pop("model")

        if "fileio" in self._configuration:

            custom_warnings.warn(
                "The provided configuration contains a 'fileio' section, which will be "
                "overwritten"
            )

            self._configuration.pop("fileio")

        # Now check that the data exists

        # As minimum there must be a evfile and a scfile
        assert (
            "evfile" in self._configuration["data"]
        ), "You must provide a evfile in the data section"
        assert (
            "scfile" in self._configuration["data"]
        ), "You must provide a scfile in the data section"

        for datum in self._configuration["data"]:

            # Sanitize file name, as fermipy is not very good at handling relative paths or env. variables

            filename = str( sanitize_filename(self._configuration["data"][datum], True) )

            self._configuration["data"][datum] = filename

            assert os.path.exists(
                self._configuration["data"][datum]
            ), "File %s (%s) not found" % (filename, datum)

        # Prepare the 'fileio' part
        # Save all output in a directory with a unique name which depends on the configuration,
        # so that the same configuration will write in the same directory and fermipy will
        # know that it doesn't need to recompute things

        self._unique_id = "__%s" % _get_unique_tag_from_configuration(
            self._configuration
        )

        self._configuration["fileio"] = {"outdir": self._unique_id}

        # Ensure that there is a complete definition of a Region Of Interest (ROI)
        assert ("ra" in self._configuration["selection"]) and (
            "dec" in self._configuration["selection"]
        ), (
            "You have to provide 'ra' and 'dec' in the 'selection' section of the configuration. Source name "
            "resolution, as well as Galactic coordinates, are not currently supported"
        )

        # This is empty at the beginning, will be instanced in the set_model method
        self._gta = None

    @staticmethod
    def get_basic_config(
        evfile,
        scfile,
        ra,
        dec,
        emin=100.0,
        emax=100000.0,
        zmax=100.0,
        evclass=128,
        evtype=3,
        filter="DATA_QUAL>0 && LAT_CONFIG==1",
    ):

        from fermipy.config import ConfigManager

        # Get default config from fermipy
        basic_config = ConfigManager.load(
            get_path_of_data_file("fermipy_basic_config.yml")
        )  # type: dict

        evfile = str(sanitize_filename(evfile) )
        scfile = str(sanitize_filename(scfile) )

        assert os.path.exists(evfile), "The provided evfile %s does not exist" % evfile
        assert os.path.exists(scfile), "The provided scfile %s does not exist" % scfile

        basic_config["data"]["evfile"] = evfile
        basic_config["data"]["scfile"] = scfile

        ra = float(ra)
        dec = float(dec)

        assert 0 <= ra <= 360, (
            "The provided R.A. (%s) is not valid. Should be 0 <= ra <= 360.0" % ra
        )
        assert -90 <= dec <= 90, (
            "The provided Dec (%s) is not valid. Should be -90 <= dec <= 90.0" % dec
        )

        basic_config["selection"]["ra"] = ra
        basic_config["selection"]["dec"] = dec

        emin = float(emin)
        emax = float(emax)

        basic_config["selection"]["emin"] = emin
        basic_config["selection"]["emax"] = emax

        zmax = float(zmax)
        assert 0.0 <= zmax <= 180.0, (
            "The provided Zenith angle cut (zmax = %s) is not valid. "
            "Should be 0 <= zmax <= 180.0" % zmax
        )

        basic_config["selection"]["zmax"] = zmax

        with fits.open(scfile) as ft2_:
            tmin = float(ft2_[0].header["TSTART"])
            tmax = float(ft2_[0].header["TSTOP"])
        
        basic_config["selection"]["tmin"] = tmin
        basic_config["selection"]["tmax"] = tmax

        evclass = int(evclass)
        assert is_power_of_2(evclass), "The provided evclass is not a power of 2."

        basic_config["selection"]["evclass"] = evclass

        evtype = int(evtype)

        basic_config["selection"]["evtype"] = evtype

        basic_config["selection"]["filter"] = filter

        return DictWithPrettyPrint(basic_config)

    @property
    def configuration(self):
        """
        Returns the loaded configuration

        :return: a dictionary containing the active configuration
        """
        return self._configuration

    @property
    def gta(self):

        assert self._gta is not None, (
            "You have to perform a fit or a bayesian analysis before accessing the "
            "gta object"
        )

        return self._gta

    def set_model(self, likelihood_model_instance):
        """
        Set the model to be used in the joint minimization. Must be a LikelihoodModel instance.
        """

        # This will take a long time if it's the first time we run, as it will select the data,
        # produce livetime cube, expomap, source maps and so on

        self._likelihood_model = likelihood_model_instance

        self._gta, self._pts_energies = _get_fermipy_instance(
            self._configuration, likelihood_model_instance
        )

    def _update_model_in_fermipy(self):

        # Substitute all spectra for point sources with FileSpectrum, so that we will be able to control
        # them from 3ML
        for point_source in list(
            self._likelihood_model.point_sources.values()
        ):  # type: astromodels.PointSource

            # Update this source only if it has free parameters (to gain time)
            if point_source.has_free_parameters():

                # Now set the spectrum of this source to the right one
                dnde = point_source(self._pts_energies)  # ph / (cm2 s keV)
                dnde_MeV = dnde * 1000.0  # ph / (cm2 s MeV)

                # NOTE: I use update_source=False because it makes things 100x faster and I verified that
                # it does not change the result.

                self._gta.set_source_dnde(point_source.name, dnde_MeV, False)
            else:

                # Nothing to do for a fixed source_

                continue

        # Same for extended source
        for extended_source in list(
            self._likelihood_model.extended_sources.values()
        ):  # type: astromodels.ExtendedSource

            raise NotImplementedError("Extended sources are not supported yet")

    def get_log_like(self):
        """
        Return the value of the log-likelihood with the current values for the
        parameters stored in the ModelManager instance
        """

        # Update all sources on the fermipy side
        self._update_model_in_fermipy()

        # Get value of the log likelihood

        try:

            value = self._gta.like.logLike.value()

        except:

            raise

        return value - logfactorial(self._gta.like.total_nobs())

    def inner_fit(self):
        """
        This is used for the profile likelihood. Keeping fixed all parameters in the
        LikelihoodModel, this method minimize the logLike over the remaining nuisance
        parameters, i.e., the parameters belonging only to the model for this
        particular detector. If there are no nuisance parameters, simply return the
        logLike value.
        """
        return self.get_log_like()
