from __future__ import division
from past.utils import old_div
import astromodels
import numpy as np
import os
import yaml
import astropy.io.fits as fits
from astropy.stats import circmean
from astropy import units as u
import collections

from astromodels import Model, Parameter
from astromodels.core import parameter_transformation

from threeML.exceptions.custom_exceptions import custom_warnings
from threeML.io.file_utils import sanitize_filename
from threeML.plugin_prototype import PluginPrototype
from threeML.utils.statistics.gammaln import logfactorial
from threeML.utils.unique_deterministic_tag import get_unique_deterministic_tag
from threeML.utils.power_of_two_utils import is_power_of_2
from threeML.io.package_data import get_path_of_data_file
from threeML.io.dict_with_pretty_print import DictWithPrettyPrint
from threeML.io.logging import setup_logger

log = setup_logger(__name__)

from threeML.io.logging import setup_logger
log = setup_logger(__name__)

__instrument_name = "Fermi LAT (with fermipy)"


#########################################################
# NOTE:
# Fermipy does NOT support Unbinned Likelihood analysis
#########################################################


# A lookup map for the correspondence between IRFS and evclass
# See https://fermi.gsfc.nasa.gov/ssc/data/analysis/documentation/Cicerone/Cicerone_Data/LAT_DP.html#PhotonClassification
evclass_irf = {
    8: "P8R3_TRANSIENT020E_V3",
    16: "P8R3_TRANSIENT020_V3",
    32: "P8R3_TRANSIENT010E_V3",
    64: "P8R3_TRANSIENT010_V3",
    128: "P8R3_SOURCE_V3",
    256: "P8R3_CLEAN_V3",
    512: "P8R3_ULTRACLEAN_V3",
    1024: "P8R3_ULTRACLEANVETO_V3",
    2048: "P8R3_SOURCEVETO_V3",
    65536: "P8R3_TRANSIENT015S_V3",
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

        if not section in configuration:
            log.critical(
                "Configuration lacks section %s, which is required" % section
            )

        for key in keys:

            if not key in configuration[section]:
                log.critical(
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

    log.info(f"Using IRFs {irfs}")

    if "gtlike" in configuration and "irfs" in configuration["gtlike"]:

        if irfs.upper() != configuration["gtlike"]["irfs"].upper():
            log.critical(
                "Evclass points to IRFS %s, while you specified %s in the "
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

        this_source = {"Index": 2.56233, "Scale": 572.78, "Prefactor": 2.4090e-12}
        this_source["name"] = extended_source.name
        # The spectrum used here is unconsequential, as it will be substituted by a FileFunction
        # later on. So I will just use PowerLaw for everything
        this_source["SpectrumType"] = "PowerLaw"
        
        theShape = extended_source.spatial_shape
        
        if theShape.name == "Disk_on_sphere":
            this_source["SpatialModel"] = "RadialDisk"
            this_source["ra"] = theShape.lon0.value
            this_source["dec"] = theShape.lat0.value
            this_source["SpatialWidth"] = theShape.radius.value

        elif theShape.name == "Gaussian_on_sphere":
            this_source["SpatialModel"] = "RadialGaussian"
            this_source["ra"] = theShape.lon0.value
            this_source["dec"] = theShape.lat0.value
            #fermipy/fermi tools expect 68% containment radius = 1.36 sigma
            this_source["SpatialWidth"] = 1.36 * theShape.sigma.value

        elif theShape.name == "SpatialTemplate_2D":
            
            try:
                (ra_min, ra_max), (dec_min, dec_max) = theShape.get_boundaries()
                this_source["ra"] = circmean( [ra_min, ra_max]*u.deg ).value
                this_source["dec"] = circmean( [dec_min, dec_max]*u.deg ).value
                
            except:
                log.critical( f"Source {extended_source.name} does not have a template file set; must call read_file first()"  )
                
            this_source["SpatialModel"] = "SpatialMap"
            this_source["Spatial_Filename"] = theShape._fitsfile

        else:
        
            log.critical(f"Extended source {extended_source.name}: shape {theShape.name} not yet implemented for FermipyLike")

        sources.append(this_source)


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

            if not np.all(energies_keV == this_energies_keV):
                log.critical("All sources should be evaluated at the same energies.")

        dnde = point_source(energies_keV)  # ph / (cm2 s keV)
        dnde_per_MeV = np.maximum(dnde * 1000.0, 1e-300) # ph / (cm2 s MeV)
        gta.set_source_dnde(point_source.name, dnde_per_MeV, False)

    # Same for extended source
    for extended_source in list(
        likelihood_model.extended_sources.values()
    ):  # type: astromodels.ExtendedSource

        # This will substitute the current spectrum with a FileFunction with the same shape and flux
        gta.set_source_spectrum(extended_source.name, "FileFunction", update_source=False)

        # Get the energies at which to evaluate this source
        this_log_energies, _flux = gta.get_source_dnde(extended_source.name)
        this_energies_keV = (
            10 ** this_log_energies * 1e3
        )  # fermipy energies are in GeV, we need keV

        if energies_keV is None:

            energies_keV = this_energies_keV

        else:

            # This is to make sure that all sources are evaluated at the same energies

            if not np.all(energies_keV == this_energies_keV):
                log.critical("All sources should be evaluated at the same energies.")

        dnde = extended_source.get_spatially_integrated_flux(energies_keV)  # ph / (cm2 s keV)
        dnde_per_MeV = np.maximum(dnde * 1000.0, 1e-300) # ph / (cm2 s MeV)
        gta.set_source_dnde(extended_source.name, dnde_per_MeV, False)

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

            if not os.path.exists(fermipy_config):
                log.critical(
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
        if not "evfile" in self._configuration["data"]:
            log.critical( "You must provide a evfile in the data section" )
        if not "scfile" in self._configuration["data"]:
            log.critical( "You must provide a scfile in the data section" )

        for datum in self._configuration["data"]:

            # Sanitize file name, as fermipy is not very good at handling relative paths or env. variables

            filename = str( sanitize_filename(self._configuration["data"][datum], True) )

            self._configuration["data"][datum] = filename

            if not os.path.exists( self._configuration["data"][datum] ):
                log.critical( "File %s (%s) not found" % (filename, datum) )

        # Prepare the 'fileio' part
        # Save all output in a directory with a unique name which depends on the configuration,
        # so that the same configuration will write in the same directory and fermipy will
        # know that it doesn't need to recompute things

        self._unique_id = "__%s" % _get_unique_tag_from_configuration(
            self._configuration
        )

        self._configuration["fileio"] = {"outdir": self._unique_id}

        # Ensure that there is a complete definition of a Region Of Interest (ROI)
        if not (("ra" in self._configuration["selection"]) and ("dec" in self._configuration["selection"])):
            log.critical(
                "You have to provide 'ra' and 'dec' in the 'selection' section of the configuration. Source name "
                "resolution, as well as Galactic coordinates, are not currently supported"
            )

        # This is empty at the beginning, will be instanced in the set_model method
        self._gta = None
 
        self.set_inner_minimization(True)


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
        fermipy_verbosity = 2,
        fermitools_chatter = 2,
    ):

        from fermipy.config import ConfigManager

        # Get default config from fermipy
        basic_config = ConfigManager.load(
            get_path_of_data_file("fermipy_basic_config.yml")
        )  # type: dict

        evfile = str(sanitize_filename(evfile) )
        scfile = str(sanitize_filename(scfile) )

        if not os.path.exists(evfile):
            log.critical( "The provided evfile %s does not exist" % evfile )
        if not os.path.exists(scfile):
            log.critical( "The provided scfile %s does not exist" % scfile )

        basic_config["data"]["evfile"] = evfile
        basic_config["data"]["scfile"] = scfile

        ra = float(ra)
        dec = float(dec)

        if not (( 0 <= ra) and ( ra <= 360 )):
            log.critical(
                "The provided R.A. (%s) is not valid. Should be 0 <= ra <= 360.0" % ra
            )
        if not ((-90 <= dec) and (dec <= 90)):
            log.critical (
                "The provided Dec (%s) is not valid. Should be -90 <= dec <= 90.0" % dec
            )

        basic_config["selection"]["ra"] = ra
        basic_config["selection"]["dec"] = dec

        emin = float(emin)
        emax = float(emax)

        basic_config["selection"]["emin"] = emin
        basic_config["selection"]["emax"] = emax

        zmax = float(zmax)
        if not (( 0.0 <= zmax) and (zmax <= 180.0)):
            log.critical(
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
        if not is_power_of_2(evclass):
            log.critical( "The provided evclass is not a power of 2." )

        basic_config["selection"]["evclass"] = evclass

        evtype = int(evtype)

        basic_config["selection"]["evtype"] = evtype

        basic_config["selection"]["filter"] = filter

        basic_config["logging"]["verbosity"] = fermipy_verbosity
        #(In fermipy convention, 0 = critical only, 1 also errors, 2 also warnings, 3 also info, 4 also debug)
        basic_config["logging"]["chatter"] = fermitools_chatter #0 = no screen output. 2 = some output, 4 = lot of output.


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

        if self._gta is None:
            log.warning(
                "You have to perform a fit or a bayesian analysis before accessing the "
                "gta object. Returning None"
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
        self._update_model_in_fermipy( update_dictionary = True, force_update = True)
            
        # Build the list of the nuisance parameters
        new_nuisance_parameters = self._set_nuisance_parameters()
        self.update_nuisance_parameters(new_nuisance_parameters)


    def _update_model_in_fermipy(self, update_dictionary = False, delta = 0.0, force_update = False):

        # Substitute all spectra for point sources with FileSpectrum, so that we will be able to control
        # them from 3ML
        for point_source in list(
            self._likelihood_model.point_sources.values()
        ):  # type: astromodels.PointSource

  
            #Update this source only if it has free parameters (to gain time)
            if not ( point_source.has_free_parameters or force_update):
                continue

            #Update source position if free
            if force_update or ( point_source.position.ra.free or point_source.position.dec.free ):
 
                model_pos = point_source.position.sky_coord
                fermipy_pos = self._gta.roi.get_source_by_name(point_source.name).skydir
                
                if  model_pos.separation( fermipy_pos ).to("degree").value > delta :
                    #modeled after how this is done in fermipy
                    #(cf https://fermipy.readthedocs.io/en/latest/_modules/fermipy/sourcefind.html#SourceFind.localize)
                    temp_source = self._gta.delete_source(point_source.name)
                    temp_source.set_position( model_pos )
                    self._gta.add_source(point_source.name, temp_source, free=False)
                    self._gta.free_source(point_source.name, False)
                    self._gta.set_source_spectrum(point_source.name, "FileFunction", update_source=update_dictionary)

            # Now set the spectrum of this source to the right one
            dnde = point_source(self._pts_energies)  # ph / (cm2 s keV)
            dnde_MeV = np.maximum(dnde * 1000.0, 1e-300)  # ph / (cm2 s MeV)
            # NOTE: I use update_source=False because it makes things 100x faster and I verified that
            # it does not change the result.
            # (HF: Not sure who wrote the above but I think sometimes we do want to update fermipy dictionaries.)

            self._gta.set_source_dnde(point_source.name, dnde_MeV, update_source = update_dictionary)
                
        # Same for extended source
        for extended_source in list(
            self._likelihood_model.extended_sources.values()
        ):  # type: astromodels.ExtendedSource

            #Update this source only if it has free parameters (to gain time)
            if not ( extended_source.has_free_parameters or force_update):
                continue

            theShape = extended_source.spatial_shape
            if theShape.has_free_parameters or force_update:
        
                fermipySource = self._gta.roi.get_source_by_name(extended_source.name)
                fermipyPars = [fermipySource["ra"], fermipySource["dec"], fermipySource["SpatialWidth"] ]

                if theShape.name == "Disk_on_sphere":
                
                    amPars = [theShape.lon0.value, theShape.lat0.value, theShape.radius.value]
                    if not np.allclose( fermipyPars, amPars, 1e-10):
                
                        temp_source = self._gta.delete_source(extended_source.name)
                        temp_source.set_spatial_model("RadialDisk", {'ra': theShape.lon0.value,
                            'dec': theShape.lat0.value, 'SpatialWidth': theShape.radius.value})
                        # from fermipy: FIXME: Issue with source map cache with source is initialized as fixed.
                        self._gta.add_source(extended_source.name, temp_source, free=True)
                        self._gta.free_source(extended_source.name, free=False)
                        self._gta.set_source_spectrum(extended_source.name, "FileFunction", update_source=update_dictionary)


                elif theShape.name == "Gaussian_on_sphere":

                    amPars = [theShape.lon0.value, theShape.lat0.value, 1.36*theShape.sigma.value]
                    if not np.allclose( fermipyPars, amPars, 1e-10):
                
                        temp_source = self._gta.delete_source(extended_source.name)
                        temp_source.set_spatial_model("RadialGaussian", {'ra': theShape.lon0.value,
                            'dec': theShape.lat0.value, 'SpatialWidth': 1.36*theShape.sigma.value})
                        # from fermipy: FIXME: Issue with source map cache with source is initialized as fixed.
                        self._gta.add_source(extended_source.name, temp_source, free=True)
                        self._gta.free_source(extended_source.name, free=False)
                        self._gta.set_source_spectrum(extended_source.name, "FileFunction", update_source=update_dictionary)

                elif theShape.name == "SpatialTemplate_2D":
                    #for now, assume we're not updating the fits file
                    pass

                else:
                    #eventually, implement other shapes here.
                    pass

            # Now set the spectrum of this source to the right one
            dnde = extended_source.get_spatially_integrated_flux(self._pts_energies)  # ph / (cm2 s keV)
            dnde_MeV = np.maximum(dnde * 1000.0, 1e-300)  # ph / (cm2 s MeV)
            # NOTE: I use update_source=False because it makes things 100x faster and I verified that
            # it does not change the result.
            # (HF: Not sure who wrote the above but I think sometimes we do want to update fermipy dictionaries.)
            self._gta.set_source_dnde(extended_source.name, dnde_MeV, update_source = update_dictionary)

    def get_log_like(self):
        """
        Return the value of the log-likelihood with the current values for the
        parameters stored in the ModelManager instance
        """

        # Update all sources on the fermipy side
        self._update_model_in_fermipy()

        #update nuisance parameters
        if self._fit_nuisance_params:

            for parameter in self.nuisance_parameters:
                self.set_nuisance_parameter_value(parameter, self.nuisance_parameters[parameter].value)

            #self.like.syncSrcParams()


        # Get value of the log likelihood
        try:

            value = self._gta.like.logLike.value()

        except:

            raise

        return value - logfactorial(int(self._gta.like.total_nobs()))

    def inner_fit(self):
        """
        This is used for the profile likelihood. Keeping fixed all parameters in the
        LikelihoodModel, this method minimize the logLike over the remaining nuisance
        parameters, i.e., the parameters belonging only to the model for this
        particular detector. If there are no nuisance parameters, simply return the
        logLike value.
        """
        
        return self.get_log_like()

    def set_inner_minimization(self, flag: bool) -> None:

        """
        Turn on the minimization of the internal Fermi
        parameters

        :param flag: turing on and off the minimization  of the Fermipy internal parameters
        :type flag: bool
        :returns:

        """
        self._fit_nuisance_params: bool = bool(flag)

        for parameter in self.nuisance_parameters:

            self.nuisance_parameters[parameter].free = self._fit_nuisance_params


    def get_number_of_data_points(self):
        """
        Return the number of spatial/energy bins

        :return: number of bins
        """
        
        num = len(self._gta.components)
        
        if self._gta.projtype == "WCS":

            num = num * self._gta._enumbins * int(self._gta.npix[0]) * int(self._gta.npix[1])

        if self._gta.projtype == "HPX":
        
            num = num * np.sum(self.geom.npix)
            
        return num

    def _set_nuisance_parameters(self):

        # Get the list of the sources
        sources = list(self.gta.roi.get_sources() )
        sources = [s.name for s in sources if "diff" in s.name]
        

        bg_param_names = []
        nuisance_parameters = collections.OrderedDict()

        for src_name in sources:

            if self._fit_nuisance_params:
                self.gta.free_norm(src_name)
    
            pars = self.gta.get_free_source_params(src_name)
                        
            for par in pars:
            
                thisName = f"{self.name}_{src_name}_{par}"
                bg_param_names.append(thisName)

                thePar = self.gta._get_param( src_name, par)

                value = thePar["value"] * thePar["scale"]

                nuisance_parameters[thisName] = Parameter(
                    thisName,
                    value,
                    min_value=thePar["min"],
                    max_value=thePar["max"],
                    delta=0.01*value,
                    transformation=parameter_transformation.get_transformation("log10")
                )

                nuisance_parameters[thisName].free = self._fit_nuisance_params
                
                log.debug(f"Added nuisance parameter {nuisance_parameters[thisName]}")

        return nuisance_parameters

    def _split_nuisance_parameter(self, param_name):

        tokens = param_name.split("_")
        pname = tokens[-1]
        src_name = "_".join(tokens[1:-1])
        
        return src_name, pname
        
    def set_nuisance_parameter_value(self, paramName, value):

        srcName, parName = self._split_nuisance_parameter(paramName)
        self.gta.set_parameter(srcName, parName, value, scale = 1, update_source=False)

