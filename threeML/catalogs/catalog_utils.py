from __future__ import division

import re
from builtins import map, str

import numpy
from astromodels import *
from astromodels.utils.angular_distance import angular_distance
from astropy.stats import circmean
from astropy import units as u

from threeML.config.config import threeML_config
from threeML.exceptions.custom_exceptions import custom_warnings
from threeML.io.dict_with_pretty_print import DictWithPrettyPrint
from threeML.io.logging import setup_logger
from pkg_resources import resource_filename
import os.path, os


log = setup_logger(__name__)

_trigger_name_match = re.compile("^GRB\d{9}$")

def _gbm_and_lle_valid_source_check(source):
    """
    checks if source name is valid for both GBM and LLE data

    :param source: source name
    :return: bool
    """

    warn_string = (
        "The trigger %s is not valid. Must be in the form GRB080916009" % source
    )

    match = _trigger_name_match.match(source)

    if match is None:

        log.warning(warn_string)

        answer = False

    else:

        answer = True

    return answer

#########

def _sanitize_fgl_name(fgl_name):
    swap = (
        fgl_name.replace(" ", "_").replace("+", "p").replace("-", "m").replace(".", "d")
    )

    if swap[0] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
        swap = "_%s" % swap

    return swap


def _get_point_source_from_fgl(fgl_name, catalog_entry, fix=False):
    """
    Translate a spectrum from the nFGL into an astromodels point source
    """
    name = _sanitize_fgl_name(fgl_name)

    spectrum_type = catalog_entry["spectrum_type"]
    ra = float(catalog_entry["ra"])
    dec = float(catalog_entry["dec"])

    log.debug(f"source parameters for {fgl_name}")
    log.debug(catalog_entry)
    1

    if spectrum_type == "PowerLaw":

        this_spectrum = Powerlaw()

        this_source = PointSource(name, ra=ra, dec=dec, spectral_shape=this_spectrum)

        this_spectrum.index = float(catalog_entry["pl_index"]) * -1
        this_spectrum.index.fix = fix
        this_spectrum.K = float(catalog_entry["pl_flux_density"]) / (
            u.cm ** 2 * u.s * u.MeV
        )
        this_spectrum.K.fix = fix
        this_spectrum.K.bounds = (
            this_spectrum.K.value / 1000.0,
            this_spectrum.K.value * 1000,
        )
        this_spectrum.piv = float(catalog_entry["pivot_energy"]) * u.MeV

    elif spectrum_type == "LogParabola":

        this_spectrum = Log_parabola()

        this_source = PointSource(name, ra=ra, dec=dec, spectral_shape=this_spectrum)

        this_spectrum.alpha = float(catalog_entry["lp_index"]) * -1
        this_spectrum.alpha.fix = fix
        this_spectrum.beta = float(catalog_entry["lp_beta"])
        this_spectrum.beta.fix = fix
        this_spectrum.piv = float(catalog_entry["pivot_energy"]) * u.MeV
        this_spectrum.K = float(catalog_entry["lp_flux_density"]) / (
            u.cm ** 2 * u.s * u.MeV
        )
        this_spectrum.K.fix = fix
        this_spectrum.K.bounds = (
            this_spectrum.K.value / 1000.0,
            this_spectrum.K.value * 1000,
        )

    elif spectrum_type == "PLExpCutoff":

        this_spectrum = Cutoff_powerlaw()

        this_source = PointSource(name, ra=ra, dec=dec, spectral_shape=this_spectrum)

        this_spectrum.index = float(catalog_entry["plec_index"]) * -1
        this_spectrum.index.fix = fix
        this_spectrum.piv = float(catalog_entry["pivot_energy"]) * u.MeV
        this_spectrum.K = float(catalog_entry["plec_flux_density"]) / (
            u.cm ** 2 * u.s * u.MeV
        )
        this_spectrum.K.fix = fix
        this_spectrum.K.bounds = (
            this_spectrum.K.value / 1000.0,
            this_spectrum.K.value * 1000,
        )
        this_spectrum.xc = float(catalog_entry["cutoff"]) * u.MeV
        this_spectrum.xc.fix = fix

    elif spectrum_type in ["PLSuperExpCutoff", "PLSuperExpCutoff2"]:
        # This is the new definition, from the 4FGL catalog.
        # Note that in version 19 of the 4FGL, cutoff spectra are designated as PLSuperExpCutoff
        # rather than PLSuperExpCutoff2 as in version , but the same parametrization is used.
        this_spectrum = Super_cutoff_powerlaw()

        this_source = PointSource(name, ra=ra, dec=dec, spectral_shape=this_spectrum)
        # new parameterization 4FGLDR3:
        if ('plec_index_s' in catalog_entry.keys()):
            d  = float(catalog_entry["plec_exp_factor_s"])
            E0 = float(catalog_entry["pivot_energy"]) * u.MeV
            b  = float(catalog_entry["plec_exp_index"])
            Gs = float(catalog_entry["plec_index_s"])

            conv = numpy.exp(d/b ** 2)
            this_spectrum.index =  d/b - Gs
            this_spectrum.index.fix = fix
            this_spectrum.gamma = d/b
            this_spectrum.gamma.fix = fix
            this_spectrum.piv = E0
            this_spectrum.K = (
                conv * float(catalog_entry["plec_flux_density"]) / (u.cm ** 2 * u.s * u.MeV)
            )
            this_spectrum.xc =  E0
        else:
            # OLD parameterization 4FGL which is in fermipy:
            a = float(catalog_entry["plec_exp_factor"])
            E0 = float(catalog_entry["pivot_energy"])
            b = float(catalog_entry["plec_exp_index"])

            conv = numpy.exp(a * E0 ** b)
            this_spectrum.index = float(catalog_entry["plec_index"]) * -1
            this_spectrum.index.fix = fix
            this_spectrum.gamma = b
            this_spectrum.gamma.fix = fix
            this_spectrum.piv = E0 * u.MeV
            this_spectrum.K = (
                    conv * float(catalog_entry["plec_flux_density"]) / (u.cm ** 2 * u.s * u.MeV)
            )
            this_spectrum.xc = a ** (-1.0 / b ) * u.MeV

        this_spectrum.K.fix = fix
        this_spectrum.K.bounds = (
            this_spectrum.K.value / 1000.0,
            this_spectrum.K.value * 1000,
        )
        this_spectrum.xc.fix = fix

    else:

        raise NotImplementedError(
            "Spectrum type %s is not a valid 4FGL type" % spectrum_type
        )

    return this_source


def _get_extended_source_from_fgl(fgl_name, catalog_entry, fix=False):
    """
    Translate a spectrum from the nFGL into an astromodels extended source
    """

    name = _sanitize_fgl_name(fgl_name)

    spectrum_type = catalog_entry["spectrum_type"]
    ra = float(catalog_entry["ra"])
    dec = float(catalog_entry["dec"])

    if catalog_entry["spatial_function"] == "RadialDisk":
        this_shape = Disk_on_sphere()
    elif catalog_entry["spatial_function"] == "RadialGaussian":
        this_shape = Gaussian_on_sphere()
    elif catalog_entry["spatial_function"] == "SpatialMap":
        the_file = catalog_entry["spatial_filename"]
        if isinstance(the_file, bytes):
            the_file = the_file.decode("ascii")

        if "FERMIPY_DATA_DIR" not in os.environ:
            os.environ["FERMIPY_DATA_DIR"] = resource_dir("fermipy", "data")

        the_dir = os.path.join(os.path.expandvars(catalog_entry["extdir"]), "Templates")

        the_template = os.path.join(the_dir, the_file)

        #this_shape(fits_file=the_template)

        this_shape = SpatialTemplate_2D(fits_file=the_template)

    else:
        log.error("Spatial_Function {} not implemented yet" % catalog_entry["Spatial_Function"] )
        raise NotImplementedError()

    if spectrum_type == "PowerLaw":

        this_spectrum = Powerlaw()

        this_source = ExtendedSource(name, spatial_shape = this_shape, spectral_shape=this_spectrum)

        this_spectrum.index = float(catalog_entry["pl_index"]) * -1
        this_spectrum.index.fix = fix
        this_spectrum.K = float(catalog_entry["pl_flux_density"]) / (
            u.cm ** 2 * u.s * u.MeV
        )
        this_spectrum.K.fix = fix
        this_spectrum.K.bounds = (
            this_spectrum.K.value / 1000.0,
            this_spectrum.K.value * 1000,
        )
        this_spectrum.piv = float(catalog_entry["pivot_energy"]) * u.MeV

    elif spectrum_type == "LogParabola":

        this_spectrum = Log_parabola()

        this_source = ExtendedSource(name, spatial_shape = this_shape, spectral_shape=this_spectrum)

        this_spectrum.alpha = float(catalog_entry["lp_index"]) * -1
        this_spectrum.alpha.fix = fix
        this_spectrum.beta = float(catalog_entry["lp_beta"])
        this_spectrum.beta.fix = fix
        this_spectrum.piv = float(catalog_entry["pivot_energy"]) * u.MeV
        this_spectrum.K = float(catalog_entry["lp_flux_density"]) / (
            u.cm ** 2 * u.s * u.MeV
        )
        this_spectrum.K.fix = fix
        this_spectrum.K.bounds = (
            this_spectrum.K.value / 1000.0,
            this_spectrum.K.value * 1000,
        )

    elif spectrum_type == "PLExpCutoff":

        this_spectrum = Cutoff_powerlaw()

        this_source = ExtendedSource(name, spatial_shape = this_shape, spectral_shape=this_spectrum)

        this_spectrum.index = float(catalog_entry["plec_index"]) * -1
        this_spectrum.index.fix = fix
        this_spectrum.piv = float(catalog_entry["pivot_energy"]) * u.MeV
        this_spectrum.K = float(catalog_entry["plec_flux_density"]) / (
            u.cm ** 2 * u.s * u.MeV
        )
        this_spectrum.K.fix = fix
        this_spectrum.K.bounds = (
            this_spectrum.K.value / 1000.0,
            this_spectrum.K.value * 1000,
        )
        this_spectrum.xc = float(catalog_entry["cutoff"]) * u.MeV
        this_spectrum.xc.fix = fix

    elif spectrum_type in ["PLSuperExpCutoff", "PLSuperExpCutoff2"]:
        # This is the new definition, from the 4FGL catalog.
        # Note that in version 19 of the 4FGL, cutoff spectra are designated as PLSuperExpCutoff
        # rather than PLSuperExpCutoff2 as in version , but the same parametrization is used.
        this_spectrum = Super_cutoff_powerlaw()

        this_source = ExtendedSource(name, spatial_shape = this_shape, spectral_shape=this_spectrum)

        # new parameterization 4FGLDR3:
        if ('plec_index_s' in catalog_entry.keys()):
            d  = float(catalog_entry["plec_exp_factor_s"])
            E0 = float(catalog_entry["pivot_energy"]) * u.MeV
            b  = float(catalog_entry["plec_exp_index"])
            Gs = float(catalog_entry["plec_index_s"])

            conv = numpy.exp(d/b ** 2)
            this_spectrum.index =  d/b - Gs
            this_spectrum.index.fix = fix
            this_spectrum.gamma = d/b
            this_spectrum.gamma.fix = fix
            this_spectrum.piv = E0
            this_spectrum.K = (
                conv * float(catalog_entry["plec_flux_density"]) / (u.cm ** 2 * u.s * u.MeV)
            )
            this_spectrum.xc =  E0
        else:
            # OLD parameterization 4FGL which is in fermipy:
            a = float(catalog_entry["plec_exp_factor"])
            E0 = float(catalog_entry["pivot_energy"])
            b = float(catalog_entry["plec_exp_index"])

            conv = numpy.exp(a * E0 ** b)
            this_spectrum.index = float(catalog_entry["plec_index"]) * -1
            this_spectrum.index.fix = fix
            this_spectrum.gamma = b
            this_spectrum.gamma.fix = fix
            this_spectrum.piv = E0 * u.MeV
            this_spectrum.K = (
                    conv * float(catalog_entry["plec_flux_density"]) / (u.cm ** 2 * u.s * u.MeV)
            )
            this_spectrum.xc = a ** (-1.0 / b ) * u.MeV


        this_spectrum.K.fix = fix
        this_spectrum.K.bounds = (
            this_spectrum.K.value / 1000.0,
            this_spectrum.K.value * 1000,
        )
        this_spectrum.xc.fix = fix

    else:
        log.error(  "Spectrum type %s is not a valid 4FGL type" % spectrum_type )
        raise NotImplementedError()


    if catalog_entry["spatial_function"] == "RadialDisk":
    
        this_shape.lon0 = ra * u.degree
        this_shape.lon0.fix = True
        this_shape.lat0 = dec * u.degree
        this_shape.lat0.fix = True
        this_shape.radius = catalog_entry["Model_SemiMajor"] * u.degree
        this_shape.radius.fix = True
        this_shape.radius.bounds = (0, catalog_entry["Model_SemiMajor"]) * u.degree
        
    elif catalog_entry["spatial_function"] == "RadialGaussian":
        
        #factor 1/1.36 is the conversion from 68% containment radius to sigma
        #Max of sigma/radius is used for get_boundaries().
            
        this_shape.lon0 = ra * u.degree
        this_shape.lon0.fix = True
        this_shape.lat0 = dec * u.degree
        this_shape.lat0.fix = True
        this_shape.sigma = catalog_entry["model_semimajor"] / 1.36 * u.degree
        this_shape.sigma.fix = True
        this_shape.sigma.bounds = (0, catalog_entry["model_semimajor"] / 1.36) * u.degree

    return this_source


class ModelFromFGL(Model):
    def __init__(self, ra_center, dec_center, *sources):

        self._ra_center = float(ra_center)
        self._dec_center = float(dec_center)

        super(ModelFromFGL, self).__init__(*sources)

    def free_point_sources_within_radius(self, radius, normalization_only=True):
        """
        Free the parameters for the point sources within the given radius of the center of the search cone

        :param radius: radius in degree
        :param normalization_only: if True, frees only the normalization of the source (default: True)
        :return: none
        """
        self._free_or_fix_ps(True, radius, normalization_only)

    def fix_point_sources_within_radius(self, radius, normalization_only=True):
        """
        Fixes the parameters for the point sources within the given radius of the center of the search cone

        :param radius: radius in degree
        :param normalization_only: if True, fixes only the normalization of the source (default: True)
        :return: none
        """
        self._free_or_fix_ps(False, radius, normalization_only)

    def _free_or_fix_ps(self, free, radius, normalization_only):

        for src_name in self.point_sources:

            src = self.point_sources[src_name]

            this_d = angular_distance(
                self._ra_center,
                self._dec_center,
                src.position.ra.value,
                src.position.dec.value,
            )

            if this_d <= radius:

                if normalization_only:

                    src.spectrum.main.shape.K.free = free

                else:

                    for par in src.spectrum.main.shape.parameters:
                        
                        if par == "piv": #don't free pivot energy
                            continue
                        
                        src.spectrum.main.shape.parameters[par].free = free

    def free_extended_sources_within_radius(self, radius, normalization_only=True):
        """
        Free the parameters for the point sources within the given radius of the center of the search cone

        :param radius: radius in degree
        :param normalization_only: if True, frees only the normalization of the source (default: True)
        :return: none
        """
        self._free_or_fix_ext(True, radius, normalization_only)

    def fix_extended_sources_within_radius(self, radius, normalization_only=True):
        """
        Fixes the parameters for the point sources within the given radius of the center of the search cone

        :param radius: radius in degree
        :param normalization_only: if True, fixes only the normalization of the source (default: True)
        :return: none
        """
        self._free_or_fix_ext(False, radius, normalization_only)

    def _free_or_fix_ext(self, free, radius, normalization_only):

        for src_name in self.extended_sources:

            src = self.extended_sources[src_name]
            
            try:
                ra, dec = src.spatial_shape.lon0.value, src.spatial_shape.lat0.value
                
            except:
            
                (ra_min, ra_max), (dec_min, dec_max) = src.spatial_shape.get_boundaries()
                ra = circmean( [ra_min, ra_max]*u.deg ).value
                dec = circmean( [dec_min, dec_max]*u.deg ).value


            this_d = angular_distance(
                self._ra_center,
                self._dec_center,
                ra,
                dec,
            )

            if this_d <= radius:

                if normalization_only:

                    src.spectrum.main.shape.K.free = free

                else:

                    for par in src.spectrum.main.shape.parameters:
                        
                        if par == "piv": #don't free pivot energy
                            continue
                        
                        src.spectrum.main.shape.parameters[par].free = free

