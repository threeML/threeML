from astromodels import *
from threeML import *

from threeML.catalogs.catalog_utils import _sanitize_fgl_name
from astropy.coordinates import SkyCoord
import pytest
import numpy as np

from threeML.io.logging import setup_logger
log = setup_logger(__name__)
import copy
import yaml

from threeML.io.network import internet_connection_is_active

skip_if_internet_is_not_available = pytest.mark.skipif(
    not internet_connection_is_active(), reason="No active internet connection"
)

skip_if_fermipy_is_not_available = pytest.mark.skipif(
    not is_plugin_available("FermipyLike"), reason="No LAT environment installed"
)


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


def do_the_test(cat_name):
    from fermipy.gtanalysis import GTAnalysis

    gta = GTAnalysis(f"2config_Crab_{cat_name}.yaml",logging={'verbosity' : 3})
    gta.setup()
    gta.write_roi(f"roi_{cat_name}")

    gta = GTAnalysis.create(f"roi_{cat_name}.npy")

    lat_catalog = FermiPySourceCatalog(f"roi_{cat_name}.fits")
    ra, dec, table = lat_catalog.search_around_source("Crab", radius=30.0)
    model_fits = lat_catalog.get_model()

    lat_catalog = FermiPySourceCatalog(cat_name)
    table = lat_catalog.cone_search(ra, dec, radius=30.0)
    model_cat = lat_catalog.get_model(use_association_name=False)

    if cat_name == "4FGL-DR3":
        lat_catalog = FermiLATSourceCatalog()
        table = lat_catalog.cone_search(ra, dec, radius=30.0)
        model_vo = lat_catalog.get_model(use_association_name=False)


    for source in gta.get_sources():
        source = gta.roi.get_source_by_name(source["name"])
        name = gta.roi.get_source_by_name(source["name"]).name

        if "diff" in name:
            continue

        astro_name = _sanitize_fgl_name(name)
    
        e, f_fermipy = gta.get_source_dnde(name)
        e = 10**e


        fa_fits = (model_fits[astro_name].spectrum.main.shape(e*u.MeV)).to(u.cm**-2 / u.s / u.MeV).value
        fa_cat = (model_cat[astro_name].spectrum.main.shape(e*u.MeV)).to(u.cm**-2 / u.s / u.MeV).value
        
        if cat_name == "4FGL-DR3":
            fa_vo = (model_vo[astro_name](e*u.MeV)).to(u.cm**-2 / u.s / u.MeV).value if astro_name in model_vo.sources else np.nan
        print ('--------------------------------',name,cat_name)
        assert np.allclose( f_fermipy, fa_fits) and np.allclose(f_fermipy, fa_cat)
        assert cat_name != "4FGL-DR3" or np.allclose(f_fermipy, fa_vo)
            
        if type( model_cat[astro_name] ) == PointSource:
            pos_fits = model_fits[astro_name].position.sky_coord
            pos_cat = model_cat[astro_name].position.sky_coord
            pos_fermipy = SkyCoord( source["ra"], source["dec"], frame="icrs", unit="deg")
            
            assert( pos_fits.separation(pos_cat).value < 1e-3)
            assert( pos_fits.separation(pos_fermipy).value < 1e-3)
    
        continue


        plt.loglog( e, e**2 * f_fermipy, "b-", label = "fermipy", alpha=0.7)
        plt.loglog( e, e**2 * fa_fits, "r--", label = "from ROI fits", alpha=0.7)
        plt.loglog( e, e**2 * fa_cat, "g:", label = "from catalog", alpha=0.7)
        
        if cat_name == "4FGL-DR3":
            plt.loglog( e, e**2 * fa_vo, "y-.", label = "from VO", alpha=0.7)
    
        plt.title(name)
        plt.legend( title=model_fits[astro_name].spectrum.main.shape.name )
            
        plt.xlabel("Energy (MeV)")
        plt.ylabel("$E^2$ dN/dE (MeV/cm$^2$/s)")
        
        plt.grid()
        plt.show()
 

@skip_if_internet_is_not_available
@skip_if_fermipy_is_not_available
@pytest.mark.xfail
def test_read_model_from_catalogs():


    #Find crab and download data from Jan 01 2010 to Jan 2 2010 (needed for fermipy instance)
    
    lat_catalog = FermiLATSourceCatalog()
    ra, dec, table = lat_catalog.search_around_source("Crab", radius=20.0)

    tstart = "2010-01-01 00:00:00"
    tstop  = "2010-01-08 00:00:00"

    # Note that this will understand if you already download these files, and will
    # not do it twice unless you change your selection or the outdir

    try:

        evfile, scfile = download_LAT_data(
            ra,
            dec,
            20.0,
            tstart,
            tstop,
            time_type="Gregorian",
            destination_directory="Crab_data",
        )

    except RuntimeError:
    
        log.warning("Problems with LAT data download, will not proceed with tests.")
        
        return

    # Configuration for Fermipy

    config = FermipyLike.get_basic_config(evfile=evfile, scfile=scfile, ra=ra, dec=dec)
 
    config["binning"]["binsz"] = 0.5
    config["binning"]["roiwidth"] = 30
    
    irfs = evclass_irf[int(config["selection"]["evclass"])]
    config["gtlike"] = {"irfs":irfs, "edisp":False}
 
    for cat_name in ["4FGL", "4FGL-DR2", "4FGL-DR3"]:
    
        the_config = copy.deepcopy(config)
        
        model_dict = { "src_roiwidth" : 30.0,
                        "galdiff"  : "$CONDA_PREFIX/share/fermitools/refdata/fermi/galdiffuse/gll_iem_v07.fits",
                        "isodiff"  : "iso_P8R3_SOURCE_V3_v1.txt",
                        "catalogs": [cat_name],
                        }
        
        the_config["model"] = model_dict
        
        stream = open(f"2config_Crab_{cat_name}.yaml", 'w')
        yaml.dump(dict(the_config), stream, default_flow_style=False)
    
        
        do_the_test(cat_name)
