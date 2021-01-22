import pytest

from threeML import *
from threeML.io.network import internet_connection_is_active

skip_if_internet_is_not_available = pytest.mark.skipif(
    not internet_connection_is_active(), reason="No active internet connection"
)

skip_if_fermipy_is_not_available = pytest.mark.skipif(
    not is_plugin_available("FermipyLike"), reason="No LAT environment installed"
)


@skip_if_internet_is_not_available
@skip_if_fermipy_is_not_available
@pytest.mark.xfail
def test_FermipyLike():
    from threeML.plugins.FermipyLike import FermipyLike

    # Crab coordinates

    lat_catalog = FermiLATSourceCatalog()

    ra, dec, table = lat_catalog.search_around_source("Crab", radius=20.0)

    assert np.isclose(ra, 83.6330906247)
    assert np.isclose(dec, 22.0144947866)

    # This gets a 3ML model (a Model instance) from the table above, where every source
    # in the 3FGL becomes a Source instance. Note that by default all parameters of all
    # sources are fixed

    model = lat_catalog.get_model()

    assert model.get_number_of_point_sources() == 147

    # Let's free all the normalizations within 3 deg from the center
    model.free_point_sources_within_radius(3.0, normalization_only=True)

    assert len(model.free_parameters) == 5

    # but then let's fix the sync and the IC components of the Crab
    # (cannot fit them with just one day of data)
    # (these two methods are equivalent)
    model["Crab_IC.spectrum.main.Log_parabola.K"].fix = True
    model.Crab_synch.spectrum.main.shape.K.fix = True

    assert len(model.free_parameters) == 3

    # However, let's free the index of the Crab
    model.PSR_J0534p2200.spectrum.main.Super_cutoff_powerlaw.index.free = True

    assert len(model.free_parameters) == 4

    # Download data from Jan 01 2010 to Jan 2 2010

    tstart = "2010-01-01 00:00:00"
    tstop = "2010-01-08 00:00:00"

    # Note that this will understand if you already download these files, and will
    # not do it twice unless you change your selection or the outdir

    evfile, scfile = download_LAT_data(
        ra,
        dec,
        20.0,
        tstart,
        tstop,
        time_type="Gregorian",
        destination_directory="Crab_data",
    )

    # Configuration for Fermipy

    config = FermipyLike.get_basic_config(evfile=evfile, scfile=scfile, ra=ra, dec=dec)

    # Let's create an instance of the plugin
    # Note that here no processing is made, because fermipy still doesn't know
    # about the model you want to use

    LAT = FermipyLike("LAT", config)

    # The plugin modifies the configuration as needed to get the output files
    # in a unique place, which will stay the same as long as your selection does not change
    config.display()

    data = DataList(LAT)

    # Here is where the fermipy processing happens (the .setup method)
    jl = JointLikelihood(model, data)

    res = jl.fit()
