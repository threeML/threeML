import pytest

from threeML import *
from threeML.io.network import internet_connection_is_active

skip_if_internet_is_not_available = pytest.mark.skipif(
    not internet_connection_is_active(), reason="No active internet connection"
)

@pytest.mark.xfail
@skip_if_internet_is_not_available
def test_gbm_catalog():

    gbm_catalog = FermiGBMBurstCatalog()

    _ = gbm_catalog.cone_search(0.0, 0.0, 300.0)

    assert gbm_catalog.ra_center == 0.0
    assert gbm_catalog.dec_center == 0.0

    gbm_catalog.search_around_source("Crab", 5.0)

    models = ["band", "comp", "plaw", "sbpl"]
    intervals = ["peak", "fluence"]

    for model in models:
        for interval in intervals:

            _ = gbm_catalog.get_model(model=model, interval=interval)

    gbm_catalog.query("t90 >2")

    # test model building assertions

    with pytest.raises(AssertionError):

        _ = gbm_catalog.get_model(model="not_a_model")

    with pytest.raises(AssertionError):

        _ = gbm_catalog.get_model(interval="not_an_interval")

    _ = gbm_catalog.query_sources("GRB080916009")


@pytest.mark.xfail
@skip_if_internet_is_not_available
def test_LAT_catalog():
    lat_catalog = FermiLATSourceCatalog()

    ra, dec, table1 = lat_catalog.search_around_source("Crab", 1.0)

    table2 = lat_catalog.cone_search(ra, dec, 1.0)

    assert len(table1) == len(table2)

    assert lat_catalog.ra_center == ra
    assert lat_catalog.dec_center == dec


@pytest.mark.xfail
@skip_if_internet_is_not_available
def test_LLE_catalog():
    lle_catalog = FermiLLEBurstCatalog()

    _ = lle_catalog.cone_search(0.0, 0.0, 300.0)

    assert lle_catalog.ra_center == 0.0
    assert lle_catalog.dec_center == 0.0

    lle_catalog.search_around_source("Crab", 5.0)

    _ = lle_catalog.query_sources("GRB080916009")

    _ = lle_catalog.query('trigger_type == "GRB"')


@pytest.mark.xfail
@skip_if_internet_is_not_available
def test_swift_catalog():

    swift_catalog = SwiftGRBCatalog()

    _ = swift_catalog.cone_search(0.0, 0.0, 15.0)

    _ = swift_catalog.get_other_instrument_information()

    _ = swift_catalog.get_other_observation_information()

    assert swift_catalog.ra_center == 0.0
    assert swift_catalog.dec_center == 0.0

    _ = swift_catalog.query("bat_t90 > 2")

    _ = swift_catalog.query_sources("GRB 050525A")

    for mission in swift_catalog.other_observing_instruments:

        _ = swift_catalog.query_other_observing_instruments(mission)

    with pytest.raises(AssertionError):

        _ = swift_catalog.query_other_observing_instruments("not_a_mission")

    _ = swift_catalog.get_other_instrument_information()

    _ = swift_catalog.get_other_observation_information()

    _ = swift_catalog.get_redshift()
