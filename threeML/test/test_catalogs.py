import pytest

from threeML import *
from threeML.io.network import internet_connection_is_active

skip_if_internet_is_not_available = pytest.mark.skipif(not internet_connection_is_active(),
                                                       reason="No active internet connection")
from threeML.catalogs.Fermi import InvalidTrigger

@skip_if_internet_is_not_available
def test_GBM_catalog():

    gbm_catalog = FermiGBMBurstCatalog()

    _ = gbm_catalog.cone_search(0.0, 0.0, 5.0)

    assert gbm_catalog.ra_center == 0.0
    assert gbm_catalog.dec_center == 0.0

    gbm_catalog.search_around_source('Crab', 5.0)

    _ = gbm_catalog.search_t90(t90_greater=2.)
    _ = gbm_catalog.search_t90(t90_less=2.)
    _ = gbm_catalog.search_t90(t90_greater=2., t90_less=10)

    with pytest.raises(AssertionError):
        _ = gbm_catalog.search_t90()

    _ = gbm_catalog.search_trigger_name('bn080916009')

    # test for invalid trigger names


    with pytest.raises(AssertionError):
        _ = gbm_catalog.search_trigger_name('080916009')

    with pytest.raises(AssertionError):
        _ = gbm_catalog.search_trigger_name('blah080916009')

    with pytest.raises(AssertionError):
        _ = gbm_catalog.search_trigger_name(80916009)

    with pytest.raises(InvalidTrigger):
        _ = gbm_catalog.search_trigger_name('bn08a916009')



@skip_if_internet_is_not_available
def test_LAT_catalog():

    lat_catalog = FermiLATSourceCatalog()

    ra, dec, table1 = lat_catalog.search_around_source('Crab', 1.0)

    table2 = lat_catalog.cone_search(ra, dec, 1.0)

    assert len(table1) == len(table2)

    assert lat_catalog.ra_center == ra
    assert lat_catalog.dec_center == dec


@skip_if_internet_is_not_available
def test_LLE_catalog():
    lle_catalog = FermiLLEBurstCatalog()

    _ = lle_catalog.cone_search(0.0, 0.0, 30.0)

    assert lle_catalog.ra_center == 0.0
    assert lle_catalog.dec_center == 0.0

    lle_catalog.search_around_source('Crab', 5.0)

    _ = lle_catalog.search_trigger_name('bn080916009')

    # test for invalid trigger names


    with pytest.raises(AssertionError):
        _ = lle_catalog.search_trigger_name('080916009')

    with pytest.raises(AssertionError):
        _ = lle_catalog.search_trigger_name('blah080916009')

    with pytest.raises(AssertionError):
        _ = lle_catalog.search_trigger_name(80916009)

    with pytest.raises(InvalidTrigger):
        _ = lle_catalog.search_trigger_name('bn08a916009')
