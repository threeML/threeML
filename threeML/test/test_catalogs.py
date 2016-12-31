import pytest

from threeML import *
from threeML.io.network import internet_connection_is_active

skip_if_internet_is_not_available = pytest.mark.skipif(not internet_connection_is_active(),
                                                       reason="No active internet connection")

from threeML.exceptions.custom_exceptions import InvalidUTC


@skip_if_internet_is_not_available
def test_gbm_catalog():

    gbm_catalog = FermiGBMBurstCatalog()

    _ = gbm_catalog.cone_search(0.0, 0.0, 5.0)

    assert gbm_catalog.ra_center == 0.0
    assert gbm_catalog.dec_center == 0.0

    gbm_catalog.search_around_source('Crab', 5.0)

    # test model building

    models = ['band', 'comp', 'plaw', 'sbpl']
    intervals = ['peak', 'fluence']

    param_dict = {'band': {
        'k'    : 'ampl',
        'ep'   : 'epeak',
        'alpha': 'alpha',
        'beta' : 'beta'},
        'comp'          : {
            'k'    : 'ampl',
            'ep'   : 'epeak',
            'index': 'index'},
        'plaw'          : {
            'k'    : 'ampl',
            'index': 'index'},
        'sbpl'          : {
            'k'           : 'ampl',
            'energy_break': 'brken',
            'energy_scale': 'brksc',
            'alpha'       : 'indx1',
            'beta'        : 'indx2'}}

    for model in models:
        for interval in intervals:
            _ = gbm_catalog.get_model(model=model, interval=interval)

            _ = gbm_catalog.search_energy_flux(flux_greater=1E-5, flux_less=1E-2, model=model, interval=interval)

            for param in param_dict[model].keys():

                _ = gbm_catalog.search_model_parameters(model=model,
                                                        parameter=param,
                                                        parameter_greater=-5,
                                                        parameter_less=100,
                                                        interval=interval)

    # test model building assertions

    with pytest.raises(AssertionError):

        _ = gbm_catalog.get_model(model='not_a_model')

    with pytest.raises(AssertionError):

        _ = gbm_catalog.get_model(interval='not_an_interval')

    _ = gbm_catalog.search_t90(t90_greater=2.)
    _ = gbm_catalog.search_t90(t90_less=2.)
    _ = gbm_catalog.search_t90(t90_greater=2., t90_less=10)

    with pytest.raises(AssertionError):
        _ = gbm_catalog.search_t90()

    with pytest.raises(AssertionError):
        _ = gbm_catalog.search_model_parameters()

    with pytest.raises(AssertionError):
        _ = gbm_catalog.search_energy_flux()

    with pytest.raises(AssertionError):
        _ = gbm_catalog.search_model_parameters(model='not_a_model', parameter_greater=0)

    with pytest.raises(AssertionError):
        _ = gbm_catalog.search_model_parameters(parameter='not_a_param', parameter_greater=0)

    _ = gbm_catalog.search_trigger_name('bn080916009')

    # test for invalid trigger names


    with pytest.raises(AssertionError):
        _ = gbm_catalog.search_trigger_name('080916009')

    with pytest.raises(AssertionError):
        _ = gbm_catalog.search_trigger_name('blah080916009')

    with pytest.raises(AssertionError):
        _ = gbm_catalog.search_trigger_name(80916009)

    with pytest.raises(AssertionError):
        _ = gbm_catalog.search_trigger_name('bn08a916009')

    # test time searches

    # UTC search includes MJD search
    utc_start = '2008-01-01T00:00:00.123456789'
    utc_stop = '2009-01-01T00:00:00.123456789'

    _ = gbm_catalog.search_utc(utc_start=utc_start, utc_stop=utc_stop)

    # make sure we cannot search reversed intervals
    with pytest.raises(AssertionError):

        _ = gbm_catalog.search_utc(utc_start=utc_stop, utc_stop=utc_start)

    # make sure that non UTC values throw a bug

    with pytest.raises(InvalidUTC):

        _ = gbm_catalog.search_utc(utc_start='123', utc_stop='123')


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
        _ = lle_catalog.search_trigger_name('blah080916009')

    with pytest.raises(AssertionError):
        _ = lle_catalog.search_trigger_name(80916009)

    with pytest.raises(AssertionError):
        _ = lle_catalog.search_trigger_name('bn08a916009')

        # test time searches

        # UTC search includes MJD search
        utc_start = '2008-01-01T00:00:00.123456789'
        utc_stop = '2009-01-01T00:00:00.123456789'

        _ = lle_catalog.search_utc(utc_start=utc_start, utc_stop=utc_stop)

        # make sure we cannot search reversed intervals
        with pytest.raises(AssertionError):
            _ = lle_catalog.search_utc(utc_start=utc_stop, utc_stop=utc_start)

        # make sure that non UTC values throw a bug

        with pytest.raises(InvalidUTC):
            _ = lle_catalog.search_utc(utc_start='123', utc_stop='123')



@skip_if_internet_is_not_available
def test_swift_catalog():

    swift_catalog = SwiftGRBCatalog()

    _ = swift_catalog.cone_search(0.0, 0.0, 15.0)

    _ = swift_catalog.get_other_instrument_information()

    _ = swift_catalog.get_other_observation_information()

    assert swift_catalog.ra_center == 0.0
    assert swift_catalog.dec_center == 0.0

    _ = swift_catalog.search_t90(t90_greater=2.)
    _ = swift_catalog.search_t90(t90_less=2.)
    _ = swift_catalog.search_t90(t90_greater=2., t90_less=10)

    with pytest.raises(AssertionError):
        _ = swift_catalog.search_t90()

    _ = swift_catalog.search_redshift(z_greater=2.)
    _ = swift_catalog.search_redshift(z_less=2.)
    _ = swift_catalog.search_redshift(z_greater=2., z_less=10)

    with pytest.raises(AssertionError):
        _ = swift_catalog.search_redshift()

    _ = swift_catalog.search_trigger_name('GRB 050525A')
    _ = swift_catalog.search_trigger_name('GRB050525A')


    # test for invalid trigger names

    with pytest.raises(AssertionError):
        _ = swift_catalog.search_trigger_name('GRB080916009')

    # UTC search includes MJD search
    utc_start = '2008-01-01T00:00:00.123456789'
    utc_stop = '2009-01-01T00:00:00.123456789'

    _ = swift_catalog.search_utc(utc_start=utc_start, utc_stop=utc_stop)

    # make sure we cannot search reversed intervals
    with pytest.raises(AssertionError):
        _ = swift_catalog.search_utc(utc_start=utc_stop, utc_stop=utc_start)

    # make sure that non UTC values throw a bug

    with pytest.raises(InvalidUTC):
        _ = swift_catalog.search_utc(utc_start='123', utc_stop='123')

    for mission in swift_catalog.other_observing_instruments:

        _ = swift_catalog.search_other_observing_instruments(mission)

    with pytest.raises(AssertionError):

        _ = swift_catalog.search_other_observing_instruments('not_a_mission')


    _ = swift_catalog.search_redshift(z_greater=6)

    _ = swift_catalog.get_other_instrument_information()

    _ = swift_catalog.get_other_observation_information()

    _ = swift_catalog.get_redshift()

    with pytest.raises(AssertionError):

        _ = swift_catalog.search_redshift()


