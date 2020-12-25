from builtins import range
import os
import numpy as np
import pytest
from threeML.io.file_utils import within_directory
from threeML.utils.time_interval import TimeIntervalSet
from threeML.utils.time_series.event_list import EventListWithDeadTime, EventList
from threeML.utils.data_builders.time_series_builder import TimeSeriesBuilder
from threeML.io.file_utils import within_directory
from threeML.plugins.DispersionSpectrumLike import DispersionSpectrumLike
from threeML.plugins.OGIPLike import OGIPLike
from .conftest import get_test_datasets_directory
import astropy.io.fits as fits

datasets_directory = get_test_datasets_directory()


def test_event_list_constructor():
    dummy_times = np.linspace(0, 10, 10)
    dummy_energy = np.zeros_like(dummy_times)
    start = 0
    stop = 10

    evt_list = EventList(
        arrival_times=dummy_times,
        measurement=dummy_energy,
        n_channels=1,
        start_time=start,
        stop_time=stop,
    )

    # should only have 10 events

    assert evt_list.n_events == 10

    with pytest.raises(RuntimeError):
        evt_list.bins

    with pytest.raises(AttributeError):
        evt_list.text_bins

    assert evt_list.poly_intervals is None

    with pytest.raises(AttributeError):
        evt_list.tmax_list

    with pytest.raises(AttributeError):
        evt_list.tmin_list

    assert evt_list.polynomials is None

    assert evt_list._instrument == "UNKNOWN"

    assert evt_list._mission == "UNKNOWN"


def test_unbinned_fit():
    with within_directory(datasets_directory):
        start, stop = 0, 50

        poly = [1]

        arrival_times = np.loadtxt("test_event_data.txt")

        evt_list = EventListWithDeadTime(
            arrival_times=arrival_times,
            measurement=np.zeros_like(arrival_times),
            n_channels=1,
            start_time=arrival_times[0],
            stop_time=arrival_times[-1],
            dead_time=np.zeros_like(arrival_times),
        )

        evt_list.set_polynomial_fit_interval(
            "%f-%f" % (start + 1, stop - 1), unbinned=True, bayes=False
        )

        results = evt_list.get_poly_info()["coefficients"]

        evt_list.set_active_time_intervals("0-1")

        assert evt_list.time_intervals == TimeIntervalSet.from_list_of_edges([0, 1])

        assert evt_list._poly_counts.sum() > 0

        evt_list.__repr__()


def test_binned_fit():
    with within_directory(datasets_directory):
        start, stop = 0, 50

        poly = [1]

        arrival_times = np.loadtxt("test_event_data.txt")

        evt_list = EventListWithDeadTime(
            arrival_times=arrival_times,
            measurement=np.zeros_like(arrival_times),
            n_channels=1,
            start_time=arrival_times[0],
            stop_time=arrival_times[-1],
            dead_time=np.zeros_like(arrival_times),
        )

        evt_list.set_polynomial_fit_interval(
            "%f-%f" % (start + 1, stop - 1), unbinned=False
        )

        evt_list.set_active_time_intervals("0-1")

        results = evt_list.get_poly_info()["coefficients"]

        assert evt_list.time_intervals == TimeIntervalSet.from_list_of_edges([0, 1])

        assert evt_list._poly_counts.sum() > 0

        evt_list.__repr__()


def test_read_gbm_cspec():
    with within_directory(datasets_directory):
        data_dir = os.path.join("gbm", "bn080916009")

        nai3 = TimeSeriesBuilder.from_gbm_cspec_or_ctime(
            "NAI3",
            os.path.join(data_dir, "glg_cspec_n3_bn080916009_v01.pha"),
            rsp_file=os.path.join(data_dir, "glg_cspec_n3_bn080916009_v00.rsp2"),
            poly_order=1,
        )

        nai3.set_active_time_interval("0-1")

        assert not nai3.time_series.poly_fit_exists

        assert nai3.time_series.binned_spectrum_set.n_channels > 0

        nai3.set_background_interval("-20--10", "100-200")

        assert nai3.time_series.poly_fit_exists

        speclike = nai3.to_spectrumlike()

        assert isinstance(speclike, DispersionSpectrumLike)

        assert not speclike.background_spectrum.is_poisson

        speclike = nai3.to_spectrumlike(extract_measured_background=True)

        assert isinstance(speclike, DispersionSpectrumLike)

        assert speclike.background_spectrum.is_poisson

        nai3.write_pha_from_binner("test_from_nai3", start=0, stop=2, overwrite=True)


def test_read_gbm_tte():
    with within_directory(datasets_directory):
        data_dir = os.path.join("gbm", "bn080916009")

        nai3 = TimeSeriesBuilder.from_gbm_tte(
            "NAI3",
            os.path.join(data_dir, "glg_tte_n3_bn080916009_v01.fit.gz"),
            rsp_file=os.path.join(data_dir, "glg_cspec_n3_bn080916009_v00.rsp2"),
            poly_order=-1,
        )

        nai3.set_active_time_interval("0-1")
        nai3.set_background_interval("-20--10", "100-200", unbinned=False)

        speclike = nai3.to_spectrumlike()

        assert isinstance(speclike, DispersionSpectrumLike)

        assert not speclike.background_spectrum.is_poisson

        speclike = nai3.to_spectrumlike(extract_measured_background=True)

        assert isinstance(speclike, DispersionSpectrumLike)

        assert speclike.background_spectrum.is_poisson

        # test binning

        # should not have bins yet

        with pytest.raises(RuntimeError):
            nai3.bins

        # First catch the errors

        # This is without specifying the correct options name

        with pytest.raises(RuntimeError):
            nai3.create_time_bins(start=0, stop=10, method="constant")

        with pytest.raises(RuntimeError):
            nai3.create_time_bins(start=0, stop=10, method="significance")

        with pytest.raises(RuntimeError):
            nai3.create_time_bins(start=0, stop=10, method="constant", p0=0.1)

        with pytest.raises(RuntimeError):
            nai3.create_time_bins(start=0, stop=10, method="significance", dt=1)

        # now incorrect options

        with pytest.raises(RuntimeError):
            nai3.create_time_bins(start=0, stop=10, method="not_a_method")

        # Now test values

        nai3.create_time_bins(start=0, stop=10, method="constant", dt=1)

        assert len(nai3.bins) == 10

        assert nai3.bins.argsort() == list(range(len(nai3.bins)))

        nai3.create_time_bins(start=0, stop=10, method="bayesblocks", p0=0.1)

        assert nai3.bins.argsort() == list(range(len(nai3.bins)))

        assert len(nai3.bins) == 5

        nai3.create_time_bins(start=0, stop=10, method="significance", sigma=40)

        assert nai3.bins.argsort() == list(range(len(nai3.bins)))

        assert len(nai3.bins) == 5

        nai3.view_lightcurve(use_binner=True)

        nai3.write_pha_from_binner("test_from_nai3", overwrite=True, force_rsp_write=True)


def test_reading_of_written_pha():
    with within_directory(datasets_directory):
        # check the number of items written

        with fits.open("test_from_nai3.rsp") as f:
            # 2 ext + 5 rsp ext
            assert len(f) == 7

        # make sure we can read spectrum number

        _ = OGIPLike("test", observation="test_from_nai3.pha", spectrum_number=1)
        _ = OGIPLike("test", observation="test_from_nai3.pha", spectrum_number=2)

        os.remove("test_from_nai3.pha")


def test_read_lle():
    with within_directory(datasets_directory):
        data_dir = "lat"

        lle = TimeSeriesBuilder.from_lat_lle(
            "lle",
            os.path.join(data_dir, "gll_lle_bn080916009_v10.fit"),
            os.path.join(data_dir, "gll_pt_bn080916009_v10.fit"),
            rsp_file=os.path.join(data_dir, "gll_cspec_bn080916009_v10.rsp"),
            poly_order=-1,
        )

        lle.view_lightcurve()

        lle.set_active_time_interval("0-10")

        lle.set_background_interval("-150-0", "100-250", unbinned=True)

        speclike = lle.to_spectrumlike()

        assert isinstance(speclike, DispersionSpectrumLike)

        # will test background with lle data

        old_coefficients, old_errors = lle.get_background_parameters()

        old_tmin_list = lle._time_series.poly_intervals

        lle.save_background("temp_lle", overwrite=True)

        lle = TimeSeriesBuilder.from_lat_lle(
            "lle",
            os.path.join(data_dir, "gll_lle_bn080916009_v10.fit"),
            os.path.join(data_dir, "gll_pt_bn080916009_v10.fit"),
            rsp_file=os.path.join(data_dir, "gll_cspec_bn080916009_v10.rsp"),
            restore_background="temp_lle.h5",
        )

        new_coefficients, new_errors = lle.get_background_parameters()

        new_tmin_list = lle._time_series.poly_intervals

        assert new_coefficients == old_coefficients

        assert new_errors == old_errors

        assert old_tmin_list == new_tmin_list
