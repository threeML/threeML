import shutil
import pytest

from threeML import *
from threeML.io.network import internet_connection_is_active
from threeML.exceptions.custom_exceptions import TriggerDoesNotExist


skip_if_internet_is_not_available = pytest.mark.skipif(
    not internet_connection_is_active(), reason="No active internet connection"
)

try:

    import GtApp

except ImportError:

    has_Fermi = False

else:

    has_Fermi = True

# This defines a decorator which can be applied to single tests to
# skip them if the condition is not met
skip_if_LAT_is_not_available = pytest.mark.skipif(not has_Fermi,
    reason="Fermi Science Tools not installed",
)


@skip_if_internet_is_not_available
@pytest.mark.xfail
@skip_if_LAT_is_not_available
def test_download_LAT_data():
    # Crab
    ra = 83.6331
    dec = 22.0199
    tstart = "2010-01-01 00:00:00"
    tstop = "2010-01-02 00:00:00"

    temp_dir = "_download_temp"

    ft1, ft2 = download_LAT_data(
        ra,
        dec,
        20.0,
        tstart,
        tstop,
        time_type="Gregorian",
        destination_directory=temp_dir,
    )

    assert os.path.exists(ft1)
    assert os.path.exists(ft2)

    shutil.rmtree(temp_dir)


@skip_if_internet_is_not_available
@pytest.mark.xfail
def test_download_LLE_data():
    # test good trigger names
    good_triggers = ["080916009", "bn080916009", "GRB080916009"]

    temp_dir = "_download_temp"

    for i, trigger in enumerate(good_triggers):

        dl_info = download_LLE_trigger_data(
            trigger_name=trigger, destination_directory=temp_dir
        )

        assert os.path.exists(dl_info["rsp"])
        assert os.path.exists(dl_info["lle"])

        # we can rely on the non-repeat download to go fast

        if i == len(good_triggers) - 1:
            shutil.rmtree(temp_dir)

    # Now test that bad names block us

    with pytest.raises(AssertionError):

        download_LLE_trigger_data(
            trigger_name="blah080916009", destination_directory=temp_dir
        )

    with pytest.raises(AssertionError):

        download_LLE_trigger_data(trigger_name=80916009, destination_directory=temp_dir)

    with pytest.raises(AssertionError):

        download_LLE_trigger_data(
            trigger_name="bn08a916009", destination_directory=temp_dir
        )

    with pytest.raises(TriggerDoesNotExist):

        download_LLE_trigger_data(
            trigger_name="080916008", destination_directory=temp_dir
        )
