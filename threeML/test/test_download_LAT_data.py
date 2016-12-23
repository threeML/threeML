import shutil
import pytest

from threeML import *
from threeML.io.network import internet_connection_is_active

skip_if_internet_is_not_available = pytest.mark.skipif(not internet_connection_is_active(),
                                                       reason="No active internet connection")


@skip_if_internet_is_not_available
def test_download_LAT_data():

    # Crab
    ra = 83.6331
    dec = 22.0199
    tstart = '2010-01-01 00:00:00'
    tstop = '2010-01-02 00:00:00'

    temp_dir = '_download_temp'

    ft1, ft2 = download_LAT_data(ra, dec, 20.0,
                                 tstart, tstop, time_type='Gregorian',
                                 destination_directory=temp_dir)

    assert os.path.exists(ft1)
    assert os.path.exists(ft2)

    shutil.rmtree(temp_dir)