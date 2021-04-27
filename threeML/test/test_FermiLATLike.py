import shutil
import os
import pytest

from threeML import *
from threeML.io.network import internet_connection_is_active
from threeML.utils.data_download.Fermi_LAT.download_LAT_data import make_LAT_dataset

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


#@skip_if_internet_is_not_available
#@pytest.mark.xfail
#@skip_if_LAT_is_not_available
def test_make_LAT_dataset():

    trigger_time=243216766
    ra=119.84717
    dec=-56.638333


    make_LAT_dataset(
        ra,
        dec,
        radius = 10,
        trigger_time=trigger_time,
        tstart=-10,
        tstop =100,
        data_type="Extended",
        destination_directory=".",
        Emin=30.,
        Emax= 1000000.)

