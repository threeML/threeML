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

trigger_time = 243216766
ra           = 119.84717
dec          = -56.638333
radius       = 10.0
irf = 'p8_transient020e'
datarepository = 'FermiData'

@skip_if_internet_is_not_available
#@pytest.mark.xfail
@skip_if_LAT_is_not_available
def test_make_LAT_dataset():


    grb_name = make_LAT_dataset(
                                ra,
                                dec,
                                radius = radius+10,
                                trigger_time=trigger_time,
                                tstart=-10,
                                tstop =100,
                                data_type="Extended",
                                destination_directory=datarepository,
                                Emin=30.,
                                Emax= 1000000.)

    #from threeML.utils.data_builders.fermi.lat_transient_builder import TransientLATDataBuilder
    analysis_builder = TransientLATDataBuilder(grb_name,
                                               outfile=grb_name,
                                               roi=radius,
                                               tstarts='0',
                                               tstops = '100',
                                               irf=irf,
                                               galactic_model='template',
                                               particle_model='isotr template',
                                               datarepository=datarepository)
    analysis_builder.display()
    analysis_builder.run()

if __name__=='__main__':
    test_make_LAT_dataset()