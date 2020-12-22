import shutil
import pytest

from threeML import *
from threeML.io.network import internet_connection_is_active
from threeML.exceptions.custom_exceptions import TriggerDoesNotExist

skip_if_internet_is_not_available = pytest.mark.skipif(
    not internet_connection_is_active(), reason="No active internet connection"
)


@skip_if_internet_is_not_available
@pytest.mark.xfail
def test_download_GBM_data():
    # test good trigger names
    good_triggers = ["080916009", "bn080916009", "GRB080916009"]

    which_detector = "n1"

    for i, trigger in enumerate(good_triggers):

        temp_dir = "_download_temp"

        dl_info = download_GBM_trigger_data(
            trigger_name=trigger,
            detectors=[which_detector],
            destination_directory=temp_dir,
        )

        assert os.path.exists(dl_info[which_detector]["rsp"])
        assert os.path.exists(dl_info[which_detector]["tte"])

        # we can rely on the non-repeat download to go fast

        if i == len(good_triggers) - 1:
            shutil.rmtree(temp_dir)

    # Now test that bad names block us

    with pytest.raises(AssertionError):

        download_GBM_trigger_data(
            trigger_name="blah080916009", destination_directory=temp_dir
        )

    with pytest.raises(AssertionError):

        download_GBM_trigger_data(trigger_name=80916009, destination_directory=temp_dir)

    with pytest.raises(AssertionError):

        download_GBM_trigger_data(
            trigger_name="bn08a916009", destination_directory=temp_dir
        )

    with pytest.raises(TriggerDoesNotExist):

        download_GBM_trigger_data(
            trigger_name="080916008", destination_directory=temp_dir
        )

    # now test that bad detectors block us

    with pytest.raises(AssertionError):

        download_GBM_trigger_data(
            trigger_name="080916009", detectors="n1", destination_directory=temp_dir
        )

    with pytest.raises(AssertionError):

        download_GBM_trigger_data(
            trigger_name="080916009",
            detectors=["not_a_detector"],
            destination_directory=temp_dir,
        )
