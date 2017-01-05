import os
from threeML.io.package_data import get_path_of_data_file


def test_get_package_data():
    # Try and get the config file
    config_file = get_path_of_data_file("threeML_config.yml")

    assert os.path.exists(config_file)
