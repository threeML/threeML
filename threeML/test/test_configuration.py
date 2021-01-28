from pathlib import Path

import pytest
import yaml
from omegaconf import OmegaConf
from omegaconf.errors import ReadonlyConfigError

from threeML.config import show_configuration, get_current_configuration_copy
from threeML.io.package_data import get_path_of_user_config
from threeML.config.config_structure import Config
from pathlib import Path



def test_default_configuration():

    # We just need to instance the Config class, as it contains in itself the check for a valid
    # default configuration file (it will raise an exception if the file is not valid)

    c = Config()

    show_configuration()


    show_configuration("LAT")


    with pytest.raises(AssertionError):

        show_configuration("doesnotexist")


    _file_name = "_tmp_config.yml"

    path = get_path_of_user_config() / _file_name


    get_current_configuration_copy(_file_name, overwrite=False)


    with pytest.raises(RuntimeError):

        get_current_configuration_copy(_file_name, overwrite=False)


    get_current_configuration_copy(_file_name, overwrite=True)


    path.unlink()

    


    
        
def test_user_configuration():

    dummy_config = OmegaConf.structured(Config)

    configs = [{"logging": {"usr": "off"}}, {
        "parallel": {"profile_name": "test"}}]

    for i, c in enumerate(configs):

        path = Path(f"conf_{i}.yml")

        with path.open("w") as f:

            yaml.dump(stream=f, data=c, Dumper=yaml.SafeDumper)

        cc = OmegaConf.load(path)

        dummy_config = OmegaConf.merge(dummy_config, cc)

        path.unlink()


    

def test_frozen_config():

    # make sure we cannot overwrite HARD CODED things

    dummy_config = OmegaConf.structured(Config)

    with pytest.raises(ReadonlyConfigError):
        dummy_config.LAT.public_ftp_location = 4
