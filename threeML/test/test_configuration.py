from pathlib import Path
import os

import pytest
import yaml
from omegaconf import OmegaConf
from omegaconf.errors import ReadonlyConfigError

from threeML.config import (
    get_current_configuration_copy,
    show_configuration,
    update_config_with_user_configs,
)
from threeML.config.config_structure import Config
from threeML.config.config import get_path_of_user_config


def test_default_configuration():
    # We just need to instance the Config class, as it contains in itself the check for
    # a valid default configuration file (it will raise an exception if the file is not
    # valid)

    _ = Config()

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


def test_user_configuration(tmp_path):
    original_config_path = os.environ.get("THREEML_CONFIG")
    os.environ["THREEML_CONFIG"] = str(tmp_path)
    try:

        dummy_config = OmegaConf.structured(Config)

        configs = [
            {"logging": {"usr": "off", "startup_warnings": "off"}},
            {"parallel": {"profile_name": "test"}},
        ]

        for i, c in enumerate(configs):
            path = tmp_path / f"conf_{i}.yml"

            with path.open("w") as f:
                yaml.dump(stream=f, data=c, Dumper=yaml.SafeDumper)

        dummy_config = update_config_with_user_configs(dummy_config)
    except Exception as e:
        raise e
    finally:
        original_config_path = (
            "" if original_config_path is None else original_config_path
        )
        os.environ["THREEML_CONFIG"] = original_config_path


def test_frozen_config():
    # make sure we cannot overwrite HARD CODED things

    dummy_config = OmegaConf.structured(Config)

    with pytest.raises(ReadonlyConfigError):
        dummy_config.LAT.public_ftp_location = 4
