import os
from pathlib import Path
import warnings

from omegaconf import OmegaConf
from .config_structure import Config

# Read the default Config
threeML_config: Config = OmegaConf.structured(Config)

# now glob the config directory


def get_path_of_user_config() -> Path:
    if os.environ.get("THREEML_CONFIG") is not None:
        config_path: Path = Path(os.environ.get("THREEML_CONFIG"))

    else:
        config_path: Path = Path().home() / ".config" / "threeML"

    config_path.mkdir(parents=True, exist_ok=True)

    return config_path


def update_config_with_user_configs(threeML_config):
    for user_config_file in get_path_of_user_config().glob("*.yml"):
        _partial_conf = OmegaConf.load(user_config_file)
        if "logging" in _partial_conf.keys():
            if "startup_warnings" in _partial_conf["logging"].keys():
                warnings.warn(
                    "You've provided 'logging.startup_warnings' in "
                    + str(user_config_file)
                    + ". "
                    + "This is deprecated since v2.6.0 - will ignore it"
                )
                del _partial_conf.logging.startup_warnings

        threeML_config: Config = OmegaConf.merge(threeML_config, _partial_conf)
    return threeML_config


threeML_config = update_config_with_user_configs(threeML_config)
