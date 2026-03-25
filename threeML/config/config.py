import os
from pathlib import Path

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

    if not config_path.exists():
        config_path.mkdir(parents=True)

    return config_path


for user_config_file in get_path_of_user_config().glob("*.yml"):
    _partial_conf = OmegaConf.load(user_config_file)

    threeML_config: Config = OmegaConf.merge(threeML_config, _partial_conf)
