from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional

from asciitree import LeftAligned
from asciitree.drawing import BOX_DOUBLE, BoxStyle
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

from threeML.io.package_data import get_path_of_user_config

from .config import threeML_config


def red(x) -> str:

    return f"\x1b[31;1m{x}\x1b[0m"


def blue(x) -> str:

    return f"\x1b[34;1m{x}\x1b[0m"


def green(x) -> str:

    return f"\x1b[32;1m{x}\x1b[0m"


def _to_dict(conf):

    if not isinstance(conf, DictConfig):

        dummy: Dict[str, Dict[str, Any]] = {}
        dummy[red(f"{conf}")] = {}

        return dummy

    else:

        out: Dict[str, Dict[str, Any]] = {}

        for k, v in conf.items():

            if not isinstance(v, DictConfig):

                text = blue(f"{k}")

            else:

                text = green(f"{k}")

            out[text] = _to_dict(v)

        return out


def show_configuration(sub_menu: Optional[str] = None):
    """
    display the current configuration or a sub menu if
    provided
    """

    tr = LeftAligned(draw=BoxStyle(gfx=BOX_DOUBLE, horiz_len=1))

    out_final = {}

    if sub_menu is None:

        out_final["CONFIG"] = _to_dict(threeML_config)

    else:

        assert sub_menu in threeML_config, "not a valild topic"

        out_final[sub_menu] = _to_dict(threeML_config[sub_menu])

    print(tr(out_final))


def get_current_configuration_copy(file_name: str = "threeML_config.yml", overwrite: bool = False):
    """
    write a copy of the CURRENT configuration to the config directory
    """

    outfile: Path = get_path_of_user_config() / file_name

    if outfile.exists() and (not overwrite):

        raise RuntimeError(f"{outfile} exists! Set overwrite to True")

    else:

        _read_only_keys = ["LAT", "GBM", "catalogs"]

        _valid_keys = []

        for k, v in threeML_config.items():
            if k not in _read_only_keys:

                _valid_keys.append(k)

        config_copy = OmegaConf.masked_copy(threeML_config, _valid_keys)

        with outfile.open("w") as f:

            f.write(OmegaConf.to_yaml(config_copy, sort_keys=True, resolve=True))
