from collections import OrderedDict

from asciitree import LeftAligned
from asciitree.drawing import BOX_BLANK, BOX_DOUBLE, BoxStyle
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from typing import Dict, Any
from threeML.io.package_data import get_path_of_user_config

from .config_structure import Config

# Read the default Config
threeML_config = OmegaConf.structured(Config)

# now glob the config directory


def red(x) -> str:

    return f"\x1b[31;1m{x}\x1b[0m"


def blue(x) -> str:

    return f"\x1b[34;1m{x}\x1b[0m"


def green(x) -> str:

    return f"\x1b[32;1m{x}\x1b[0m"


for user_config_file in get_path_of_user_config().glob("*.yml"):

    _partial_conf = OmegaConf.load(user_config_file)

    threeML_config = OmegaConf.merge(threeML_config, _partial_conf)


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


def show_configuration():

    tr = LeftAligned(draw=BoxStyle(gfx=BOX_DOUBLE, horiz_len=1))

    out_final = {}
    out_final["CONFIG"] = _to_dict(threeML_config)
    print(tr(out_final))
