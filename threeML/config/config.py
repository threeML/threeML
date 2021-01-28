from collections import OrderedDict
from typing import Any, Dict, Optional

from asciitree import LeftAligned
from asciitree.drawing import BOX_BLANK, BOX_DOUBLE, BoxStyle
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

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
