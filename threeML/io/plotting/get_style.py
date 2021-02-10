import matplotlib.pyplot as plt

from threeML.io.package_data import get_path_of_data_file
from threeML.config import threeML_config

_submenu = threeML_config.plotting

def set_threeML_style() -> None:
    plt.style.use(str(get_path_of_data_file(_submenu.mplstyle)))


def get_threeML_style() -> str:

    return str(get_path_of_data_file(_submenu.mplstyle))
