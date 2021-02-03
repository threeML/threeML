import matplotlib.pyplot as plt

from threeML.io.package_data import get_path_of_data_file


def set_threeML_style() -> None:
    plt.style.use(str(get_path_of_data_file("threeml.mplstyle")))


def get_threeML_style() -> str:

    return str(get_path_of_data_file("threeml.mplstyle"))
