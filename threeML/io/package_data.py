import os
from pathlib import Path

import pkg_resources

_custom_config_path = os.environ.get("THREEML_CONFIG")


def get_path_of_data_file(data_file) -> Path:
    file_path = pkg_resources.resource_filename(
        "threeML", "data/%s" % data_file)

    return Path(file_path)


def get_path_of_data_dir() -> Path:
    file_path = pkg_resources.resource_filename("threeML", "data")

    return Path(file_path)


def get_path_of_user_dir() -> Path:
    """
    Returns the path of the directory containing the user data (~/.threeML)

    :return: an absolute path
    """
    user_dir: Path = Path().home() / ".threeML"

    if not user_dir.exists():

        user_dir.mkdir()

    return user_dir


def get_path_of_user_config() -> Path:

    if _custom_config_path is not None:

        config_path: Path = Path(_custom_config_path)

    config_path: Path = Path().home() / ".config" / "threeML"

    if not config_path.exists():

        config_path.mkdir(parents=True)

    return config_path

__all__ = ["get_path_of_data_file",
           "get_path_of_data_dir",
           "get_path_of_user_dir",
           "get_path_of_user_config",
           ]
