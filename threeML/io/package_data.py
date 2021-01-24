import os
from pathlib import Path

import pkg_resources


def get_path_of_data_file(data_file) -> Path:
    file_path = pkg_resources.resource_filename("threeML", "data/%s" % data_file)

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
    
    
