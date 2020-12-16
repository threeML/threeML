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
    return Path("~/.threeML").expanduser()


def get_path_of_log_dir() -> Path:

    return get_path_of_user_dir() / "log"


_log_file_names = ["usr.log", "dev.log"]


def get_path_of_log_file(log_file: str) -> Path:
    """
    returns the path of the log files
    """
    assert log_file in _log_file_names, f"{log_file} is not on of {_log_file_names}"

    return get_path_of_log_dir() / log_file
