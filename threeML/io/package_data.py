import pkg_resources
import os
from pathlib import Path

def get_path_of_data_file(data_file):
    file_path = pkg_resources.resource_filename("threeML", "data/%s" % data_file)

    return file_path


def get_path_of_data_dir():
    file_path = pkg_resources.resource_filename("threeML", "data")

    return file_path


def get_path_of_user_dir():
    """
    Returns the path of the directory containing the user data (~/.threeML)

    :return: an absolute path
    """

    return os.path.abspath(os.path.expanduser("~/.threeML"))


def get_path_of_log_dir() -> Path:

    return get_path_of_user_dir() / "log"


_log_file_names = ["usr.log", "dev.log"]

def get_path_of_log_file(log_file: str) -> Path:
    """
    returns the path of the log files
    """
    assert log_file in _log_file_names, f"{log_file} is not on of {_log_file_names}"

    return get_path_of_log_dir() / log_file
