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
    
    

def get_path_of_log_dir() -> Path:

    log_path: Path = get_path_of_user_dir() / "log"

    if not log_path.exists():

        log_path.mkdir()
    
    return log_path


_log_file_names = ["usr.log", "dev.log"]


def get_path_of_log_file(log_file: str) -> Path:
    """
    returns the path of the log files
    """
    assert log_file in _log_file_names, f"{log_file} is not on of {_log_file_names}"

    return get_path_of_log_dir() / log_file
