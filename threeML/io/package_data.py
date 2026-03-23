import os
from importlib import resources
from pathlib import Path

from threeML.config.config import get_path_of_user_config


def get_path_of_data_file(data_file) -> Path:
    """Used to get internal testing data and for examples. Not for user data.

    :param data_file: data file inside internal 3ML directory
    :type data_file:
    :returns:
    """

    data_file = Path(data_file)

    try:
        resource_path = resources.files("threeML").joinpath("data", *data_file.parts)

        if not resource_path.is_file():
            raise FileNotFoundError

    except Exception:
        raise IOError(
            f"Could not read or find data file {data_file}. "
            "Try reinstalling astromodels. "
            f"If this does not fix your problem, open an issue on github."
        )

    else:
        return Path(resource_path).resolve()


def get_path_of_data_dir() -> Path:
    """Used to get internal testing data and for examples. Not for user data.

    :returns:
    """

    file_path = resources.files("threeML").joinpath("data")

    return Path(file_path).resolve()


def get_path_of_user_dir() -> Path:
    """Returns the path of the directory containing the user data (~/.threeML)

    :return: an absolute path
    """
    user_dir: Path = Path().home() / ".threeML"

    if not user_dir.exists():
        user_dir.mkdir()

    return user_dir


def get_user_data_path():
    user_data = os.path.join(os.path.expanduser("~"), ".threeml", "data")

    # Create it if doesn't exist
    if os.path.exists(user_data):
        return user_data

    else:
        os.makedirs(user_data)

        return user_data


__all__ = [
    "get_path_of_data_file",
    "get_path_of_data_dir",
    "get_path_of_user_dir",
    "get_path_of_user_config",
]
