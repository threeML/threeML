import os
from pathlib import Path

from importlib import resources


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


def get_path_of_user_config() -> Path:
    if os.environ.get("THREEML_CONFIG") is not None:
        config_path: Path = Path(os.environ.get("THREEML_CONFIG"))

    else:
        config_path: Path = Path().home() / ".config" / "threeML"

    if not config_path.exists():
        config_path.mkdir(parents=True)

    return config_path


__all__ = [
    "get_path_of_data_file",
    "get_path_of_data_dir",
    "get_path_of_user_dir",
    "get_path_of_user_config",
]
