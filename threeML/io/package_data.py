import pkg_resources
import os


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
