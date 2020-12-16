import os
import shutil
import tempfile
import uuid
from builtins import str
from contextlib import contextmanager
from pathlib import Path

from threeML.exceptions.custom_exceptions import custom_warnings


def sanitize_filename(filename, abspath=False) -> Path:

    path: Path = Path(filename)

    sanitized = path.expanduser()

    if abspath:

        return sanitized.absolute()

    else:

        return sanitized


def file_existing_and_readable(filename) -> bool:

    sanitized_filename: Path = sanitize_filename(filename)

    return sanitized_filename.is_file()


def path_exists_and_is_directory(path) -> bool:

    sanitized_path: Path = sanitize_filename(path, abspath=True)

    return sanitized_path.is_dir()


def if_directory_not_existing_then_make(directory) -> None:
    """
    If the given directory does not exists, then make it

    :param directory: directory to check or make
    :return: None
    """

    sanitized_directory: Path = sanitize_filename(directory)

    try:

        sanitized_directory.mkdir(parents=True, exist_ok=False)

    except (FileExistsError):

        # should add logging here!

        pass


def get_random_unique_name():
    """
    Returns a name which is random and (with extremely high probability) unique

    :return: random file name
    """

    return str(uuid.uuid4().hex)


@contextmanager
def temporary_directory(prefix="", within_directory=None):
    """
    This context manager creates a temporary directory in the most secure possible way (with no race condition), and
    removes it at the end.

    :param prefix: the directory name will start with this prefix, if specified
    :param within_directory: create within a specific directory (assumed to exist). Otherwise, it will be created in the
    default system temp directory (/tmp in unix)
    :return: the absolute pathname of the provided directory
    """

    directory = tempfile.mkdtemp(prefix=prefix, dir=within_directory)

    yield directory

    try:

        shutil.rmtree(directory)

    except:

        custom_warnings.warn("Couldn't remove temporary directory %s" % directory)


@contextmanager
def within_directory(directory):

    current_dir = os.getcwd()

    if not os.path.exists(directory):

        raise IOError("Directory %s does not exists!" % os.path.abspath(directory))

    try:
        os.chdir(directory)

    except OSError:

        raise IOError("Cannot access %s" % os.path.abspath(directory))

    yield

    os.chdir(current_dir)
