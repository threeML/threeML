from pathlib import Path

from threeML.io.file_utils import (
    file_existing_and_readable,
    if_directory_not_existing_then_make,
    path_exists_and_is_directory,
    sanitize_filename,
)


def test_sanatize():
    file_name = sanitize_filename("test.txt")

    assert isinstance(file_name, Path)

    file_name = sanitize_filename("test.txt", abspath=True)

    assert file_name.is_absolute()


def test_directory_check(test_directory, test_file):
    assert path_exists_and_is_directory(test_directory)

    assert not path_exists_and_is_directory("this_does_not_exist")

    assert not path_exists_and_is_directory(test_file)

    if_directory_not_existing_then_make(test_directory)


def test_file_check(test_directory, test_file):
    assert not file_existing_and_readable(test_directory)

    assert not file_existing_and_readable("this_does_not_exist")

    assert file_existing_and_readable(test_file)
