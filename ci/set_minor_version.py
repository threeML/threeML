#!/usr/bin/env python

# This sets the minor version inside threeML/version.py

import sys
import re
import os
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Set the patch number in the version file")
    parser.add_argument("--patch", help="New patch number", required=True, type=int)
    parser.add_argument("--version_file", help="Path of the version file", required=True, type=str)

    args = parser.parse_args()

    # Sanitize file name
    file_path = os.path.abspath(os.path.expandvars(os.path.expanduser(args.version_file)))

    # Read current content

    assert os.path.exists(file_path), "File %s does not exist!" % file_path

    with open(file_path, "r") as f:

        lines = f.readlines()

    major = None
    minor = None
    patch = None

    line_number = None

    for i, line in enumerate(lines):

        # Look for the version. A typical line is:
        # __version__ = '0.3.2'

        match = re.match("__version__.*=.*([0-9]+)\.([0-9]+)\.([0-9]+).*", line)

        if match is not None:

            groups = match.groups()

            assert len(groups) == 3

            major, minor, patch = match.groups()

            line_number = int(i)

    if line_number is None:

        raise RuntimeError("Could not understand version in file %s" % file_path)

    # Update patch version

    lines[line_number] = "__version__ = '%s.%s.%s'\n" % (major, minor, args.patch)

    # Overwrite the file
    with open(file_path, "w+") as f:

        f.writelines(lines)

