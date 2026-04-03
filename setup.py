#!/usr/bin/env python

import os

from setuptools import setup

import versioneer

# This dynamically loads a module and return it in a variable.
# Will use it for check optional dependencies


def is_module_available(module_name):
    # Fast path: see if the module has already been imported.

    try:
        exec("import %s" % module_name)

    except ImportError:
        return False

    else:
        return True


# Create list of data files
def find_data_files(directory):
    paths = []

    for path, directories, filenames in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join("..", path, filename))

    return paths


extra_files = find_data_files("threeML/data")

# This list will contain the messages to print just before the end of the setup
# so that the user actually note them, instead of loosing them in the tons of
# messages of the build process

setup(
    packages=[
        "threeML",
        "threeML/exceptions",
        "threeML/bayesian",
        "threeML/minimizer",
        "threeML/utils",
        "threeML/utils/OGIP",
        "threeML/utils/spectrum",
        "threeML/utils/polarization",
        "threeML/utils/photometry",
        "threeML/utils/time_series",
        "threeML/utils/data_builders",
        "threeML/utils/data_builders/fermi",
        "threeML/utils/data_download",
        "threeML/utils/data_download/Fermi_LAT",
        "threeML/utils/data_download/Fermi_GBM",
        "threeML/utils/fitted_objects",
        "threeML/utils/statistics",
        "threeML/plugins",
        "threeML/classicMLE",
        "threeML/catalogs",
        "threeML/io",
        "threeML/io/plotting",
        "threeML/io/cern_root_utils",
        "threeML/parallel",
        "threeML/config",
        "threeML/test",
        "threeML/plugins/experimental",
    ],
    cmdclass=versioneer.get_cmdclass(),
    version=versioneer.get_version(),
    license="BSD-3",
    keywords=[
        "Likelihood",
        "Multi-mission",
        "3ML",
        "HAWC",
        "Fermi",
        "HESS",
        "joint",
        "fit",
        "bayesian",
        "multi-wavelength",
    ],
    # NOTE: we use '' as package name because the extra_files already contain the full
    # path from here
    package_data={
        "threeML/": [
            "data/*",
        ],
    },
    #     package_data={
    #         "": extra_files,
    #     },
)  # End of setup()
