#!/usr/bin/env python

import os
import sys

from setuptools import setup, Extension

# Get the version number
execfile('threeML/version.py')

# Now a global __version__ is available

# This list will contain the messages to print just before the end of the setup
# so that the user actually note them, instead of loosing them in the tons of
# messages of the build process

final_messages = ["REMEMBER: if you want to use C/C++ plugins (HAWC) you have to install also cthreeML"]

setup(

    name="threeML",

    packages=['threeML',
              'threeML/exceptions',
              'threeML/bayesian',
              'threeML/minimizer',
              'threeML/models',
              'threeML/models/fluxModels',
              'threeML/models/spatialModels',
              'threeML/plugins',
              'threeML/classicMLE',
              'threeML/catalogs',
              'threeML/io',
              'threeML/utils',
              'threeML/parallel',
              'threeML/config'],

    version=__version__,

    description="The Multi-Mission Maximum Likelihood framework",

    long_description="3ML can be used for single or multi-instrument likelihood modeling or Bayesian inference",
    
    license='BSD-3',

    author='Giacomo Vianello',

    author_email='giacomo.vianello@gmail.com',

    url='https://github.com/giacomov/3ML',

    download_url='https://github.com/giacomov/3ML/archive/%s' % __version__,

    keywords=['Likelihood', 'Multi-mission', '3ML', 'HAWC', 'Fermi', 'HESS', 'joint', 'fit', 'bayesian',
              'multi-wavelength'],

    classifiers=[],

    # Install configuration file in user home and in the package repository

    data_files=[(os.path.join(os.path.expanduser('~'), '.threeML'), ["threeML/config/threeML_config.yml"]),
                ('threeML/config', ["threeML/config/threeML_config.yml"])
                ],

    install_requires=[
        'numpy >= 1.6',
        'scipy',
        'emcee',
        'astropy>=1.0.3',
        'matplotlib',
        'ipython',
        'uncertainties',
        'pyyaml',
        'dill',
        'parse',
        'iminuit'
    ])

# Now print the final messages if there are any

if len(final_messages) > 0:
    print("\n#############")
    print("FINAL NOTES:")
    print("#############")

    print("\n".join(final_messages))
