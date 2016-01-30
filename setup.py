#!/usr/bin/env python

# from distutils.core import setup
#from distutils.extension import Extension
import os, sys

from setuptools import setup, Extension

# This list will contain the messages to print just before the end of the setup
# so that the user actually note them, instead of loosing them in the tons of
# messages of the build process

final_messages = []

# Decide whether to build or not the C++ part
# I really wanted to do this in a cleaner way, but couldn't find a solution other than subclassing Command and hence
# recoding the whole build process!

copy_args = sys.argv[1:]

if '--with-boost' in copy_args:

    final_messages.append("Built the boost.python extension. You can now use C/C++ plugins.")

    build_boost = True

    copy_args.remove('--with-boost')

    # Probe whether the user has specified its own boost directory through the BOOSTROOT
    # environment variable

    boost_root = os.environ.get("BOOSTROOT")

    if boost_root:

        # The user want to override pre-defined location of boost

        print("\n\n **** Using boost.python from the env. variable $BOOSTROOT (%s)" % (boost_root))

        include_dirs = [os.path.join(boost_root, 'include')]
        library_dirs = [os.path.join(boost_root, 'lib')]

        final_messages.append("Used boost.python from the env. variable BOOSTROOT")
        final_messages.append("     Include dir: %s" % include_dirs)
        final_messages.append("     Library dir: %s" % library_dirs)

    else:

        include_dirs = []
        library_dirs = []

        final_messages.append("Using boost.python from the system path.")

    # Configure the variables to build the external module with the C/C++ wrapper

    ext_modules_configuration = [

        Extension("threeML.pyModelInterface",

                  ["threeML/models/pyToCppModelInterface.cxx",
                   "threeML/models/FixedPointSource.cxx",
                   "threeML/models/ModelInterface_boost.cxx"],

                  libraries=["boost_python"],

                  include_dirs=include_dirs,

                  library_dirs=library_dirs)]

    headers_configuration = ["threeML/models/ModelInterface.h",
                             "threeML/models/pyToCppModelInterface.h",
                             "threeML/models/FakePlugin.h",
                             "threeML/models/FixedPointSource.h"]


else:

    # No need to build the C/C++ wrappers, set up the ext_modules_configuration
    # and the headers configuration accordingly

    final_messages.append("- boost.python extension not built. If you didn't built it previously, you will not be able "
                          "to use C/C++ plugins.")

    build_boost = False

    ext_modules_configuration = None
    headers_configuration = None


setup(

    script_args = copy_args,

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

    version='v0.2.0',

    description="The Multi-Mission Maximum Likelihood framework",

    author='Giacomo Vianello',

    author_email='giacomo.vianello@gmail.com',

    url='https://github.com/giacomov/3ML',

    download_url='https://github.com/giacomov/3ML/archive/v0.2.0',

    keywords=['Likelihood', 'Multi-mission', '3ML', 'HAWC', 'Fermi', 'HESS','joint', 'fit','bayesian',
              'multi-wavelength'],

    classifiers = [],

    ext_modules = ext_modules_configuration,

    headers = headers_configuration,

    # Install configuration file in user home and in the package repository

    data_files=[(os.path.join(os.path.expanduser('~'), '.threeML'), ["threeML/config/threeML_config.yml"]),
                ('threeML/config', ["threeML/config/threeML_config.yml"])
                ],

    install_requires=[
        'numpy >= 1.6',
        'scipy',
        'numexpr',
        'emcee',
        'astropy==1.0.3',
        'matplotlib',
        'ipython >= 2.0.0, < 3.9.9',
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
